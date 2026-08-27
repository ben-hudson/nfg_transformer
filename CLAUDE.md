# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Reference implementation of the ICLR 2024 paper
[NfgTransformer: Equivariant Representation Learning for Normal-form Games](https://openreview.net/forum?id=4YESQqIys7)
(Google DeepMind). JAX + [dm-haiku](https://github.com/google-deepmind/dm-haiku).

The package is deliberately small: it ships the *architecture* and the game/loss
utilities needed to exercise it. The training loop lives in
[run_experiment.ipynb](run_experiment.ipynb), not in the package.

## Commands

```bash
pip install -e .                            # install package + deps
python -m pytest nfg_transformer/*test.py   # full test suite
python -m pytest nfg_transformer/network_test.py -k test_payoff   # single test
python -m nfg_transformer.network_test      # tests are absltest, also runnable directly
```

Tests are written with `absl.testing.parameterized.TestCase` but are run under
pytest (per the README). There is no lint or build step configured.

### JAX version

`requirements.txt` pins `jax<0.6.0`. Do not relax this without porting the code —
three APIs constrain it (verified against the JAX changelog and `jax/sharding.py`
at the relevant release tags):

| API used | Location | Deprecated | **Removed** |
| --- | --- | --- | --- |
| `jax.tree_map` | [equilibria.py:160](nfg_transformer/equilibria.py#L160) | 0.4.25 | **0.6.0** ← binding |
| `jax.sharding.PositionalSharding` | [games.py:211](nfg_transformer/games.py#L211) | 0.6.0 | 0.7.0 |
| `jnp.clip(..., a_min=)` | [games.py:34](nfg_transformer/games.py#L34), [games.py:67](nfg_transformer/games.py#L67) | 0.4.27 | 0.10.0 |

`jax.tree_map` is the tight one — on JAX ≥ 0.6.0 `equilibria.nash_approx` raises
`AttributeError`, breaking the NE objective and `equilibria_test.py`.

`jax==0.4.25` (Feb 2024, contemporary with the paper) is the newest release where
none of the three even warns; `a_min` deprecation warnings start at 0.4.27. The
other pinned-era deps are `chex==0.1.86` and `dm-haiku==0.0.12`, though these are
left unpinned.

**Python must be 3.10–3.12.** The bound is `jaxlib`, not `jax`'s `python_requires`:

- `jax==0.4.25` declares `>=3.9`, but `jaxlib` 0.4.25 ships wheels only for
  cp39–cp312 — **there is no cp313 wheel**, so Python 3.13 cannot install it.
- `jax` 0.5.x raises `python_requires` to `>=3.10`, ruling out 3.9.

3.10–3.12 is therefore the intersection that works across the whole `jax<0.6`
range. There is no `.python-version` in the repo; set one with pyenv if you hit this.

To modernise instead, the first two fixes are mechanical (`jax.tree_util.tree_map`;
positional `min`/`max`), but `PositionalSharding` → `NamedSharding` over a `Mesh`
is a real rewrite of `generate_payoffs`, not a rename.

## Architecture

### Tensor conventions

These are load-bearing and consistent across every module:

- **Payoffs**: `[N, T1, ..., TN]` — leading axis indexes the *player receiving*
  the payoff; the remaining `N` axes index the joint action. `payoffs[p]` is the
  full payoff tensor for player `p`.
- **Joint mask**: `[T1, ..., TN]` boolean, marking which joint actions are
  observed. Broadcast to payoff shape inside the network.
- **Action embeddings**: a `Sequence` of length `N`, entry `p` having shape
  `[Tp, E]`. Players may have different action counts, so this is a list, never
  a stacked array.
- Everything operates on a **single game**. Batching is the caller's job via
  `jax.vmap` (see how the notebook wraps `_loss_fn`).

### Encoder / decoder split

[network.py](nfg_transformer/network.py) is the whole model:

- `NfgTransformer` — the encoder. Payoffs (+ optional mask) → refined action
  embeddings. Initial embeddings default to **zeros**: no action or player index
  is ever embedded, which is what makes the representation equivariant to
  permutations of players and actions. Callers may pass
  `initial_action_embeddings` to seed it.
- `NfgPerAction` → one scalar per action per player (used for NE marginal logits).
- `NfgPerJoint` → one scalar per joint action (used for max-deviation-gain).
- `NfgPerPayoff` → payoff estimate per player per joint action (used for payoff
  reconstruction / inpainting).

Decoders consume only action embeddings, so swapping the task means swapping the
decoder head, not touching the encoder.

### Inside `NfgTransformerBlock`

Each block refines action embeddings through three attention stages, named after
their Haiku module scopes:

1. **`a2j` (action-to-joint)** — joint-action features are formed by tabulating
   the cross-product of per-player action embeddings (`_to_joint_actions`),
   concatenating the scalar payoff, and self-attending *across players* at each
   joint action.
2. **`a2p` (action-to-play)** — cross-attention where each of player `p`'s
   actions queries all joint actions it participates in. Implemented by moving
   axis `p` to the front and flattening the rest.
3. **`a2a` (action-to-action)** — self-attention over the concatenation of every
   action of every player (`sum(Tp)` tokens), repeated
   `num_self_attend_per_block` times.

### Masking semantics

Non-obvious and asserted by tests. A joint mask reduces to per-action masks via
`kv_mask.any(axis=-1)` — an action is "valid" if it participates in at least one
observed joint action. Invalid actions have their embeddings **restored to the
block's input values** at the end of each block
([network.py:400-408](nfg_transformer/network.py#L400-L408)); combined with the
zero initialisation this means fully-masked actions come out as zero embeddings.
This is defensive-by-design so downstream decoders can't silently consume garbage.
`test_mask_joint_actions` and `test_mask_player_actions` in
[network_test.py](nfg_transformer/network_test.py) pin this behaviour — changing
masking will break them.

### Supporting modules

- [games.py](nfg_transformer/games.py) — payoff tensor samplers. `l2_invariant`
  (centred, L2-normalised random payoffs — the canonical training distribution)
  and `empirical_disc_game` (win-rate games from latent vectors, with a
  symmetric random joint mask for the reconstruction task).
  `generate_payoffs()` returns a jitted batch generator that also applies
  `PositionalSharding` across local devices and threads the PRNG key through.
- [equilibria.py](nfg_transformer/equilibria.py) — objectives. `nash_approx`
  returns NashConv (max deviation gain under the product distribution implied by
  per-player marginals) and `max_deviation_gain` returns an MSE loss against the
  true per-joint-action max deviation gain. `_cce_gain_per_player` is the shared
  primitive; the `swapaxes` trick in `_make_cce_gain_swapaxis` computes deviation
  gains for all alternative actions at once.

### Where the pieces are wired together

`run_experiment.ipynb` (cell 2) holds the objective↔decoder↔loss mapping, the
optax optimiser (adaptive grad clipping + AdamW with a weight-decay mask that
excludes biases and norm params), and the update loop. If you add a task, that
notebook cell is the integration point.

## Style

Google Python style: 2-space indent, `'single quotes'` in the package modules
(the tests use `"double"`), Apache 2.0 header on every file, module-level
docstring, typed public signatures. Attention modules follow the pre-norm
ViT-22B recipe (bias-free QKV projections, RMSNorm on Q and K before softmax).
