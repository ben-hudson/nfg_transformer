# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# @title Implementation for supervised learning experiments for the NfgTransfomer.

import enum
import os
import re
from typing import Mapping
from typing import Sequence

from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import wandb

from nfg_transformer import equilibria
from nfg_transformer import games
from nfg_transformer import network

######
# OBJECTIVES
######


class Objective(enum.Enum):
    RECONSTRUCTION = enum.auto()
    NE = enum.auto()
    MAX_DEVIATION_GAIN = enum.auto()


def dist_to_normal_cone(operator, strategies):
    """Euclidean distance from ``operator`` to the normal cone of the probability simplex at
    ``strategies`` (actions on the last dim): the VI stationarity residual, zero exactly where
    ``strategies`` is a fixed point of the ascent dynamics ``z <- project(z + h operator(z))``.

    The cone at ``z`` is ``{u : u_i = lambda on the support, u_i <= lambda off it}``, so the
    squared distance is ``min`` over ``lambda`` of the support's squared deviations from ``lambda``
    plus the off-support upward deviations -- the same sort-and-threshold solve as
    ``project_onto_simplex``: with the coordinates ordered support-first then by operator value,
    the minimizing ``lambda`` is the mean of the largest self-consistent active prefix.
    """
    support = strategies > 0
    order = jnp.argsort(jnp.where(support, -jnp.inf, -operator), axis=-1)
    sorted_operator = jnp.take_along_axis(operator, order, axis=-1)

    positions = jnp.arange(1, operator.shape[-1] + 1)
    prefix_mean = jnp.cumsum(sorted_operator, axis=-1) / positions
    active = (positions <= support.sum(axis=-1, keepdims=True)) | (sorted_operator > prefix_mean)
    multiplier = jnp.take_along_axis(prefix_mean, active.sum(axis=-1, keepdims=True) - 1, axis=-1)

    deviation = jnp.where(support, operator - multiplier, jnp.maximum(operator - multiplier, 0.0))
    return jnp.sqrt(jnp.square(deviation).sum(axis=-1))


def _stationarity_residual(payoffs: jnp.ndarray, marginals: Sequence[jnp.ndarray]) -> jnp.ndarray:
    """Returns the distance to the normal cone at ``marginals``, as one scalar per game.

    Two-player games with equal action counts only. The operator is each player's
    expected payoff per action against their opponent's marginal; the two per-player
    residuals are combined as their L2 norm, which is the distance to the normal cone
    of the *product* of the players' simplices.
    """
    strategies = jnp.stack(marginals)  # [2, T]
    # Transposing player 1's payoffs puts each player's own action on the first axis.
    own_payoffs = jnp.stack([payoffs[0], payoffs[1].T])  # [2, T_own, T_opponent]
    operator = jnp.einsum("pab,pb->pa", own_payoffs, strategies[::-1])
    return jnp.linalg.norm(dist_to_normal_cone(operator, strategies))


######
# MODEL & LEARNING RULES
######

# Entries of the loss' `extra` that hold one scalar per game and are logged, per
# objective. Everything else in `extra` is a per-game tensor or a per-player
# sequence, which only the end-of-run inspection consumes.
_LOGGED_SCALARS = {
    Objective.NE: ("dist_to_normal_cone",),
    Objective.MAX_DEVIATION_GAIN: (),
    Objective.RECONSTRUCTION: ("inpainting_loss",),
}


def _joint_mask_to_action_masks(mask: jnp.ndarray) -> Sequence[jnp.ndarray]:
    action_masks = []
    for p in range(mask.ndim):
        not_p = [q for q in range(mask.ndim) if q != p]
        action_masks.append(jnp.any(mask, axis=not_p))
    return action_masks  # [N, T]


def make_model(
    objective: Objective,
    optim: optax.GradientTransformation,
    num_heads: int,
    num_action_channels: int,
    num_qkv_channels: int,
    num_blocks: int,
    num_self_attend_per_block: int,
):
    """Returns neural network model functions.

    Args:
      objective: the objective that the model is optimised for.
      optim: the optimiser used.
      num_heads: the number of attention heads.
      num_action_channels: the action embedding dimension.
      num_qkv_channels: the internal dimensionality of the query, key, value
        vectors.
      num_blocks: the number of NfgTransformerBlock.
      num_self_attend_per_block: the number of action-to-action self-attention
        layers in each NfgTransformerBlock.

    Returns:
      initial_params: a function that returns intialised parameters for the model.
      update: a function that returns updated parameters given the current
        parameters and data.
      evaluate: a function that run forward inference and loss computation for
        the (masked) input payoffs.
    """

    def _forward(payoffs, masks):
        """Encodes payoffs as action embeddings and decodes for an objective.

        Args:
          payoffs: a tensor of shape [N, T1, ..., TN] representing payoffs.
          masks: a tensor of shape [N, T1, ..., TN] indicating if each joint action
            is masked.

        Returns:
          outputs: a tree of arrays representing the decoded outputs.
        """
        # Encodes masked payoffs as action embeddings.
        actions = network.NfgTransformer(
            num_heads=num_heads,
            num_action_channels=num_action_channels,
            num_qkv_channels=num_qkv_channels,
            num_blocks=num_blocks,
            num_self_attend_per_block=num_self_attend_per_block,
        )(payoffs, masks)

        # Select an appropriate decoder architecture for an objective.
        if objective == Objective.NE:
            outputs = network.NfgPerAction(name=objective.name)(actions)
        elif objective == Objective.MAX_DEVIATION_GAIN:
            outputs = network.NfgPerJoint(name=objective.name)(actions)
        elif objective == Objective.RECONSTRUCTION:
            outputs = network.NfgPerPayoff(
                num_heads=num_heads,
                qk_channels=num_qkv_channels,
                v_channels=num_qkv_channels,
                name=objective.name,
            )(actions)
        else:
            raise ValueError(f"Unrecognised objective ({objective}).")

        return outputs

    def _loss_fn(payoffs, masks):
        """Returns loss and extra statistics from payoffs."""
        outputs = _forward(payoffs, masks)
        if objective == Objective.RECONSTRUCTION:
            inpainting_loss = jnp.mean(jnp.square(payoffs - outputs) * (1 - masks))
            loss = jnp.mean(jnp.square(payoffs - outputs))
            extra = dict(pred=outputs, inpainting_loss=inpainting_loss)
        elif objective == Objective.NE:
            logits = outputs
            action_mask = _joint_mask_to_action_masks(masks)
            logits = [jnp.where(m, l, -jnp.inf) for m, l in zip(action_mask, logits)]
            loss, extra = equilibria.nash_approx(payoffs, logits, joint_mask=masks)
            marginals = [jax.nn.softmax(l) for l in logits]
            extra = dict(
                marginals=marginals,
                logits=logits,
                action_mask=action_mask,
                dist_to_normal_cone=_stationarity_residual(payoffs, marginals),
                **extra,
            )
        elif objective == Objective.MAX_DEVIATION_GAIN:
            loss, extra = equilibria.max_deviation_gain(payoffs, outputs)
        else:
            raise ValueError(f"Unrecognised objective ({objective}).")

        extra = {"payoffs": payoffs, "masks": masks, **extra}
        return loss, extra

    def initial_params(key, payoffs, masks):
        return hk.transform(_loss_fn).init(key, payoffs, masks)

    @jax.jit
    def evaluate(params, key, payoffs, masks):
        key, this_key = jax.random.split(key)
        loss_fn = hk.transform(jax.vmap(_loss_fn))
        loss, extra = loss_fn.apply(params, this_key, payoffs, masks)
        # Batch means of the logged scalars, taken before `extra` is narrowed below
        # to the first game of the batch for inspection.
        metrics = {k: jnp.mean(extra[k]) for k in _LOGGED_SCALARS[objective]}
        extra = jax.tree_map(lambda arr: arr[0], extra)

        return jnp.mean(loss), (key, dict(metrics=metrics, **extra))

    @jax.jit
    def update(params, key, opt_state, payoffs, masks):
        (loss, (key, extra)), grads = jax.value_and_grad(evaluate, has_aux=True)(params, key, payoffs, masks)
        extra["grad_norm"] = optax.global_norm(grads)
        update, opt_state = optim.update(grads, opt_state, params=params)
        extra["update_norm"] = optax.global_norm(update)
        params = optax.apply_updates(params, update)
        extra["params_norm"] = optax.global_norm(params)
        return params, key, opt_state, (loss, extra)

    return initial_params, update, evaluate


######
# OPTIMISER
######


def _generate_mask(
    pattern_include: re.Pattern[str],
    pattern_exclude: re.Pattern[str],
    params,
    parent_key: str = "",
):
    """Get a weight mask based on parameter names."""
    processed = {}
    for k, v in params.items():
        path = parent_key + "/" + k if parent_key else k
        if isinstance(v, Mapping):
            processed[k] = _generate_mask(pattern_include, pattern_exclude, v, path)
        else:
            processed[k] = pattern_include.match(path) is not None and pattern_exclude.match(path) is None
    return processed


_WEIGHT_DECAY_INCLUDE_PARAMS = "()"
_WEIGHT_DECAY_EXCLUDE_PARAMS = "(.*/b$|.*/gamma$|.*layer_norm.*)"


def _weight_decay_param_mask(params: hk.Params) -> Mapping[str, bool]:
    include_re = re.compile(_WEIGHT_DECAY_INCLUDE_PARAMS)
    exclude_re = re.compile(_WEIGHT_DECAY_EXCLUDE_PARAMS)
    mask = _generate_mask(include_re, exclude_re, params)
    logging.info("Optimiser weight decay mask: %s", mask)
    return mask


@optax.inject_hyperparams
def optimiser(
    learning_rate: float,
) -> optax.GradientTransformation:
    """Returns optax.chain of optimiser transforms."""
    return optax.chain(
        optax.adaptive_grad_clip(0.01),
        optax.adamw(
            learning_rate,
            weight_decay=0.1,
            mask=_weight_decay_param_mask,
        ),
    )


# @title Select a learning objective and optimise an instance of the NfgTransformer on n-player general-sum NFGs.

max_num_updates = 20000  # @param {type:"integer"}
batch_size = 32  # @param {type:"integer"}
num_strategies = (16, 16)  # @param {type:"raw"}

objective = (
    Objective.NE
)  # @param ["Objective.NE", "Objective.MAX_DEVIATION_GAIN", "Objective.RECONSTRUCTION"] {type:"raw"}

if objective in (Objective.NE, Objective.MAX_DEVIATION_GAIN):
    # Sample batches of payoff tensors from the L2-invariant game subspace.
    generate_payoffs = games.generate_payoffs(
        games.Game.ZERO_SUM,
        {},
        num_strategies=num_strategies,
        batch_size=batch_size,
    )
elif objective == Objective.RECONSTRUCTION:
    # Sample batches of payoff tensors of empirical disc games for payoff
    # prediction. Observing a random subset of the joint-actions.
    generate_payoffs = games.generate_payoffs(
        games.Game.EMPIRICAL_DISC_GAME,
        {"joint_action_keep_prob": 0.5, "latent_size": 4},
        num_strategies=num_strategies,
        batch_size=batch_size,
    )
else:
    raise ValueError(f"Unknown objective: {objective}")

# Use an adam optimiser with annealed learning rate and weight decay.
optim = optimiser(
    optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1e-4,
        warmup_steps=1_000,
        decay_steps=max_num_updates,
        end_value=1e-6,
        exponent=1.0,
    )
)

# NfgTransformer(D=64, K=8, A=2)
model_config = dict(
    num_heads=8,
    num_action_channels=64,
    num_qkv_channels=64,
    num_blocks=8,
    num_self_attend_per_block=2,
)
initial_params, update, evaluate = make_model(
    objective=objective,
    optim=optim,
    **model_config,
)

key = hk.PRNGSequence(42)
params = initial_params(
    next(key),
    payoffs=jnp.zeros((len(num_strategies),) + num_strategies),
    masks=jnp.ones(num_strategies),
)
opt_state = optim.init(params)

key = next(key)

# Set WANDB_PROJECT to log elsewhere, or WANDB_MODE=offline/disabled to not log.
run = wandb.init(
    project=os.environ.get("WANDB_PROJECT", "nfg-transformer"),
    config=dict(
        objective=objective.name,
        max_num_updates=max_num_updates,
        batch_size=batch_size,
        num_strategies=num_strategies,
        **model_config,
    ),
)

for i in range(max_num_updates):
    (payoffs, masks), key = generate_payoffs(key)
    params, key, opt_state, (loss, extra) = update(params, key, opt_state, payoffs, masks)
    if i % 1000 == 0:
        # Logging forces a host sync, so it follows the reporting cadence.
        metrics = jax.device_get(dict(loss=loss, **extra["metrics"]))
        run.log(metrics, step=i)
        print(f"Iteration {i}: " + ", ".join(f"{k} = {v:.4f}" for k, v in metrics.items()))

# @title Randomly sample a batch of new payoffs and inspect the outputs.
import numpy as np

(payoffs, masks), key = generate_payoffs(key)

loss, (key, extra) = evaluate(params, key, payoffs, masks)

final_metrics = jax.device_get(dict(loss=loss, **extra["metrics"]))
run.summary.update({f"final_{k}": v for k, v in final_metrics.items()})
run.finish()

print(f"Loss (average): {loss}")
np.set_printoptions(suppress=True, precision=2, linewidth=180)

print("Payoffs:\n", extra.pop("payoffs"))
print("Masks:\n", extra.pop("masks"))

print("######")
print("# Model outputs:")
print("######\n")

for k, v in extra.items():
    print(f"{k}:\n{v}")
