"""Feedforward PPO in JAX. Run --help for small, reproducible training runs."""

import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault(
    "JAX_COMPILATION_CACHE_DIR", str(Path(__file__).resolve().parents[1] / "runs" / ".jax_cache")
)
import argparse
import csv
from dataclasses import asdict, dataclass
import json
import pickle
import time

import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jp
import numpy as np
import optax

from rl.rl_env import EnvConfig, ParamotorEnv


@dataclass
class PPOConfig:
    seed: int = 0
    num_envs: int = 4096  # sized for a 16 GB desktop GPU; lower on laptops (see docs/RL.md)
    rollout_steps: int = 128
    updates: int = 1000
    epochs: int = 4
    minibatches: int = 4
    hidden_size: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.997  # ~13 s effective horizon at 25 Hz
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy: float = 0.001
    max_grad_norm: float = 0.5
    eval_every: int = 25
    checkpoint_every: int = 25


class ActorCritic(nn.Module):
    hidden_size: int = 128
    thrust_bias: float = 0.693  # tanh(0.693) = 0.6 -> 0.8 N of a 1.0 N range

    @nn.compact
    def __call__(self, obs):
        actor, critic = obs, obs
        for _ in range(2):
            actor = nn.tanh(nn.Dense(self.hidden_size)(actor))
            critic = nn.tanh(nn.Dense(self.hidden_size)(critic))
        # Begin near cruise with mostly released brakes, not both brakes at 50%.
        bias = lambda key, shape, dtype: jp.array([self.thrust_bias, -2.0, -2.0], dtype)
        mean = nn.Dense(
            3, kernel_init=nn.initializers.orthogonal(0.01), bias_init=bias
        )(actor)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(critic)[..., 0]
        log_std = self.param("log_std", nn.initializers.constant(-0.7), (3,))
        return mean, jp.clip(log_std, -5, 1), value


def log_probability(latent, mean, log_std):
    """Tanh-normal density; store latent actions to avoid unstable arctanh."""
    normal = (
        -0.5 * ((latent - mean) * jp.exp(-log_std)) ** 2
        - log_std
        - 0.5 * np.log(2 * np.pi)
    )
    log_jacobian = 2 * (np.log(2) - latent - jax.nn.softplus(-2 * latent))
    return jp.sum(normal - log_jacobian, axis=-1)


def advantages(reward, value, next_value, terminated, truncated, gamma, lam):
    """Bootstrap at time limits; do not propagate GAE into the reset episode."""
    delta = reward + gamma * (1 - terminated) * next_value - value

    def reverse(carry, inputs):
        residual, done = inputs
        advantage = residual + gamma * lam * (1 - done) * carry
        return advantage, advantage

    _, adv = jax.lax.scan(
        reverse, jp.zeros_like(delta[0]), (delta, terminated | truncated), reverse=True
    )
    return adv, adv + value


def make_rollout(env, network, cfg):
    batched_step = jax.vmap(env.step)
    batched_reset = jax.vmap(env.reset, in_axes=(0, None))

    def rollout(params, states, key, difficulty):
        # One fresh start per environment for the whole rollout, built once,
        # rather than rebuilding all num_envs starts on every step where any
        # episode ends. An environment that ends twice within one rollout
        # (rare: episodes last far longer than rollout_steps) restarts from
        # the same route and launch noise both times.
        key, rk = jax.random.split(key)
        fresh = batched_reset(jax.random.split(rk, cfg.num_envs), difficulty)

        def one(carry, _):
            states, key = carry
            key, ak = jax.random.split(key)
            mean, std, value = network.apply(params, states.obs)
            latent = mean + jp.exp(std) * jax.random.normal(ak, mean.shape)
            logp = log_probability(latent, mean, std)
            next_states, reward, term, trunc, metrics = batched_step(
                states, jp.tanh(latent)
            )
            next_value = network.apply(params, next_states.obs)[2]
            transition = (
                states.obs,
                latent,
                logp,
                value,
                reward,
                next_value,
                term,
                trunc,
                metrics,
            )
            done = term | trunc

            def reset_finished(states):
                return jax.tree.map(
                    lambda x, y: jp.where(
                        done.reshape((len(done),) + (1,) * (x.ndim - 1)), y, x
                    ),
                    states,
                    fresh,
                )

            next_states = jax.lax.cond(
                jp.any(done), reset_finished, lambda x: x, next_states
            )
            return (next_states, key), transition

        return jax.lax.scan(one, (states, key), None, length=cfg.rollout_steps)

    return jax.jit(rollout)


def make_update(network, cfg):
    total = cfg.num_envs * cfg.rollout_steps
    if total % cfg.minibatches:
        raise ValueError("rollout batch must divide evenly into minibatches")

    def loss(params, batch):
        obs, latent, old_logp, old_value, adv, target = batch
        mean, log_std, value = network.apply(params, obs)
        logp = log_probability(latent, mean, log_std)
        ratio = jp.exp(logp - old_logp)
        actor = -jp.mean(
            jp.minimum(ratio * adv, jp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv)
        )
        clipped = old_value + jp.clip(value - old_value, -cfg.clip, cfg.clip)
        critic = 0.5 * jp.mean(
            jp.maximum((value - target) ** 2, (clipped - target) ** 2)
        )
        # Base-normal entropy encourages exploration; action bounds use tanh.
        entropy = jp.sum(log_std + 0.5 * np.log(2 * np.pi * np.e))
        kl = jp.mean((ratio - 1) - (logp - old_logp))
        return actor + 0.5 * critic - cfg.entropy * entropy, jp.array(
            [actor, critic, entropy, kl]
        )

    def update(learner, key, transitions):
        obs, latent, logp, value, reward, nv, term, trunc, _ = transitions
        adv, target = advantages(
            reward, value, nv, term, trunc, cfg.gamma, cfg.gae_lambda
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        batch = jax.tree.map(
            lambda x: x.reshape((total,) + x.shape[2:]),
            (obs, latent, logp, value, adv, target),
        )

        def epoch(carry, _):
            learner, key = carry
            key, pk = jax.random.split(key)
            indices = jax.random.permutation(pk, total)
            batches = jax.tree.map(
                lambda x: x[indices].reshape((cfg.minibatches, -1) + x.shape[1:]), batch
            )

            def minibatch(learner, batch):
                (_, metrics), grads = jax.value_and_grad(loss, has_aux=True)(
                    learner.params, batch
                )
                return learner.apply_gradients(grads=grads), metrics

            learner, metrics = jax.lax.scan(minibatch, learner, batches)
            return (learner, key), metrics.mean(0)

        (learner, key), metrics = jax.lax.scan(
            epoch, (learner, key), None, length=cfg.epochs
        )
        return learner, key, metrics.mean(0)

    return jax.jit(update)


def save_checkpoint(
    path, learner, env_cfg, ppo_cfg, update, difficulty, key, env_steps
):
    payload = dict(
        params=jax.device_get(learner.params),
        opt_state=jax.device_get(learner.opt_state),
        train_step=int(learner.step),
        env=asdict(env_cfg),
        ppo=asdict(ppo_cfg),
        update=update,
        difficulty=difficulty,
        key=np.asarray(key),
        env_steps=env_steps,
    )
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as f:
        pickle.dump(payload, f)
    temporary.replace(path)


def load_checkpoint(path):
    # Only load checkpoints you created/trust: pickle is not an untrusted file format.
    with Path(path).open("rb") as f:
        return pickle.load(f)


def evaluate_batch(
    env, network, params, seed=10000, count=8, difficulty=0.0, steps=None
):
    """Fixed-seed gate; means are over active steps, including failed episodes."""
    states = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(jax.random.PRNGKey(seed), count), difficulty
    )

    def body(carry, _):
        states, active = carry
        actions = jp.tanh(network.apply(params, states.obs)[0])
        new, _, term, trunc, metrics = jax.vmap(env.step)(states, actions)
        record = jp.concatenate(
            (metrics[:, :4], metrics[:, 5, None], active[:, None]), axis=1
        )
        keep = active & ~(term | trunc)
        states = jax.tree.map(
            lambda old, nxt: jp.where(
                keep.reshape((count,) + (1,) * (old.ndim - 1)), nxt, old
            ),
            states,
            new,
        )
        return (states, keep), record

    (_, active), records = jax.lax.scan(
        body,
        (states, jp.ones(count, dtype=bool)),
        None,
        length=steps or env.episode_steps,
    )
    weights = records[:, :, 5]
    denom = jp.maximum(weights.sum(), 1)
    return jp.array(
        [
            jp.sum(records[:, :, 0] * weights) / denom,
            jp.sum(records[:, :, 1] * weights) / denom,
            jp.sum(records[:, :, 3] * weights) / denom,
            jp.sum(records[:, :, 4] * weights) / count,
            jp.sum(
                jp.maximum(
                    jp.max(jp.where(weights > 0, records[:, :, 2], 0.0), axis=0), 0.0
                )
            )
            / (denom * env.control_dt),
        ]
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config", type=Path, help="JSON with optional env and ppo dictionaries"
    )
    ap.add_argument("--output", type=Path, default=Path("runs/ppo"))
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--num-envs", type=int)
    ap.add_argument("--updates", type=int)
    ap.add_argument("--rollout-steps", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny end-to-end PPO run, not a trained policy",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="also print PPO losses, entropy, KL, tracking errors and episode stats every update",
    )
    args = ap.parse_args()
    saved = load_checkpoint(args.resume) if args.resume else None
    raw = json.loads(args.config.read_text()) if args.config else {}
    ec = EnvConfig(**(saved["env"] if saved else raw.get("env", {})))
    pc = PPOConfig(**(saved["ppo"] if saved else raw.get("ppo", {})))
    for name in ("num_envs", "updates", "rollout_steps", "seed"):
        value = getattr(args, name)
        if value is not None:
            setattr(pc, name, value)
    if args.smoke:
        pc.num_envs, pc.rollout_steps, pc.updates, pc.epochs, pc.minibatches = (
            2,
            8,
            2,
            1,
            1,
        )
        ec.episode_seconds = 0.4  # exercise automatic reset within the second rollout
        pc.eval_every = pc.checkpoint_every = 1
    if (
        min(
            pc.num_envs,
            pc.rollout_steps,
            pc.updates,
            pc.epochs,
            pc.minibatches,
            pc.eval_every,
            pc.checkpoint_every,
        )
        <= 0
    ):
        ap.error("training sizes must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "checkpoint.pkl"
    if checkpoint.exists() and not args.resume:
        ap.error("output already has a checkpoint; use --resume or a new directory")
    print("Devices:", jax.devices(), flush=True)
    env = ParamotorEnv(ec)
    # Initial mean thrust = the env's initial thrust, wherever thrust_max sits.
    thrust_bias = float(np.arctanh(np.clip(2 * ec.initial_thrust / ec.thrust_max - 1, -0.99, 0.99)))
    network = ActorCritic(pc.hidden_size, thrust_bias)
    key = jax.random.PRNGKey(pc.seed)
    key, nk, rk = jax.random.split(key, 3)
    params = network.init(nk, jp.zeros(env.obs_size))
    learner = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=optax.chain(
            optax.clip_by_global_norm(pc.max_grad_norm), optax.adam(pc.learning_rate)
        ),
    )
    learner = learner.replace(step=jp.array(0, jp.int32))
    start, difficulty = 0, 0.0 if ec.curriculum else 1.0
    env_steps = 0
    if saved:
        learner = learner.replace(
            params=saved["params"],
            opt_state=saved["opt_state"],
            step=jp.array(saved["train_step"], jp.int32),
        )
        start, difficulty, key = (
            saved["update"],
            saved["difficulty"],
            jp.array(saved["key"]),
        )
        env_steps = saved["env_steps"]
        key, rk = jax.random.split(key)
    print("Compiling environment reset...", flush=True)
    states = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))(
        jax.random.split(rk, pc.num_envs), difficulty
    )
    rollout, update = make_rollout(env, network, pc), make_update(network, pc)
    gate = jax.jit(
        lambda params, diff: evaluate_batch(
            env, network, params, difficulty=diff, steps=min(env.episode_steps, 250)
        )
    )
    (args.output / "config.json").write_text(
        json.dumps({"env": asdict(ec), "ppo": asdict(pc)}, indent=2)
    )
    print(
        f"{env.obs_size} observations, {pc.num_envs} environments; compiling first rollout...",
        flush=True,
    )
    mode = "a" if saved and (args.output / "metrics.csv").exists() else "w"
    gate_mode = "a" if saved and (args.output / "eval.csv").exists() else "w"
    with (args.output / "metrics.csv").open(mode, newline="", buffering=1) as f, (
        args.output / "eval.csv"
    ).open(gate_mode, newline="", buffering=1) as gate_file:
        writer = csv.writer(f)
        gate_writer = csv.writer(gate_file)
        if gate_mode == "w":
            gate_writer.writerow(
                [
                    "update",
                    "steps",
                    "difficulty",
                    "cross_track_m",
                    "altitude_error_m",
                    "alpha_outside_fraction",
                    "failure_rate",
                    "progress_m_s",
                    "passed",
                ]
            )
        if mode == "w":
            writer.writerow(
                [
                    "update",
                    "steps",
                    "reward",
                    "cross_track_m",
                    "altitude_error_m",
                    "alpha_outside_fraction",
                    "ended_episodes",
                    "mean_finished_return",
                    "policy_loss",
                    "value_loss",
                    "entropy",
                    "approx_kl",
                    "difficulty",
                    "steps_per_second",
                ]
            )
        for iteration in range(start, start + pc.updates):
            begin = time.monotonic()
            (states, key), batch = rollout(learner.params, states, key, difficulty)
            learner, key, loss = update(learner, key, batch)
            loss = np.asarray(loss)
            if not np.all(np.isfinite(loss)):
                raise FloatingPointError("Nonfinite PPO update")
            metrics = np.asarray(batch[-1])
            done = np.asarray(batch[6] | batch[7])
            ended = done.sum()
            finished = (metrics[:, :, 4] * done).sum() / max(ended, 1)
            count = pc.num_envs * pc.rollout_steps
            env_steps += count
            speed = count / (time.monotonic() - begin)
            reward = float(batch[4].mean())
            writer.writerow(
                [
                    iteration + 1,
                    env_steps,
                    reward,
                    *metrics[:, :, :2].mean((0, 1)),
                    metrics[:, :, 3].mean(),
                    ended,
                    finished,
                    *loss,
                    difficulty,
                    speed,
                ]
            )
            print(
                f"update {iteration+1}: reward={reward:.3f} lateral={metrics[:,:,0].mean():.2f} m "
                f"level={difficulty:.2f} {speed:.1f} steps/s",
                flush=True,
            )
            if args.verbose:
                print(
                    f"    ppo: policy_loss={loss[0]:+.4f} value_loss={loss[1]:.4f} "
                    f"entropy={loss[2]:.3f} approx_kl={loss[3]:.5f} | altitude_err="
                    f"{metrics[:,:,1].mean():.2f} m outside_alpha={metrics[:,:,3].mean():.1%} | "
                    f"episodes_ended={int(ended)} finished_return={finished:.1f} | "
                    f"{count / speed:.1f} s/update, {env_steps / 1e6:.2f} M steps total",
                    flush=True,
                )
            if ec.curriculum and (iteration + 1) % pc.eval_every == 0:
                result = np.asarray(gate(learner.params, difficulty))
                print(
                    "curriculum validation [lateral, vertical, envelope, failures, progress_m/s]:",
                    result,
                    flush=True,
                )
                passed = bool(
                    result[0] < 3
                    and result[1] < 2
                    and result[2] < 0.05
                    and result[3] < 0.1
                    and result[4] > 2.0
                )
                gate_writer.writerow(
                    [iteration + 1, env_steps, difficulty, *result, int(passed)]
                )
                if passed:
                    difficulty = min(1.0, difficulty + 0.2)
            if (
                iteration + 1
            ) % pc.checkpoint_every == 0 or iteration == start + pc.updates - 1:
                save_checkpoint(
                    checkpoint,
                    learner,
                    ec,
                    pc,
                    iteration + 1,
                    difficulty,
                    key,
                    env_steps,
                )
    print("Saved", checkpoint, flush=True)


if __name__ == "__main__":
    main()
