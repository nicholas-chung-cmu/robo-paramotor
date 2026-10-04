"""PPO for the closed-loop control policy (MLP, no recurrent state), in JAX. Run --help for small, reproducible training runs."""

import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from rl import machine  # noqa: E402  (before JAX: memory cap from machine.toml)

machine.limit_jax_memory()
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
import warp as wp

from rl.rl_env import ENDING_REASONS, EnvConfig, ParamotorEnv, config_from_saved, select


@dataclass
class PPOConfig:
    seed: int = 0
    num_envs: int = machine.load().get("num_envs", 1024)  # set per computer in machine.toml
    rollout_steps: int = machine.load().get("rollout_steps", 128)  # set per computer in machine.toml
    updates: int = 1000
    epochs: int = 4
    minibatches: int = 32  # 16,384-sample minibatches at 4096 envs: 128 gradient steps per update
    hidden_size: int = 128
    learning_rate: float = 3e-4  # annealed linearly to learning_rate_final over training
    learning_rate_final: float = 3e-5
    gamma: float = machine.load().get("gamma", 0.997)  # set in machine.toml (0.997 ~ 13 s at 25 Hz)
    gae_lambda: float = machine.load().get("gae_lambda", 0.95)  # set in machine.toml
    clip: float = 0.2
    value_clip: float = 1.0  # value-loss clip, in return units (not the policy's 0.2)
    entropy: float = 0.001   # bonus at the start, decayed linearly to 0 over training
    initial_log_std: float = -3.0  # pre-tanh action noise std 0.05: almost none at the start
    max_grad_norm: float = 0.5
    eval_every: int = 25
    checkpoint_every: int = 5  # a crash or retry loses at most this many updates
    # The critic also sees ground truth the vehicle cannot measure (canopy shape
    # and attitude, true airspeed, route error; ParamotorEnv._privileged). It is
    # used only in training, so the deployed policy stays sensor-only.
    asymmetric_critic: bool = True


class ActorCritic(nn.Module):
    hidden_size: int = 128
    thrust_bias: float = 0.693  # default only; main() sets it from initial_thrust / thrust_max
    initial_log_std: float = -0.7  # default only; main() passes PPOConfig.initial_log_std

    @nn.compact
    def __call__(self, obs, critic_obs=None):
        """(mean, log_std, value). critic_obs is the critic's input: obs, or
        obs plus privileged features for an asymmetric critic. Without it the
        critic is skipped and value is zero (policy-only callers).

        Layer names are fixed to the ones Flax assigned when actor and critic
        layers were built interleaved, so older checkpoints still load.
        """
        actor = obs
        for i in range(2):
            actor = nn.tanh(nn.Dense(self.hidden_size, name=f"Dense_{2 * i}")(actor))
        # Begin near cruise with brakes released: actions are [thrust, brake,
        # diff]; brake tanh(-2) is ~2% and diff 0 is no differential. (diff
        # was -2 here, i.e. nearly full LEFT brake, a steep left spiral.)
        bias = lambda key, shape, dtype: jp.array([self.thrust_bias, -2.0, 0.0], dtype)
        mean = nn.Dense(
            3, kernel_init=nn.initializers.orthogonal(0.01), bias_init=bias, name="Dense_4"
        )(actor)
        log_std = self.param("log_std", nn.initializers.constant(self.initial_log_std), (3,))
        if critic_obs is None:
            value = jp.zeros(obs.shape[:-1])
        else:
            critic = critic_obs
            for i in range(2):
                critic = nn.tanh(nn.Dense(self.hidden_size, name=f"Dense_{2 * i + 1}")(critic))
            value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0),
                             name="Dense_5")(critic)[..., 0]
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


def critic_input(states, cfg):
    """The critic's observation: the policy's, plus privileged features."""
    if cfg.asymmetric_critic:
        return jp.concatenate((states.obs, states.privileged), axis=-1)
    return states.obs


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
            critic_obs = critic_input(states, cfg)
            mean, std, value = network.apply(params, states.obs, critic_obs)
            latent = mean + jp.exp(std) * jax.random.normal(ak, mean.shape)
            logp = log_probability(latent, mean, std)
            next_states, reward, term, trunc, metrics = batched_step(
                states, jp.tanh(latent)
            )
            next_value = network.apply(
                params, next_states.obs, critic_input(next_states, cfg))[2]
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
                critic_obs,
            )
            done = term | trunc

            def reset_finished(states):
                return select(done, fresh, states)

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

    def loss(params, batch, entropy_coef):
        obs, critic_obs, latent, old_logp, old_value, adv, target = batch
        mean, log_std, value = network.apply(params, obs, critic_obs)
        logp = log_probability(latent, mean, log_std)
        ratio = jp.exp(logp - old_logp)
        actor = -jp.mean(
            jp.minimum(ratio * adv, jp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv)
        )
        clipped = old_value + jp.clip(value - old_value, -cfg.value_clip, cfg.value_clip)
        critic = 0.5 * jp.mean(
            jp.maximum((value - target) ** 2, (clipped - target) ** 2)
        )
        # Base-normal entropy encourages exploration; action bounds use tanh.
        entropy = jp.sum(log_std + 0.5 * np.log(2 * np.pi * np.e))
        kl = jp.mean((ratio - 1) - (logp - old_logp))
        clip_fraction = jp.mean(jp.abs(ratio - 1) > cfg.clip)
        return actor + 0.5 * critic - entropy_coef * entropy, jp.array(
            [actor, critic, entropy, kl, clip_fraction]
        )

    def update(learner, key, transitions, entropy_coef):
        obs, latent, logp, value, reward, nv, term, trunc, _, critic_obs = transitions
        adv, target = advantages(
            reward, value, nv, term, trunc, cfg.gamma, cfg.gae_lambda
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        batch = jax.tree.map(
            lambda x: x.reshape((total,) + x.shape[2:]),
            (obs, critic_obs, latent, logp, value, adv, target),
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
                    learner.params, batch, entropy_coef
                )
                return learner.apply_gradients(grads=grads), metrics

            learner, metrics = jax.lax.scan(minibatch, learner, batches)
            return (learner, key), metrics.mean(0)

        (learner, key), metrics = jax.lax.scan(
            epoch, (learner, key), None, length=cfg.epochs
        )
        return learner, key, metrics.mean(0)

    return jax.jit(update)


MEMORY_FIELDS = (
    "update", "device_used_gb", "outside_jax_gb", "jax_pool_gb", "jax_in_use_gb",
    "jax_peak_gb", "jax_largest_free_gb",
)


def gpu_memory():
    """GPU memory in GB. JAX (policy and PPO) keeps its own pool and reports
    it. MuJoCo Warp's physics buffers do not show in Warp's mempool counters,
    so they are measured as outside_jax: whole-GPU use minus JAX's pool, which
    also holds fixed overhead (CUDA contexts, other processes such as the
    watcher). Growth across updates is what matters."""
    gb = 1e-9
    stats = jax.devices()[0].memory_stats() or {}
    device = wp.get_device("cuda:0") if wp.is_cuda_available() else None
    used = (device.total_memory - device.free_memory) * gb if device else 0.0
    pool = stats.get("pool_bytes", stats.get("bytes_reserved", 0)) * gb
    return {
        "device_used_gb": used,
        "outside_jax_gb": used - pool,
        "jax_pool_gb": pool,
        "jax_in_use_gb": stats.get("bytes_in_use", 0) * gb,
        "jax_peak_gb": stats.get("peak_bytes_in_use", 0) * gb,
        "jax_largest_free_gb": stats.get("largest_free_block_bytes", 0) * gb,
    }


def save_checkpoint(
    path, learner, env_cfg, ppo_cfg, update, difficulty, key, env_steps, end_update
):
    payload = dict(
        end_update=end_update,  # the run's target; --finish resumes to it
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
        states = select(keep, new, states)
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
    ap.add_argument(
        "--finish",
        action="store_true",
        help="with --resume: train to the checkpoint's original target update "
             "(what docker/train.sh does when it retries a crashed run)",
    )
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
    ec = config_from_saved(saved["env"]) if saved else EnvConfig(**raw.get("env", {}))
    # Checkpoints from before the asymmetric critic trained a symmetric one.
    pc = PPOConfig(**({"asymmetric_critic": False, **saved["ppo"]} if saved
                      else raw.get("ppo", {})))
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
    network = ActorCritic(pc.hidden_size, thrust_bias, pc.initial_log_std)
    key = jax.random.PRNGKey(pc.seed)
    key, nk, rk = jax.random.split(key, 3)
    critic_size = env.obs_size + (env.privileged_size if pc.asymmetric_critic else 0)
    params = network.init(nk, jp.zeros(env.obs_size), jp.zeros(critic_size))
    # Updates are counted absolutely, so a resumed or retried run continues the
    # learning-rate and entropy schedules instead of restarting them.
    start_update = saved["update"] if saved else 0
    if args.finish:
        if not saved or "end_update" not in saved:
            ap.error("--finish needs --resume with a checkpoint that records its target")
        end_update = saved["end_update"]
    else:
        end_update = start_update + pc.updates
    pc.updates = end_update - start_update
    if pc.updates <= 0:
        print(f"Already at update {start_update} of {end_update}; nothing to train.", flush=True)
        return
    learner = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=optax.chain(
            optax.clip_by_global_norm(pc.max_grad_norm),
            optax.adam(optax.linear_schedule(
                pc.learning_rate, pc.learning_rate_final,
                end_update * pc.epochs * pc.minibatches)),
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
    # The gate flies whole episodes (to the end of the current route, a
    # failure, or the time limit), so promotion is judged on the full route
    # rather than the first few seconds after launch.
    gate = jax.jit(
        lambda params, diff: evaluate_batch(env, network, params, difficulty=diff)
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
    endings_mode = "a" if saved and (args.output / "endings.csv").exists() else "w"
    endings_file = (args.output / "endings.csv").open(endings_mode, newline="", buffering=1)
    endings_writer = csv.writer(endings_file)
    if endings_mode == "w":
        endings_writer.writerow(("update", "ended", *ENDING_REASONS, "completed", "time_limit"))
    memory_mode = "a" if saved and (args.output / "memory.csv").exists() else "w"
    memory_file = (args.output / "memory.csv").open(memory_mode, newline="", buffering=1)
    memory_writer = csv.writer(memory_file)
    if memory_mode == "w":
        memory_writer.writerow(MEMORY_FIELDS)
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
            # Entropy bonus decays linearly to 0 at the run's target update.
            entropy_coef = pc.entropy * max(0.0, 1.0 - iteration / end_update)
            learner, key, loss = update(learner, key, batch, jp.float32(entropy_coef))
            loss = np.asarray(loss)
            if not np.all(np.isfinite(loss)):
                raise FloatingPointError("Nonfinite PPO update")
            metrics = np.asarray(batch[8])
            done = np.asarray(batch[6] | batch[7])
            ended = done.sum()
            finished = (metrics[:, :, 4] * done).sum() / max(ended, 1)
            # How the episodes ended: failure reasons (metrics after `completed`;
            # one step can trip several), success, and the time limit.
            terminated, truncated = np.asarray(batch[6]), np.asarray(batch[7])
            causes = [int(metrics[:, :, 7 + i][done].sum()) for i in range(len(ENDING_REASONS))]
            causes += [int(metrics[:, :, 6][done].sum()), int((truncated & ~terminated).sum())]
            endings_writer.writerow((iteration + 1, int(ended), *causes))
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
                    *loss[:4],  # clip fraction is printed, not in metrics.csv
                    difficulty,
                    speed,
                ]
            )
            mem = gpu_memory()
            memory_writer.writerow([iteration + 1, *(f"{mem[k]:.3f}" for k in MEMORY_FIELDS[1:])])
            print(
                f"update {iteration+1}: reward={reward:.3f} lateral={metrics[:,:,0].mean():.2f} m "
                f"level={difficulty:.2f} {speed:.1f} steps/s | gpu {mem['device_used_gb']:.1f} GB "
                f"(jax pool {mem['jax_pool_gb']:.1f}, outside jax {mem['outside_jax_gb']:.1f})",
                flush=True,
            )
            if args.verbose:
                print(
                    f"    ppo: policy_loss={loss[0]:+.4f} value_loss={loss[1]:.4f} "
                    f"entropy={loss[2]:.3f} approx_kl={loss[3]:.5f} clip_frac={loss[4]:.3f} | altitude_err="
                    f"{metrics[:,:,1].mean():.2f} m outside_alpha={metrics[:,:,3].mean():.1%} | "
                    f"episodes_ended={int(ended)} finished_return={finished:.1f} "
                    f"({', '.join(f'{n} {c}' for n, c in zip((*ENDING_REASONS, 'completed', 'time_limit'), causes) if c)}) | "
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
                    end_update,
                )
    memory_file.close()
    endings_file.close()
    print("Saved", checkpoint, flush=True)


if __name__ == "__main__":
    main()
