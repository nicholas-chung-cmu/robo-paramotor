"""Evaluate saved policies on repeatable routes; optionally replay one flight.

    python -m rl.evaluate runs/first/checkpoint.pkl
    python -m rl.evaluate runs/base/seed*/checkpoint.pkl --episodes 50

Every checkpoint flies the same flights (same routes, same seeds) in ONE
batch, so evaluating five seeds takes about as long as evaluating one: the
cost is the number of sequential physics steps, not the number of flights.
Results go to <checkpoint dir>/eval/ unless --output is given (one checkpoint).

Per-flight statistics are accumulated on the GPU; only the first --trace
episodes per route are recorded step by step (flights.csv, routes.csv), so
thousands of flights stay cheap. The run stops as soon as every flight ended.
"""

import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault(
    "JAX_COMPILATION_CACHE_DIR", str(Path(__file__).resolve().parents[1] / "runs" / ".jax_cache")
)
import argparse
import csv
import json
import time

import jax
import jax.numpy as jp
import numpy as np
from tqdm import tqdm

from rl import routes
from rl.rl_env import EnvConfig, ParamotorEnv
from rl.train import ActorCritic, load_checkpoint

COLUMNS = [
    "time_s",
    "x_m",
    "y_m",
    "z_m",
    "vx_m_s",
    "vy_m_s",
    "vz_m_s",
    "thrust_N",
    "brake_L_rad",
    "brake_R_rad",
    "cross_track_m",
    "altitude_error_m",
    "progress_m",
    "alpha_outside",
    "episode_return",
    "failed",
    "completed",
    "reward",
    "terminated",
    "truncated",
]
LATERAL, ALTITUDE, OUTSIDE = 10, 11, 13


def make_evaluator(env, network, checkpoints, trace, view):
    """Jitted chunk of control steps over all flights of all checkpoints.

    Flights are laid out checkpoint-major: flight n belongs to checkpoint
    n // (N / checkpoints). `trace` indexes the flights recorded per step.
    """

    def act(params, obs):
        obs = obs.reshape((checkpoints, -1, obs.shape[-1]))
        mean = jax.vmap(lambda p, o: network.apply(p, o)[0])(params, obs)
        return jp.tanh(mean.reshape((-1, mean.shape[-1])))

    def run(params, states, active, stats, length):
        def step(carry, _):
            states, active, stats = carry
            new, reward, terminated, truncated, metrics = jax.vmap(env.step)(
                states, act(params, states.obs)
            )
            data = new.data
            rows = jp.concatenate(
                (
                    data.time[:, None],
                    data.site_xpos[:, env.m.site("pod_com").id],
                    data.sensordata[:, env.slices["pod_vel"]],
                    data.ctrl[:, env.physics.thrust, None],
                    new.brakes,
                    metrics,
                    reward[:, None],
                    terminated[:, None],
                    truncated[:, None],
                ),
                axis=1,
            )
            # The step that ends a flight still counts toward its statistics.
            w = active.astype(rows.dtype)
            stats = dict(
                steps=stats["steps"] + w,
                lateral=stats["lateral"] + w * rows[:, LATERAL],
                lateral_max=jp.where(active, jp.maximum(stats["lateral_max"], rows[:, LATERAL]),
                                     stats["lateral_max"]),
                altitude=stats["altitude"] + w * rows[:, ALTITUDE],
                outside=stats["outside"] + w * rows[:, OUTSIDE],
                last=jp.where(active[:, None], rows, stats["last"]),
            )
            done = terminated | truncated
            # Freeze completed episodes so later scan steps cannot diverge.
            keep = active & ~done
            states = jax.tree.map(
                lambda old, nxt: jp.where(
                    keep.reshape((keep.shape[0],) + (1,) * (old.ndim - 1)), nxt, old
                ),
                states,
                new,
            )
            record = (rows[trace], active[trace])
            if view:
                record += (data.qpos[0], data.qvel[0], data.ctrl[0])
            return (states, keep, stats), record

        (states, active, stats), record = jax.lax.scan(
            step, (states, active, stats), None, length=length
        )
        return states, active, stats, record

    return jax.jit(run, static_argnames="length")


def replay(env, route, qpos, qvel, ctrl):
    import mujoco
    import mujoco.viewer
    from model.paramotor_aero import ParamotorAero
    from model.paramotor_params import PEEK_1M

    model = env.m
    data = mujoco.MjData(model)
    aero = ParamotorAero(model, PEEK_1M)
    mujoco.set_mjcb_passive(aero)
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 10
            viewer.cam.azimuth, viewer.cam.elevation = 120, -20
            for q, v, u in zip(qpos, qvel, ctrl):
                if not viewer.is_running():
                    break
                start = time.monotonic()
                data.qpos[:], data.qvel[:], data.ctrl[:] = q, v, u
                mujoco.mj_forward(model, data)
                viewer.cam.lookat[:] = data.site_xpos[model.site("pod_com").id]
                with viewer.lock():
                    scene = viewer.user_scn
                    scene.ngeom = 0
                    for a, b in zip(route[:-1:3], route[3::3]):
                        if scene.ngeom >= scene.maxgeom:
                            break
                        g = scene.geoms[scene.ngeom]
                        mujoco.mjv_initGeom(
                            g,
                            mujoco.mjtGeom.mjGEOM_LINE,
                            np.zeros(3),
                            np.zeros(3),
                            np.eye(3).ravel(),
                            np.array([1.0, 0.65, 0.1, 1.0]),
                        )
                        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 3.0, a, b)
                        scene.ngeom += 1
                viewer.sync()
                time.sleep(max(0.0, env.control_dt - (time.monotonic() - start)))
    finally:
        mujoco.set_mjcb_passive(None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", type=Path, nargs="+")
    ap.add_argument("--path", choices=("all",) + routes.KINDS, default="all")
    ap.add_argument(
        "--episodes", type=int, default=20, help="number of fixed seeds per route"
    )
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument(
        "--trace", type=int, default=3,
        help="episodes per route recorded step by step in flights.csv (all are in summary.csv)",
    )
    ap.add_argument("--output", type=Path, help="output directory (single checkpoint only)")
    ap.add_argument(
        "--view", action="store_true", help="replay a single route/episode in MuJoCo"
    )
    args = ap.parse_args()
    if args.episodes < 1 or args.seconds <= 0 or args.trace < 0:
        ap.error("episodes and seconds must be positive, trace nonnegative")
    if args.view and (args.path == "all" or args.episodes != 1 or len(args.checkpoints) != 1):
        ap.error("--view requires one checkpoint, a specific --path and --episodes 1")
    if args.output and len(args.checkpoints) != 1:
        ap.error("--output needs exactly one checkpoint; otherwise each goes to <checkpoint dir>/eval")
    # A --view replay is one flight: keep it away from the real evaluation results.
    default = "replay" if args.view else "eval"
    outputs = [args.output or c.parent / default for c in args.checkpoints]

    saved = [load_checkpoint(c) for c in args.checkpoints]
    for c, s in zip(args.checkpoints[1:], saved[1:]):
        if s["env"] != saved[0]["env"] or s["ppo"]["hidden_size"] != saved[0]["ppo"]["hidden_size"]:
            ap.error(f"{c} has a different env config or network size; evaluate it separately")
    cfg = EnvConfig(**saved[0]["env"])
    cfg.episode_seconds = args.seconds
    env = ParamotorEnv(cfg)
    network = ActorCritic(saved[0]["ppo"]["hidden_size"])
    params = jax.tree.map(lambda *p: jp.stack(p), *[s["params"] for s in saved])

    # The same flights for every checkpoint: identical routes and seeds.
    kinds = routes.KINDS[1:] if args.path == "all" else (args.path,)
    names = [kind for kind in kinds for _ in range(args.episodes)]
    seeds = [args.seed + i for _ in kinds for i in range(args.episodes)]
    keys = jp.stack([jax.random.PRNGKey(seed) for seed in seeds])
    states = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))(keys, 1.0)
    tracks = [routes.make_path(key, 1.0, name) for key, name in zip(keys, names)]
    points = jp.stack([p for p, _ in tracks]) + states.origin[:, None, :]
    states = states.replace(points=points, arc=jp.stack([a for _, a in tracks]))
    states = jax.vmap(env._observation)(states)
    C, F = len(saved), len(names)
    N = C * F
    states = jax.tree.map(lambda x: jp.concatenate([x] * C), states)

    traced = [j for j, s in enumerate(seeds) if s - args.seed < args.trace]
    trace = jp.array([c * F + j for c in range(C) for j in traced], dtype=jp.int32)
    run = make_evaluator(env, network, C, trace, args.view)
    zeros = jp.zeros(N)
    stats = dict(steps=zeros, lateral=zeros, lateral_max=zeros, altitude=zeros,
                 outside=zeros, last=jp.zeros((N, len(COLUMNS))))
    active = jp.ones(N, bool)

    print(f"Evaluating {C} checkpoint(s) x {F} flights = {N} flights on {jax.devices()}; "
          "first chunk compiles...", flush=True)
    chunk, total, done_steps, records = cfg.control_hz, env.episode_steps, 0, []
    bar = tqdm(total=total, unit="step", desc="flights", mininterval=1.0, dynamic_ncols=True)
    while done_steps < total:
        length = min(chunk, total - done_steps)
        states, active, stats, record = run(params, states, active, stats, length=length)
        records.append(jax.device_get(record))
        done_steps += length
        airborne = int(active.sum())
        failed = int(stats["last"][:, COLUMNS.index("failed")].sum())
        bar.update(length)
        bar.set_postfix_str(f"t={done_steps * env.control_dt:.0f}s airborne={airborne}/{N} failed={failed}")
        if airborne == 0 and done_steps < total:
            bar.write(f"all flights ended at t={done_steps * env.control_dt:.1f}s")
            break
    bar.close()

    stats = jax.device_get(stats)
    rows = np.concatenate([r[0] for r in records])  # (steps, traced, columns)
    alive = np.concatenate([r[1] for r in records])
    route_points, route_arcs = np.asarray(points), np.asarray(states.arc[:F])
    count = np.maximum(stats["steps"], 1)
    header = ["path", "seed", "duration_s", "progress_m", "mean_cross_track_m", "max_cross_track_m",
              "mean_altitude_error_m", "alpha_outside_fraction", "return", "failed", "completed"]
    for c, (checkpoint, out) in enumerate(zip(args.checkpoints, outputs)):
        out.mkdir(parents=True, exist_ok=True)
        summary = []
        for j, (name, seed) in enumerate(zip(names, seeds)):
            n, last = c * F + j, stats["last"][c * F + j]
            summary.append([name, seed, last[0], last[12], stats["lateral"][n] / count[n],
                            stats["lateral_max"][n], stats["altitude"][n] / count[n],
                            stats["outside"][n] / count[n], last[14], int(last[15]), int(last[16])])
        with (out / "summary.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(summary)
        with (out / "flights.csv").open("w", newline="") as f, (out / "routes.csv").open("w", newline="") as g:
            flights, route_writer = csv.writer(f), csv.writer(g)
            flights.writerow(["path", "seed"] + COLUMNS)
            route_writer.writerow(["path", "seed", "arc_m", "x_m", "y_m", "z_m"])
            for t, j in enumerate(traced):
                k = c * len(traced) + t
                for row in rows[alive[:, k], k]:
                    flights.writerow([names[j], seeds[j], *row])
                for arc, point in zip(route_arcs[j], route_points[j]):
                    route_writer.writerow([names[j], seeds[j], arc, *point])
        (out / "meta.json").write_text(json.dumps(
            {"seconds": args.seconds, "episodes": args.episodes, "seed": args.seed,
             "trace": args.trace, "launch_speed": cfg.launch_speed,
             "checkpoint": str(checkpoint)}, indent=2))
        s = np.array([r[3] for r in summary]), np.array([r[9] for r in summary])
        print(f"{checkpoint}: {F} flights, mean progress {s[0].mean():.1f} m, "
              f"failed {s[1].mean():.0%} -> {out}", flush=True)
    if args.view:
        qpos, qvel, ctrl = (np.concatenate([r[i] for r in records]) for i in (2, 3, 4))
        mask = alive[:, 0]
        replay(env, route_points[0], qpos[mask], qvel[mask], ctrl[mask])


if __name__ == "__main__":
    main()
