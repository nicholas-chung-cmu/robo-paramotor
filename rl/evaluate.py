"""Evaluate a saved policy on repeatable routes; optionally replay one flight."""

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


def make_evaluator(env, network, steps):
    def run(params, states):
        def step(carry, _):
            states, active = carry
            action = jp.tanh(network.apply(params, states.obs)[0])
            new, reward, terminated, truncated, metrics = jax.vmap(env.step)(
                states, action
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
            record = (rows, active, data.qpos, data.qvel, data.ctrl)
            return (states, keep), record

        return jax.lax.scan(
            step, (states, jp.ones(states.steps.shape, bool)), None, length=steps
        )[1]

    return jax.jit(run)


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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--path", choices=("all",) + routes.KINDS, default="all")
    ap.add_argument(
        "--episodes", type=int, default=3, help="number of fixed seeds per route"
    )
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--output", type=Path, default=Path("runs/evaluation"))
    ap.add_argument(
        "--view", action="store_true", help="replay a single route/episode in MuJoCo"
    )
    args = ap.parse_args()
    if args.episodes < 1 or args.seconds <= 0:
        ap.error("episodes and seconds must be positive")
    if args.view and (args.path == "all" or args.episodes != 1):
        ap.error("--view requires a specific --path and --episodes 1")
    saved = load_checkpoint(args.checkpoint)
    cfg = EnvConfig(**saved["env"])
    cfg.episode_seconds = args.seconds
    env = ParamotorEnv(cfg)
    network = ActorCritic(saved["ppo"]["hidden_size"])
    kinds = routes.KINDS[1:] if args.path == "all" else (args.path,)
    names = [kind for kind in kinds for _ in range(args.episodes)]
    seeds = [args.seed + i for _ in kinds for i in range(args.episodes)]
    keys = jp.stack([jax.random.PRNGKey(seed) for seed in seeds])
    states = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))(keys, 1.0)
    tracks = [routes.make_path(key, 1.0, name) for key, name in zip(keys, names)]
    points = jp.stack([p for p, _ in tracks]) + states.origin[:, None, :]
    states = states.replace(points=points, arc=jp.stack([a for _, a in tracks]))
    states = jax.vmap(env._observation)(states)
    print(
        f"Evaluating {len(names)} flights on {jax.devices()}; first call compiles...",
        flush=True,
    )
    rows, active, qpos, qvel, ctrl = jax.device_get(
        make_evaluator(env, network, env.episode_steps)(saved["params"], states)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    with (args.output / "flights.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "seed"] + COLUMNS)
        for i, (name, seed) in enumerate(zip(names, seeds)):
            flight = rows[active[:, i], i]
            for row in flight:
                writer.writerow([name, seed, *row])
            last = flight[-1]
            lateral = np.nan_to_num(
                flight[:, 10], nan=cfg.max_cross_track_m, posinf=cfg.max_cross_track_m
            )
            summary.append(
                [
                    name,
                    seed,
                    last[0],
                    last[12],
                    np.mean(lateral),
                    np.max(lateral),
                    np.nan_to_num(flight[:, 11], nan=25.0).mean(),
                    flight[:, 13].mean(),
                    last[14],
                    int(last[15]),
                    int(last[16]),
                ]
            )
            print(
                f"{name:12s} seed={seed} time={last[0]:.1f}s progress={last[12]:.1f}m "
                f"lateral={np.mean(lateral):.2f}m failed={bool(last[15])}",
                flush=True,
            )
    with (args.output / "summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "path",
                "seed",
                "duration_s",
                "progress_m",
                "mean_cross_track_m",
                "max_cross_track_m",
                "mean_altitude_error_m",
                "alpha_outside_fraction",
                "return",
                "failed",
                "completed",
            ]
        )
        writer.writerows(summary)
    (args.output / "meta.json").write_text(json.dumps(
        {"seconds": args.seconds, "episodes": args.episodes, "seed": args.seed,
         "launch_speed": cfg.launch_speed, "checkpoint": str(args.checkpoint)}, indent=2))
    arcs = np.asarray(states.arc)
    with (args.output / "routes.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "seed", "arc_m", "x_m", "y_m", "z_m"])
        for i, (name, seed) in enumerate(zip(names, seeds)):
            for arc, point in zip(arcs[i], np.asarray(points[i])):
                writer.writerow([name, seed, arc, *point])
    print("Wrote", args.output / "summary.csv", "flights.csv and routes.csv", flush=True)
    if args.view:
        mask = active[:, 0]
        replay(env, np.asarray(points[0]), qpos[mask, 0], qvel[mask, 0], ctrl[mask, 0])


if __name__ == "__main__":
    main()
