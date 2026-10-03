"""Watch a policy train: replay the best flight of every new checkpoint.

    python -m viewer.watch_training runs/first
    docker/train.sh --name first --headed ...     # training + this window

Each time training writes a checkpoint, the latest policy flies --flights
routes at the run's current curriculum difficulty (the same seeds every time,
so checkpoints are comparable). The flight that gets furthest along its route
is replayed on a loop, with its route drawn, until the next checkpoint's best
flight replaces it. Evaluation runs in a background thread, so the window
keeps playing while the next one computes.

For a directory of seed runs (runs/<name>/seed0, seed1, ...) it follows
whichever seed has the newest checkpoint, i.e. the one training right now.
"""

import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault(
    "JAX_COMPILATION_CACHE_DIR", str(Path(__file__).resolve().parents[1] / "runs" / ".jax_cache")
)
import argparse
import queue
import threading
import time

import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer
import numpy as np

from model.paramotor_aero import ParamotorAero
from model.paramotor_params import PEEK_1M
from rl.rl_env import ParamotorEnv, config_from_saved
from rl.train import ActorCritic, load_checkpoint

ROUTE_RGBA = np.array([1.0, 0.65, 0.1, 1.0])


def newest_checkpoint(run):
    candidates = [run / "checkpoint.pkl", *run.glob("seed*/checkpoint.pkl")]
    existing = [c for c in candidates if c.exists()]
    return max(existing, key=lambda c: c.stat().st_mtime) if existing else None


def make_flyer(env, network, steps):
    """All flights in one batch; records every flight's trajectory."""

    def fly(params, states):
        def step(carry, _):
            states, active = carry
            action = jp.tanh(network.apply(params, states.obs)[0])
            new, _, terminated, truncated, metrics = jax.vmap(env.step)(states, action)
            done = terminated | truncated
            keep = active & ~done
            states = jax.tree.map(
                lambda old, nxt: jp.where(
                    keep.reshape((keep.shape[0],) + (1,) * (old.ndim - 1)), nxt, old
                ),
                states,
                new,
            )
            d = new.data
            return (states, keep), (d.qpos, d.ctrl, metrics, active)

        return jax.lax.scan(step, (states, jp.ones(states.steps.shape, bool)), None, length=steps)[1]

    return jax.jit(fly)


class Evaluator(threading.Thread):
    """Polls for new checkpoints; puts the best flight of each on a queue."""

    def __init__(self, run, flights, seconds, seed, results):
        super().__init__(daemon=True)
        self.run_dir, self.flights, self.seconds, self.seed = run, flights, seconds, seed
        self.results, self.env, self.fly, self.env_key = results, None, None, None
        self.best_ever, self.best_dir = -np.inf, None

    def setup(self, saved):
        cfg = config_from_saved(saved["env"])
        cfg.episode_seconds = self.seconds
        key = (repr(saved["env"]), saved["ppo"]["hidden_size"])
        if key != self.env_key:  # rebuild (and recompile) only if the config changed
            self.env = ParamotorEnv(cfg)
            self.network = ActorCritic(saved["ppo"]["hidden_size"])
            self.fly = make_flyer(self.env, self.network, self.env.episode_steps)
            self.reset = jax.jit(jax.vmap(self.env.reset, in_axes=(0, None)))
            self.env_key = key

    def evaluate(self, path):
        saved = load_checkpoint(path)
        self.setup(saved)
        if path.parent != self.best_dir:  # a new seed started: its own record
            self.best_ever, self.best_dir = -np.inf, path.parent
        difficulty = float(saved["difficulty"])
        keys = jax.random.split(jax.random.PRNGKey(self.seed), self.flights)
        states = self.reset(keys, difficulty)
        qpos, ctrl, metrics, active = jax.device_get(self.fly(saved["params"], states))
        progress = np.where(active, metrics[:, :, 2], -np.inf).max(axis=0)
        best = int(np.argmax(progress))
        mask = active[:, best]
        failed = bool(metrics[mask, best, 5].max() > 0)
        record = progress[best] > self.best_ever
        self.best_ever = max(self.best_ever, progress[best])
        self.results.put(dict(
            model=self.env.m,
            qpos=qpos[mask, best], ctrl=ctrl[mask, best],
            route=np.asarray(states.points[best]),
            dt=self.env.control_dt,
            text=(
                f"{path.parent.name}  update {saved['update']}  "
                f"({saved['env_steps'] / 1e6:.1f} M steps)\n"
                f"difficulty {difficulty:.1f}\n"
                f"best of {self.flights}: {progress[best]:.0f} m along route, "
                f"{mask.sum() * self.env.control_dt:.0f} s, {'crashed' if failed else 'still flying'}\n"
                f"median flight: {np.median(progress):.0f} m"
                + ("   NEW BEST" if record else f"   (best so far {self.best_ever:.0f} m)")
            ),
        ))

    def run(self):
        seen = None
        while True:
            path = newest_checkpoint(self.run_dir)
            stamp = (path, path.stat().st_mtime) if path else None
            if stamp and stamp != seen:
                seen = stamp
                try:
                    self.evaluate(path)
                except (EOFError, OSError, KeyError) as exc:  # mid-write or vanished; retry
                    print(f"watch: skipped {path}: {exc}", flush=True)
                    seen = None
            time.sleep(2.0)


def draw_route(viewer, route):
    scene = viewer.user_scn
    scene.ngeom = 0
    for a, b in zip(route[:-1:3], route[3::3]):
        if scene.ngeom >= scene.maxgeom:
            break
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3),
                            np.eye(3).ravel(), ROUTE_RGBA)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 3.0, a, b)
        scene.ngeom += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="run directory (or a directory of seed runs)")
    ap.add_argument("--flights", type=int, default=32, help="flights per checkpoint; the best is shown")
    ap.add_argument("--seconds", type=float, default=30.0, help="length of each flight")
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (2 = twice real time)")
    args = ap.parse_args()

    results = queue.Queue()
    Evaluator(args.run, args.flights, args.seconds, args.seed, results).start()
    print(f"Watching {args.run}: waiting for the first checkpoint...", flush=True)
    current = results.get()  # block until the first checkpoint has been flown

    model = current["model"]
    data = mujoco.MjData(model)
    mujoco.set_mjcb_passive(ParamotorAero(model, PEEK_1M))
    font = mujoco.mjtFontScale.mjFONTSCALE_150
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 10, 120, -20
            while viewer.is_running():
                with viewer.lock():
                    draw_route(viewer, current["route"])
                viewer.set_texts((font, mujoco.mjtGridPos.mjGRID_TOPLEFT, current["text"], None))
                for q, u in zip(current["qpos"], current["ctrl"]):
                    if not viewer.is_running() or not results.empty():
                        break
                    start = time.monotonic()
                    data.qpos[:], data.ctrl[:] = q, u
                    mujoco.mj_forward(model, data)
                    viewer.cam.lookat[:] = data.site_xpos[model.site("pod_com").id]
                    viewer.sync()
                    time.sleep(max(0.0, current["dt"] / args.speed - (time.monotonic() - start)))
                while not results.empty():  # jump to the newest checkpoint's best flight
                    current = results.get()
                time.sleep(0.5)  # brief pause before the replay loops
    finally:
        mujoco.set_mjcb_passive(None)


if __name__ == "__main__":
    main()
