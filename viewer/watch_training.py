"""Watch a policy train: replay the best flight of every new checkpoint.

    python -m viewer.watch_training runs/first
    docker/train.sh --name first --headed ...     # training + this window

Each time training writes a checkpoint, the latest policy flies --flights
routes at the run's current curriculum difficulty (the same seeds every time,
so checkpoints are comparable). The flight that gets furthest along its route
is replayed on a loop, with its route drawn, until the next checkpoint's best
flight replaces it. Simulating is slower than real time, so the window starts
playing the leading flight after the first 10 s and extends it as the flights
are flown further. A trail shows the path flown, red at the route's height,
magenta below it and blue above it. Evaluation runs in a background thread, so the window
keeps playing while the next one computes.

For a directory of seed runs (runs/<name>/seed0, seed1, ...) it follows
whichever seed has the newest checkpoint, i.e. the one training right now.
"""

import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from rl import machine  # noqa: E402  (before JAX: memory cap from machine.toml)

machine.limit_jax_memory("watcher_memory_fraction")
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
from rl.rl_env import ParamotorEnv, config_from_saved, select
from rl.train import ActorCritic, load_checkpoint

ROUTE_RGBA = np.array([1.0, 0.65, 0.1, 1.0])
# The flown trail, coloured by height against the route: red within the
# trail band, shading to magenta below the route and to blue above it.
TRAIL_ON, TRAIL_BELOW, TRAIL_ABOVE = (np.array(c) for c in (
    [1.0, 0.1, 0.1, 1.0], [1.0, 0.1, 1.0, 1.0], [0.2, 0.4, 1.0, 1.0]))
TRAIL_BAND_M = 1.0   # green inside this, full red/blue by TRAIL_FULL_M off
TRAIL_FULL_M = 6.0
TRAIL_EVERY = 3      # one trail segment per this many control steps (0.12 s)


def newest_checkpoint(run):
    candidates = [run / "checkpoint.pkl", *run.glob("seed*/checkpoint.pkl")]
    existing = [c for c in candidates if c.exists()]
    return max(existing, key=lambda c: c.stat().st_mtime) if existing else None


CHUNK_STEPS = 250  # flights are flown 10 s at a time, until every one has ended


def make_flyer(env, network, steps):
    """All flights in one batch, `steps` control steps further; returns the
    carried (states, active) and every flight's trajectory over the chunk."""

    def fly(params, states, active):
        def step(carry, _):
            states, active = carry
            action = jp.tanh(network.apply(params, states.obs)[0])
            new, _, terminated, truncated, metrics = jax.vmap(env.step)(states, action)
            done = terminated | truncated
            keep = active & ~done
            states = select(keep, new, states)
            d = new.data
            return (states, keep), (d.qpos, d.ctrl, metrics, active)

        return jax.lax.scan(step, (states, active), None, length=steps)

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
        if self.seconds:  # else the run's own episode length (whole flights)
            cfg.episode_seconds = self.seconds
        key = (repr(saved["env"]), saved["ppo"]["hidden_size"])
        if key != self.env_key:  # rebuild (and recompile) only if the config changed
            self.env = ParamotorEnv(cfg)
            self.network = ActorCritic(saved["ppo"]["hidden_size"])
            self.fly = make_flyer(self.env, self.network, CHUNK_STEPS)
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
        # Chunk by chunk, stopping once every flight has crashed or finished,
        # so the cost is the longest flight rather than the whole time limit.
        # Simulating is slower than real time, so the window plays the leading
        # flight while the rest is still being flown: each chunk publishes it
        # so far, and the window extends the replay.
        carry, chunks, shown = (states, jp.ones(self.flights, bool)), [], None
        for done in range(0, self.env.episode_steps, CHUNK_STEPS):
            carry, chunk = self.fly(saved["params"], *carry)
            chunks.append(jax.device_get(chunk))
            flying = np.asarray(carry[1])
            final = not flying.any() or done + CHUNK_STEPS >= self.env.episode_steps
            qpos, ctrl, metrics, active = (np.concatenate(x) for x in zip(*chunks))
            progress = np.where(active, metrics[:, :, 2], -np.inf).max(axis=0)
            # Keep showing a flight while it is still flying; switch only when
            # it has ended and another has gone further (or at the end, the best).
            leader = int(np.argmax(progress))
            if shown is None or (not flying[shown] and progress[leader] > progress[shown]):
                shown = leader
            self.publish(path, saved, states, shown, qpos, ctrl, metrics, active,
                         progress, final, len(chunks) * CHUNK_STEPS)
            print(f"watch: update {saved['update']}: {len(chunks) * CHUNK_STEPS * self.env.control_dt:.0f} s flown, "
                  f"{int(flying.sum())} of {self.flights} still flying", flush=True)
            if final:
                break

    def publish(self, path, saved, states, best, qpos, ctrl, metrics, active, progress, final, steps):
        mask = active[:, best]
        failed = bool(metrics[mask, best, 5].max() > 0)
        status = ("crashed" if failed else "still flying") if final or not active[-1, best] else "flying..."
        if final:
            record = progress[best] > self.best_ever
            self.best_ever = max(self.best_ever, progress[best])
            tail = "   NEW BEST" if record else f"   (best so far {self.best_ever:.0f} m)"
        else:
            tail = f"   (simulating: {steps * self.env.control_dt:.0f} s flown)"
        self.results.put(dict(
            id=(path, saved["update"], best),
            final=final,
            model=self.env.m,
            qpos=qpos[mask, best], ctrl=ctrl[mask, best],
            route=np.asarray(states.points[best]),
            dt=self.env.control_dt,
            text=(
                f"{path.parent.name}  update {saved['update']}  "
                f"({saved['env_steps'] / 1e6:.1f} M steps)\n"
                f"difficulty {float(saved['difficulty']):.1f}\n"
                f"{'best' if final else 'leader'} of {self.flights}: {progress[best]:.0f} m along route, "
                f"{mask.sum() * self.env.control_dt:.0f} s, {status}\n"
                f"median flight: {np.median(progress):.0f} m" + tail
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


SCENE = Path(__file__).resolve().parents[1] / "model" / "scene.xml"


def display_model(trained):
    """scene.xml (the same aircraft plus sky, ground and lighting) for the
    window, with the ground moved to z = 0 where the env counts a ground
    crash. Falls back to the training model if the scene does not match it."""
    try:
        model = mujoco.MjModel.from_xml_path(str(SCENE))
    except ValueError as exc:
        print(f"watch: no scene ({exc}); showing the bare model", flush=True)
        return trained
    if (model.nq, model.nu) != (trained.nq, trained.nu):
        print("watch: scene.xml does not match the trained model; showing the bare model", flush=True)
        return trained
    for name in ("ground", "mountains"):
        model.geom(name).pos[2] = 0.0
    model.vis.map.zfar = 2000.0  # whole 1 km routes stay in view
    return model


def add_line(scene, a, b, rgba, width):
    """Append one line segment to the viewer's extra geoms; False when full."""
    if scene.ngeom >= scene.maxgeom:
        return False
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3),
                        np.eye(3).ravel(), rgba)
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width, a, b)
    scene.ngeom += 1
    return True


def draw_route(viewer, route):
    scene = viewer.user_scn
    scene.ngeom = 0
    for a, b in zip(route[:-1:3], route[3::3]):
        if not add_line(scene, a, b, ROUTE_RGBA, 3.0):
            break


def trail_colour(position, route):
    """Height against the nearest route point (horizontally): below is magenta."""
    nearest = route[np.argmin(np.sum((route[:, :2] - position[:2]) ** 2, axis=1))]
    error = position[2] - nearest[2]
    mix = np.clip((abs(error) - TRAIL_BAND_M) / (TRAIL_FULL_M - TRAIL_BAND_M), 0.0, 1.0)
    return (1 - mix) * TRAIL_ON + mix * (TRAIL_BELOW if error < 0 else TRAIL_ABOVE)


def wait_for_monitor():
    """GLFW aborts the whole process (glfwGetVideoMode: monitor != NULL) if
    the window opens while no monitor is connected, e.g. displays asleep
    overnight. Wait for one instead."""
    import glfw
    warned = False
    while True:
        if glfw.init():
            found = glfw.get_primary_monitor() is not None
            glfw.terminate()
            if found:
                return
        if not warned:
            print("watch: no monitor connected (displays asleep?); waiting to open the window", flush=True)
            warned = True
        time.sleep(10)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="run directory (or a directory of seed runs)")
    ap.add_argument("--flights", type=int, default=16, help="flights per checkpoint; the best is shown")
    ap.add_argument("--seconds", type=float, default=None,
                    help="length of each flight (default: the run's whole episode, 300 s)")
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (2 = twice real time)")
    args = ap.parse_args()

    results = queue.Queue()
    Evaluator(args.run, args.flights, args.seconds, args.seed, results).start()
    print(f"Watching {args.run}: waiting for the first checkpoint...", flush=True)
    current = results.get()  # block until the first 10 s of the first checkpoint

    model = display_model(current["model"])
    data = mujoco.MjData(model)
    mujoco.set_mjcb_passive(ParamotorAero(model, PEEK_1M))
    font = mujoco.mjtFontScale.mjFONTSCALE_150
    wait_for_monitor()
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 10, 120, -20
            i, last, restart = 0, None, True
            while viewer.is_running():
                while not results.empty():
                    new = results.get()
                    # Same flight, flown further: keep playing. Otherwise start over.
                    restart |= new["id"] != current["id"]
                    current = new
                if restart:
                    with viewer.lock():
                        draw_route(viewer, current["route"])  # also clears the trail
                    i, last, restart = 0, None, False
                viewer.set_texts((font, mujoco.mjtGridPos.mjGRID_TOPLEFT, current["text"], None))
                if i >= len(current["qpos"]):
                    if current["final"]:  # pause, then loop the replay
                        time.sleep(0.5)
                        restart = True
                    else:  # caught up with the simulation: wait for the next chunk
                        time.sleep(0.1)
                        viewer.sync()
                    continue
                start = time.monotonic()
                data.qpos[:], data.ctrl[:] = current["qpos"][i], current["ctrl"][i]
                mujoco.mj_forward(model, data)
                pod = data.site_xpos[model.site("pod_com").id].copy()
                viewer.cam.lookat[:] = pod
                if last is None or i % TRAIL_EVERY == 0:  # the path flown so far
                    if last is not None:
                        with viewer.lock():
                            add_line(viewer.user_scn, last, pod,
                                     trail_colour(pod, current["route"]), 4.0)
                    last = pod
                viewer.sync()
                i += 1
                time.sleep(max(0.0, current["dt"] / args.speed - (time.monotonic() - start)))
    finally:
        mujoco.set_mjcb_passive(None)


if __name__ == "__main__":
    main()
