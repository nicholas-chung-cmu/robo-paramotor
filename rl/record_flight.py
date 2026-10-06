"""Record the best of N flights of a checkpoint as raw RGB frames on stdout.

    python -m rl.record_flight runs/<run>/checkpoint.pkl --difficulty 0 \
        | ffmpeg -f rawvideo -pix_fmt rgb24 -s 1280x720 -r 25 -i - out.mp4

Flights use the full ~1 km route at the given difficulty (like the watcher).
The video is sped up to last at most --max-seconds (30 s) at 25 fps.
Progress messages go to stderr.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "glfw")  # hidden window on the shared X display
from rl import machine  # noqa: E402

machine.limit_jax_memory()
import argparse

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from rl.rl_env import ParamotorEnv, config_from_saved
from rl.train import ActorCritic, load_checkpoint
from viewer.watch_training import (CHUNK_STEPS, ROUTE_RGBA, TRAIL_EVERY, add_line,
                                   display_model, make_flyer, trail_colour)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--difficulty", type=float, default=0.0)
    ap.add_argument("--flights", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--seconds", type=float, default=None, help="flight length (default: the run's episode)")
    ap.add_argument("--max-seconds", type=float, default=30.0, help="longest video; longer flights play faster")
    ap.add_argument("--fps", type=int, default=25, help="must match ffmpeg's -r")
    args = ap.parse_args()

    saved = load_checkpoint(args.checkpoint)
    cfg = config_from_saved(saved["env"])
    cfg.min_route_points = cfg.route_points - 1  # the whole route
    if args.seconds:
        cfg.episode_seconds = args.seconds
    env = ParamotorEnv(cfg)
    network = ActorCritic(saved["ppo"]["hidden_size"])
    fly = make_flyer(env, network, CHUNK_STEPS)
    reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    states = reset(jax.random.split(jax.random.PRNGKey(args.seed), args.flights), args.difficulty)

    carry, chunks = (states, jp.ones(args.flights, bool)), []
    for done in range(0, env.episode_steps, CHUNK_STEPS):
        carry, chunk = fly(saved["params"], *carry)
        chunks.append(jax.device_get(chunk))
        flying = int(np.asarray(carry[1]).sum())
        log(f"record: {(done + CHUNK_STEPS) * env.control_dt:.0f} s flown, {flying} of {args.flights} still flying")
        if not flying:
            break
    qpos, ctrl, metrics, active = (np.concatenate(x) for x in zip(*chunks))
    progress = np.where(active, metrics[:, :, 2], -np.inf).max(axis=0)
    best = int(np.argmax(progress))
    mask = active[:, best]
    failed = bool(metrics[mask, best, 5].max() > 0)
    completed = bool(metrics[mask, best, 6].max() > 0)
    log(f"record: update {saved['update']}, difficulty {args.difficulty}: best flight {best} "
        f"{progress[best]:.0f} m in {mask.sum() * env.control_dt:.0f} s "
        f"({'completed' if completed else 'crashed' if failed else 'time limit'}); "
        f"median {np.median(progress):.0f} m; "
        f"completed {int((metrics[:, :, 6] * active).max(axis=0).sum())} of {args.flights}")

    model = display_model(env.m)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, args.height, args.width)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 35, 120, -25
    route = np.asarray(states.points[best])
    pod_id = model.site("pod_com").id
    trail = []  # (start, end, rgba)
    last = None
    out = sys.stdout.buffer
    frames = int(mask.sum())
    stride = max(1, int(np.ceil(frames / (args.max_seconds * args.fps))))
    log(f"record: {frames * env.control_dt:.0f} s flight -> {frames / stride / args.fps:.0f} s video "
        f"({stride * env.control_dt * args.fps:.1f}x speed)")
    for i, (q, u) in enumerate(zip(qpos[mask, best], ctrl[mask, best])):
        data.qpos[:], data.ctrl[:] = q, u
        mujoco.mj_forward(model, data)
        pod = data.site_xpos[pod_id].copy()
        if last is None or i % TRAIL_EVERY == 0:
            if last is not None:
                trail.append((last, pod, trail_colour(pod, route)))
            last = pod
        if i % stride:  # skipped frame: the trail above still grows
            continue
        camera.lookat[:] = pod
        renderer.update_scene(data, camera)
        scene = renderer.scene
        for a, b in zip(route[:-1:3], route[3::3]):
            add_line(scene, a, b, ROUTE_RGBA, 3.0)
        for a, b, rgba in trail[-(scene.maxgeom - scene.ngeom):]:
            add_line(scene, a, b, rgba, 4.0)
        out.write(renderer.render().tobytes())
        if i % 250 == 0:
            log(f"record: rendered {i * env.control_dt:.0f} s")
    out.flush()


if __name__ == "__main__":
    main()
