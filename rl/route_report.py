"""Fly one policy on several route types and plot every flight plus the average.

    python -m rl.route_report runs/<run>/checkpoint.pkl --flights 20 --seconds 60

Each route type gets --flights flights (fixed seeds, full-length routes). All
flights run in one batch. Writes, into --out (default <run>/routes/):
    tracks.png      top-down: each route type's routes and the flown tracks
    cross_track.png sideways error over time: every flight, and the average
    height.png      height error over time (positive = above the route)
    turns.png       sideways error against route curvature (is it lagging turns?)
    summary.csv     per route type: survival, mean/p90 errors, distance flown
"""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from rl import machine  # noqa: E402

machine.limit_jax_memory()
import jax
import jax.numpy as jp
import numpy as np

from rl.analyze import CRITICAL, INK, INK_2, SERIES, SURFACE, figure, plt, style
from rl.rl_env import ParamotorEnv, _map_batched, config_from_saved
from rl.train import ActorCritic, load_checkpoint
from viewer.watch_training import make_flyer

# (label, path kind, difficulty). Fixed kinds ignore difficulty except "random".
ROUTES = [
    ("Straight", "straight", 0.0),
    ("Random, difficulty 0.2", "random", 0.2),
    ("Random, difficulty 0.5", "random", 0.5),
    ("Left circle (R 60 m)", "left", 1.0),
    ("Right circle (R 60 m)", "right", 1.0),
    ("S-turns (R 50 m)", "s_turn", 1.0),
    ("Figure eight", "figure_eight", 1.0),
    ("Climb", "climb", 1.0),
    ("Descend", "descend", 1.0),
]
FLIGHT = SERIES[0]
MEAN = INK


def fly_all(path, flights, seconds, seed):
    saved = load_checkpoint(path)
    cfg = config_from_saved(saved["env"])
    cfg.min_route_points = cfg.route_points - 1  # full-length routes
    cfg.episode_seconds = seconds
    env = ParamotorEnv(cfg)
    network = ActorCritic(saved["ppo"]["hidden_size"])
    keys = jax.random.split(jax.random.PRNGKey(seed), flights)
    batches = []
    for _, kind, difficulty in ROUTES:
        env.cfg.path_kind = kind  # read while tracing, so each kind gets its own reset
        batches.append(jax.jit(jax.vmap(env.reset, in_axes=(0, None)))(keys, difficulty))
    # MJX Warp shares some Data buffers across the batch: join only per-flight leaves.
    states = _map_batched(lambda *x: jp.concatenate(x), *batches)
    fly = make_flyer(env, network, env.episode_steps)
    _, (qpos, _, metrics, active) = fly(saved["params"], states, jp.ones(len(ROUTES) * flights, bool))
    return saved, env, np.asarray(states.points), np.asarray(qpos[:, :, :3]), np.asarray(metrics), np.asarray(active)


def signed_offset(points, xy):
    """Signed horizontal distance from the route polyline (left of travel > 0)
    and the route's curvature there (left turn > 0), for each position."""
    seg = points[1:, :2] - points[:-1, :2]
    rel = xy[:, None, :] - points[None, :-1, :2]
    u = np.clip(np.einsum("tsk,sk->ts", rel, seg) / np.maximum((seg ** 2).sum(-1), 1e-9), 0, 1)
    closest = points[None, :-1, :2] + u[..., None] * seg[None]
    dist = np.linalg.norm(xy[:, None] - closest, axis=-1)
    i = dist.argmin(axis=1)
    d = xy - points[i, :2]
    side = np.sign(seg[i, 0] * d[:, 1] - seg[i, 1] * d[:, 0])
    heading = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
    curvature = np.gradient(heading) / np.linalg.norm(seg, axis=1)
    return side * dist[np.arange(len(xy)), i], curvature[i]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--flights", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=40000)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    out = args.out or args.checkpoint.parent / "routes"
    out.mkdir(parents=True, exist_ok=True)

    saved, env, points, pos, metrics, active = fly_all(args.checkpoint, args.flights, args.seconds, args.seed)
    n, dt = args.flights, env.control_dt
    t = np.arange(metrics.shape[0]) * dt
    name = f"{args.checkpoint.parent.name}, update {saved['update']}"
    rows, cols = 3, 3

    tracks, _ = figure(rows, cols, f"Flown tracks, top-down: {n} flights per route type, {args.seconds:.0f} s ({name})", height=3.4)
    lat_fig, _ = figure(rows, cols, f"Sideways error from the route: {n} flights (thin) and their average (bold) ({name})")
    alt_fig, _ = figure(rows, cols, f"Height error, + above the route: {n} flights (thin) and their average (bold) ({name})")
    turn_fig, _ = figure(rows, cols, f"Signed sideways error against route curvature, + = outside of the turn ({name})")
    summary = []
    for r, (label, kind, difficulty) in enumerate(ROUTES):
        k = slice(r * n, (r + 1) * n)
        act, met, xyz = active[:, k], metrics[:, k], pos[:, k]
        lateral = np.where(act, met[:, :, 0], np.nan)
        vertical_abs = np.where(act, met[:, :, 1], np.nan)
        failed = (met[:, :, 5] * act).max(axis=0) > 0
        ax = tracks.axes[r]
        style(ax, label)
        signed_h, outside, curv_all = np.full_like(lateral, np.nan), [], []
        for f in range(n):
            route = points[r * n + f]
            ax.plot(route[:, 0], route[:, 1], color=INK_2, linewidth=1, alpha=0.35)
            m = act[:, f]
            ax.plot(xyz[m, f, 0], xyz[m, f, 1], color=FLIGHT, linewidth=0.9, alpha=0.7)
            if failed[f] and m.any():
                end = np.flatnonzero(m)[-1]
                ax.plot(xyz[end, f, 0], xyz[end, f, 1], marker="x", color=CRITICAL, markersize=6)
            # Signed height: above (+) or below (-) the route's height at the closest point.
            off, curv = signed_offset(route, xyz[m, f, :2])
            nearest = np.argmin(np.linalg.norm(route[None, :, :2] - xyz[m, f, None, :2], axis=-1), axis=1)
            signed_h[m, f] = xyz[m, f, 2] - route[nearest, 2]
            turning = np.abs(curv) > 1e-3
            outside.append(-off[turning] * np.sign(curv[turning]))  # left turn: outside is right (-)
            curv_all.append(np.abs(curv[turning]))
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x (m)", color=INK_2, fontsize=8)

        for fig, data, unit in ((lat_fig, lateral, "m"), (alt_fig, signed_h, "m")):
            a = fig.axes[r]
            style(a, label)
            a.plot(t, data, color=FLIGHT, linewidth=0.8, alpha=0.35)
            a.plot(t, np.nanmean(data, axis=1), color=MEAN, linewidth=2)
            a.set_xlabel("time (s)", color=INK_2, fontsize=8)
        a = turn_fig.axes[r]
        style(a, label)
        if sum(len(o) for o in outside):
            a.scatter(np.concatenate(curv_all) * 100, np.concatenate(outside), s=2, color=FLIGHT, alpha=0.15)
            a.axhline(0, color=INK_2, linewidth=1)
            a.set_xlabel("curvature (1/100 m)", color=INK_2, fontsize=8)
        else:
            a.text(0.5, 0.5, "straight: no turns", transform=a.transAxes, ha="center", color=INK_2)
        outside_all = np.concatenate(outside) if outside else np.array([])
        summary.append(dict(
            route=label, flights=n, survived=int((~failed).sum()),
            cross_track_mean_m=np.nanmean(lateral), cross_track_p90_m=np.nanpercentile(lateral, 90),
            height_error_mean_m=np.nanmean(vertical_abs),
            outside_of_turn_mean_m=outside_all.mean() if len(outside_all) else np.nan,
            distance_m=np.median(np.where(act, met[:, :, 2], 0).max(axis=0)),
        ))
    for fig, file in ((tracks, "tracks.png"), (lat_fig, "cross_track.png"), (alt_fig, "height.png"), (turn_fig, "turns.png")):
        fig.tight_layout()
        fig.savefig(out / file, dpi=130, facecolor=SURFACE)
        plt.close(fig)
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, summary[0].keys())
        w.writeheader()
        for row in summary:
            w.writerow({k: f"{v:.2f}" if isinstance(v, float) else v for k, v in row.items()})
    for row in summary:
        print(", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
