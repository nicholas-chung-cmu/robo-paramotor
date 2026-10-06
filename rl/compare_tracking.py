"""Fly every checkpoint on the same routes for a fixed time; compare tracking.

    python -m rl.compare_tracking runs/a/checkpoint.pkl runs/b/checkpoint.pkl ... \
        --seconds 60 --difficulty 0.2 --out runs/tracking_60s.csv

Each checkpoint flies --flights routes (same seeds, full ~1 km length, the
given difficulty). Errors are averaged over the steps each flight was still
flying; a flight that ends early counts as not surviving.
"""

import os
import sys
import csv
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from rl import machine  # noqa: E402

machine.limit_jax_memory()
import argparse

import jax
import numpy as np

from rl.rl_env import ParamotorEnv, config_from_saved
from rl.train import ActorCritic, load_checkpoint
from viewer.watch_training import make_flyer

FIELDS = ["checkpoint", "update", "env_steps_m", "trained_difficulty", "survived", "cross_track_m",
          "cross_track_p90_m", "altitude_error_m", "progress_m", "error"]


def evaluate(path, args):
    saved = load_checkpoint(path)
    cfg = config_from_saved(saved["env"])
    cfg.min_route_points = cfg.route_points - 1
    cfg.episode_seconds = args.seconds
    env = ParamotorEnv(cfg)
    network = ActorCritic(saved["ppo"]["hidden_size"])
    fly = make_flyer(env, network, env.episode_steps)
    states = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))(
        jax.random.split(jax.random.PRNGKey(args.seed), args.flights), args.difficulty)
    _, (qpos, ctrl, metrics, active) = fly(saved["params"], states, np.ones(args.flights, bool))
    metrics, active = np.asarray(metrics), np.asarray(active)
    failed = (metrics[:, :, 5] * active).max(axis=0) > 0
    lateral, vertical = metrics[:, :, 0][active], metrics[:, :, 1][active]
    progress = np.where(active, metrics[:, :, 2], 0).max(axis=0)
    return dict(
        update=saved.get("update", ""), env_steps_m=f"{saved.get('env_steps', 0) / 1e6:.1f}",
        trained_difficulty=f"{float(saved.get('difficulty', 0)):.1f}",
        survived=f"{int((~failed).sum())}/{args.flights}",
        cross_track_m=f"{lateral.mean():.2f}", cross_track_p90_m=f"{np.percentile(lateral, 90):.2f}",
        altitude_error_m=f"{vertical.mean():.2f}", progress_m=f"{np.median(progress):.0f}", error="")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", type=Path, nargs="+")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--difficulty", type=float, default=0.2)
    ap.add_argument("--flights", type=int, default=16)
    ap.add_argument("--seed", type=int, default=30000)
    ap.add_argument("--out", type=Path, default=Path("runs/tracking_60s.csv"))
    args = ap.parse_args()
    with args.out.open("w", newline="", buffering=1) as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        for path in args.checkpoints:
            start = time.time()
            try:
                row = evaluate(path, args)
            except Exception as exc:  # older checkpoints may not fit the current env
                row = dict(error=f"{type(exc).__name__}: {str(exc)[:120]}")
            row["checkpoint"] = str(path)
            writer.writerow(row)
            print(f"{path}: {row} ({time.time() - start:.0f} s)", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
