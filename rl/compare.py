"""Compare training configurations across seeds with rliable.

    python -m rl.compare runs/baseline runs/high_lr --output runs/compare/lr

Each argument is one configuration: either a directory of seed runs
(runs/baseline/seed0, seed1, ...) as written by `docker/train.sh --seeds N`,
or a single run directory (one seed). Every run needs eval/summary.csv from
rl.evaluate, and all of them should be evaluated with the same --seed,
--episodes and --seconds so they fly identical routes.

Score per run and route: the distance flown along the route, as a fraction of
what cruise speed covers in the evaluation window (launch_speed * seconds),
clipped to [0, 1]. A crash stops the distance, so this rewards staying up and
staying on the route. Routes play the role of rliable's "tasks".

Statistics follow Agarwal et al., "Deep RL at the Edge of the Statistical
Precipice" (NeurIPS 2021): IQM / mean / median / optimality gap with stratified
bootstrap 95% CIs, performance profiles, and probability of improvement.

Writes into --output:
    aggregates.png    IQM, median, mean, optimality gap per config, with CIs
    improvement.png   P(config X beats config Y) for every pair, with CIs
    profiles.png      fraction of runs scoring above tau, with CI bands
    learning.png      IQM across seeds vs environment steps (sample efficiency)
    scores.csv        the run x route score matrix
    report.md         tables of all of the above
"""

import argparse
import csv
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import numpy as np
from rliable import library as rly
from rliable import metrics

from rl.analyze import GRID, INK, INK_2, SURFACE, figure, read_csv, smooth, style

# Reference categorical palette (dataviz skill), light mode, fixed order.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
AGGREGATES = [
    ("IQM", metrics.aggregate_iqm),
    ("Median", metrics.aggregate_median),
    ("Mean", metrics.aggregate_mean),
    ("Optimality gap", metrics.aggregate_optimality_gap),
]


def find_runs(path):
    seeds = sorted(p for p in path.iterdir() if p.is_dir() and (p / "metrics.csv").exists())
    if (path / "metrics.csv").exists():
        return [path]
    if not seeds:
        raise SystemExit(f"rl.compare: no runs (metrics.csv) under {path}")
    return seeds


def route_scores(run):
    """One score per route: mean over that route's evaluation flights."""
    summary = read_csv(run / "eval" / "summary.csv")
    if summary is None:
        raise SystemExit(f"rl.compare: {run} has no eval/summary.csv; run rl.evaluate on it first")
    meta_path = run / "eval" / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    config = json.loads((run / "config.json").read_text())
    reach = meta.get("launch_speed", config["env"]["launch_speed"]) * meta.get("seconds", 60.0)
    routes = list(dict.fromkeys(summary["path"]))
    score = np.clip(summary["progress_m"] / reach, 0, 1)
    survival = np.clip(summary["duration_s"] / meta.get("seconds", 60.0), 0, 1)
    pick = lambda values: np.array([values[summary["path"] == r].mean() for r in routes])
    return routes, pick(score), pick(survival), pick(summary["mean_cross_track_m"])


def learning_curves(runs, key, points=60):
    """Each run's smoothed training metric, resampled onto a shared step grid."""
    data = [read_csv(r / "metrics.csv") for r in runs]
    end = min(d["steps"][-1] for d in data)
    grid = np.linspace(min(d["steps"][0] for d in data), end, points)
    shape = (lambda y: y) if key == "difficulty" else smooth  # difficulty moves in steps
    curves = np.stack([np.interp(grid, d["steps"], shape(d[key])) for d in data])
    return grid, curves[:, None, :]  # (runs, tasks=1, points) as rliable expects


def steps_to_full_difficulty(run):
    m = read_csv(run / "metrics.csv")
    hit = np.nonzero(m["difficulty"] >= 1.0)[0]
    return m["steps"][hit[0]] if len(hit) else np.nan


def color_of(names):
    if len(names) > len(CATEGORICAL):
        raise SystemExit(f"rl.compare: at most {len(CATEGORICAL)} configurations per plot")
    return dict(zip(names, CATEGORICAL))


def interval_plot(names, estimates, cis, colors, out):
    fig, axes = figure(1, len(AGGREGATES), "Final evaluation score: aggregate over runs and routes (95% bootstrap CI)",
                       height=0.5 * len(names) + 1.4)
    y = np.arange(len(names))
    for ax, (label, _) in zip(axes[0], AGGREGATES):
        style(ax, label + (" (lower is better)" if label == "Optimality gap" else ""))
        for i, n in enumerate(names):
            lo, hi = cis[n][:, AGGREGATES.index((label, _))]
            ax.barh(i, hi - lo, left=lo, height=0.5, color=colors[n], alpha=0.35)
            ax.plot([estimates[n][AGGREGATES.index((label, _))]] * 2, [i - 0.25, i + 0.25],
                    color=colors[n], linewidth=2.5)
        ax.set_yticks(y, names if ax is axes[0][0] else [""] * len(names))
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Route score (fraction of reachable distance)", color=INK_2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "aggregates.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)


def improvement_plot(pairs, probs, cis, out):
    labels = [f"P({a} > {b})" for a, b in pairs]
    fig, axes = figure(1, 1, "Probability of improvement on a random route (95% bootstrap CI)",
                       height=0.45 * len(labels) + 1.2)
    ax = axes[0][0]
    fig.set_figwidth(8)
    style(ax, "0.5 = no difference; above 0.5 favours the first configuration")
    for i, key in enumerate(f"{a},{b}" for a, b in pairs):
        lo, hi = cis[key]
        ax.barh(i, hi - lo, left=lo, height=0.5, color=CATEGORICAL[0], alpha=0.35)
        ax.plot([probs[key]] * 2, [i - 0.25, i + 0.25], color=CATEGORICAL[0], linewidth=2.5)
        ax.text(1.01, i, f"{probs[key]:.2f}", va="center", fontsize=8, color=INK_2,
                transform=ax.get_yaxis_transform())
    ax.axvline(0.5, color=INK_2, linewidth=1, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "improvement.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)


def profile_plot(names, scores, colors, out):
    taus = np.linspace(0, 1, 41)
    profiles, cis = rly.create_performance_profile(scores, taus, reps=2000)
    fig, axes = figure(1, 1, "Performance profile: fraction of (run, route) scores above tau", height=3.6)
    ax = axes[0][0]
    fig.set_figwidth(8)
    style(ax, "Higher and further right is better")
    for n in names:
        ax.fill_between(taus, cis[n][0], cis[n][1], color=colors[n], alpha=0.15, linewidth=0)
        ax.plot(taus, profiles[n], color=colors[n], linewidth=2, label=n)
    ax.set_xlabel("Route score tau", color=INK_2, fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out / "profiles.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)


def learning_plot(names, groups, colors, out):
    panels = [("reward", "Training reward per step"),
              ("difficulty", "Curriculum difficulty"),
              ("cross_track_m", "Training cross-track error (m)")]
    fig, axes = figure(1, len(panels), "Sample efficiency: IQM across seeds (95% bootstrap CI band)", height=3.2)
    iqm = lambda s: np.array([metrics.aggregate_iqm(s[..., j]) for j in range(s.shape[-1])])
    for ax, (key, label) in zip(axes[0], panels):
        style(ax, label)
        curves, grids = {}, {}
        for n in names:
            grids[n], curves[n] = learning_curves(groups[n], key)
        est, cis = rly.get_interval_estimates(curves, iqm, reps=1000)
        for n in names:
            ax.fill_between(grids[n], cis[n][0], cis[n][1], color=colors[n], alpha=0.15, linewidth=0)
            ax.plot(grids[n], est[n], color=colors[n], linewidth=2, label=n)
        ax.xaxis.set_major_formatter(EngFormatter())
        ax.set_xlabel("Environment steps", color=INK_2, fontsize=8)
        if key == "difficulty":
            ax.set_ylim(-0.05, 1.05)
    axes[0][0].legend(fontsize=8, frameon=False, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out / "learning.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", type=Path, nargs="+", help="config directories (seed runs inside) or single runs")
    ap.add_argument("--output", type=Path, help="default runs/compare/<names joined>")
    ap.add_argument("--reps", type=int, default=20000, help="bootstrap repetitions")
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})
    names = [c.name for c in args.configs]
    if len(set(names)) != len(names):
        ap.error("configuration directory names must be unique")
    out = args.output or Path("runs/compare") / "-vs-".join(names)
    out.mkdir(parents=True, exist_ok=True)
    colors = color_of(names)

    groups, scores, survival, cross, routes = {}, {}, {}, {}, None
    with (out / "scores.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        for config, n in zip(args.configs, names):
            groups[n] = find_runs(config)
            rows = []
            for run in groups[n]:
                r, score, surv, ct = route_scores(run)
                if routes is None:
                    routes = r
                    writer.writerow(["config", "run", *routes])
                elif r != routes:
                    ap.error(f"{run} was evaluated on different routes: {r} vs {routes}")
                writer.writerow([n, run.name, *np.round(score, 4)])
                rows.append((score, surv, ct))
            scores[n] = np.stack([s for s, _, _ in rows])
            survival[n] = np.stack([s for _, s, _ in rows])
            cross[n] = np.stack([c for _, _, c in rows])

    aggregate = lambda x: np.array([fn(x) for _, fn in AGGREGATES])
    estimates, cis = rly.get_interval_estimates(scores, aggregate, reps=args.reps)
    interval_plot(names, estimates, cis, colors, out)
    pairs = list(itertools.permutations(names, 2)) if len(names) <= 3 else list(itertools.combinations(names, 2))
    probs, prob_cis = {}, {}
    if pairs:
        probs, prob_cis = rly.get_interval_estimates(
            {f"{a},{b}": (scores[a], scores[b]) for a, b in pairs},
            metrics.probability_of_improvement, reps=min(args.reps, 2000))
        # A scalar statistic: flatten to a float and a (low, high) pair.
        probs = {k: float(np.ravel(v)[0]) for k, v in probs.items()}
        prob_cis = {k: np.ravel(v)[:2] for k, v in prob_cis.items()}
        improvement_plot(pairs, probs, prob_cis, out)
    profile_plot(names, scores, colors, out)
    learning_plot(names, groups, colors, out)

    fmt = lambda e, c: f"{e:.3f} [{c[0]:.3f}, {c[1]:.3f}]"
    lines = ["# Run comparison", "",
             "Route score = distance flown along the route / (launch speed x evaluation seconds), in [0, 1].",
             "Brackets are 95% stratified-bootstrap confidence intervals. "
             "With fewer than ~5 seeds per configuration they are wide; treat small differences as noise.", "",
             "## Aggregate score", "",
             "| config | runs | " + " | ".join(l for l, _ in AGGREGATES) + " | survival IQM | mean cross-track (m) | steps to full difficulty |",
             "| --- | --- | " + " | ".join("---" for _ in AGGREGATES) + " | --- | --- | --- |"]
    for n in names:
        reach = [steps_to_full_difficulty(r) for r in groups[n]]
        reached = [x for x in reach if np.isfinite(x)]
        reach_text = f"{np.median(reached) / 1e6:.2f} M ({len(reached)}/{len(reach)} runs)" if reached else f"not reached (0/{len(reach)})"
        lines.append(f"| {n} | {len(groups[n])} | " +
                     " | ".join(fmt(estimates[n][i], cis[n][:, i]) for i in range(len(AGGREGATES))) +
                     f" | {metrics.aggregate_iqm(survival[n]):.3f} | {cross[n].mean():.2f} | {reach_text} |")
    if pairs:
        lines += ["", "## Probability of improvement", "", "| comparison | P | 95% CI |", "| --- | --- | --- |"]
        for a, b in pairs:
            k = f"{a},{b}"
            lo, hi = prob_cis[k]
            lines.append(f"| P({a} > {b}) | {probs[k]:.3f} | [{lo:.3f}, {hi:.3f}] |")
    lines += ["", "## Mean score per route", "", "| config | " + " | ".join(routes) + " |",
              "| --- | " + " | ".join("---" for _ in routes) + " |"]
    for n in names:
        lines.append(f"| {n} | " + " | ".join(f"{v:.3f}" for v in scores[n].mean(0)) + " |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("Wrote", out, flush=True)


if __name__ == "__main__":
    main()
