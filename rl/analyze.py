"""Plot and summarize a training run. Safe to run while training is in progress.

    python -m rl.analyze runs/first

Reads whatever exists in the run directory and writes, into <run>/analysis/:
    training.png    PPO learning curves from metrics.csv (every update)
    curriculum.png  fixed-seed curriculum gate from eval.csv, with pass thresholds
    evaluation.png  per-route results from eval/summary.csv and eval/flights.csv
    report.md       the headline numbers as text
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import numpy as np

# Reference palette (dataviz skill), light mode.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # first three slots: all-pairs safe
CRITICAL = "#d03b3b"

# The curriculum gate in rl/train.py: (column, label, threshold, direction).
GATE = [
    ("cross_track_m", "Cross-track error (m)", 3.0, "below"),
    ("altitude_error_m", "Altitude error (m)", 2.0, "below"),
    ("alpha_outside_fraction", "Time outside alpha envelope", 0.05, "below"),
    ("failure_rate", "Failure rate", 0.1, "below"),
    ("progress_m_s", "Progress along route (m/s)", 2.0, "above"),
    ("completion", "Flights finishing the route", 0.75, "above"),  # PPOConfig.gate_completion
]


def read_csv(path):
    if not path.exists():
        return None
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    out = {}
    for key in rows[0]:
        try:
            out[key] = np.array([float(r[key]) for r in rows])
        except ValueError:
            out[key] = np.array([r[key] for r in rows])
    return out


def smooth(y, frac=0.05):
    """Exponential moving average with a span of ~5% of the run."""
    alpha = 1.0 / max(1.0, frac * len(y))
    out, acc = np.empty_like(y, dtype=float), y[0]
    for i, v in enumerate(y):
        acc = acc + alpha * (v - acc) if np.isfinite(v) else acc
        out[i] = acc
    return out


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=6)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)


def figure(rows, cols, title, height=2.4):
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.6 * cols, height * rows + 0.6), squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK)
    return fig, axes


def curve(ax, x, y, title):
    style(ax, title)
    if len(y) > 20:
        ax.plot(x, y, color=SERIES[0], linewidth=1, alpha=0.25)
        ax.plot(x, smooth(y), color=SERIES[0], linewidth=2)
    else:
        ax.plot(x, y, color=SERIES[0], linewidth=2, marker="o", markersize=4)


def training_plots(m, out):
    x = m["steps"]
    panels = [
        ("reward", "Mean reward per step"),
        ("mean_finished_return", "Return of finished episodes"),
        ("ended_episodes", "Episodes ended per update"),
        ("difficulty", "Curriculum difficulty"),
        ("cross_track_m", "Cross-track error (m)"),
        ("altitude_error_m", "Altitude error (m)"),
        ("alpha_outside_fraction", "Time outside alpha envelope"),
        ("steps_per_second", "Throughput (env steps/s)"),
        ("policy_loss", "Policy loss"),
        ("value_loss", "Value loss"),
        ("entropy", "Policy entropy"),
        ("approx_kl", "Approx. KL per update"),
    ]
    fig, axes = figure(3, 4, "Training: rollouts from the current policy (raw faint, smoothed solid)")
    for ax, (key, label) in zip(axes.ravel(), panels):
        curve(ax, x, m[key], label)
        if key == "difficulty":
            ax.set_ylim(-0.05, 1.05)
    for ax in axes.ravel():
        ax.xaxis.set_major_formatter(EngFormatter())
    for ax in axes[-1]:
        ax.set_xlabel("Environment steps", color=INK_2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "training.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)


def curriculum_plots(g, out):
    x = g["steps"]
    fig, axes = figure(2, 4, "Curriculum gate: 8 fixed seeds every eval_every updates (dashed = pass threshold)")
    passed = g["passed"] > 0
    gate = [row for row in GATE if row[0] in g]  # runs before the completion check lack it
    for ax in axes.ravel()[len(gate) + 1:]:
        ax.axis("off")
    for ax, (key, label, limit, direction) in zip(axes.ravel(), gate):
        style(ax, f"{label}, {direction} {limit:g}")
        ax.plot(x, g[key], color=SERIES[0], linewidth=2, marker="o", markersize=4)
        ax.axhline(limit, color=INK_2, linewidth=1, linestyle="--")
        if passed.any():
            ax.plot(x[passed], g[key][passed], linestyle="none", marker="o",
                    markersize=8, markerfacecolor="none", markeredgecolor=INK, markeredgewidth=1.2)
    ax = axes.ravel()[len(gate)]
    style(ax, "Difficulty at each gate (ringed = gate passed)")
    ax.step(x, g["difficulty"], where="post", color=SERIES[0], linewidth=2)
    ax.set_ylim(-0.05, 1.05)
    for ax in axes.ravel():
        ax.xaxis.set_major_formatter(EngFormatter())
    for ax in axes[-1]:
        ax.set_xlabel("Environment steps", color=INK_2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "curriculum.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)


def evaluation_plots(s, flights, route_points, out, route_spacing=10.0):
    paths = list(dict.fromkeys(s["path"]))
    by = {p: s["path"] == p for p in paths}
    fig, axes = figure(2, 4, "Evaluation: fixed routes at full difficulty", height=2.8)
    bars = [
        ("completed", "Completion rate", lambda v: v.mean()),
        ("failed", "Failure rate", lambda v: v.mean()),
        ("mean_cross_track_m", "Mean cross-track error (m)", np.mean),
        ("progress_m", "Distance along route (m)", np.mean),
    ]
    y = np.arange(len(paths))
    for ax, (key, label, agg) in zip(axes[0], bars):
        style(ax, label)
        values = [agg(s[key][by[p]]) for p in paths]
        ax.barh(y, values, height=0.6, color=SERIES[0])
        ax.set_yticks(y, paths)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        for yi, v in zip(y, values):
            ax.text(v, yi, f" {v:.2g}", va="center", fontsize=7, color=INK_2)
        if key in ("completed", "failed"):
            ax.set_xlim(0, 1.15)
    # Top-down tracks, one panel per route, one line per seed.
    shown = paths[:4]
    for ax, p in zip(axes[1], shown):
        style(ax, f"{p}: top-down track (m)")
        if flights is None:
            continue
        mask = flights["path"] == p
        seeds = list(dict.fromkeys(flights["seed"][mask]))[:3]
        if route_points is not None and seeds:
            # The first seed's route, up to a little past the furthest progress.
            r = (route_points["path"] == p) & (route_points["seed"] == seeds[0])
            reach = flights["progress_m"][mask].max() + 40
            r &= route_points["point"] * route_spacing <= reach
            ax.plot(route_points["x_m"][r], route_points["y_m"][r], color=INK_2,
                    linewidth=1.2, linestyle="--", label="route")
        for color, seed in zip(SERIES, seeds):
            f = mask & (flights["seed"] == seed)
            ax.plot(flights["x_m"][f], flights["y_m"][f], color=color, linewidth=2,
                    label=f"seed {int(float(seed))}")
            if flights["failed"][f][-1] > 0:
                ax.plot(flights["x_m"][f][-1], flights["y_m"][f][-1], marker="x",
                        markersize=9, color=CRITICAL, markeredgewidth=2)
        ax.set_aspect("equal", adjustable="datalim")
        if seeds:
            ax.legend(fontsize=7, frameon=False, labelcolor=INK_2)
    for ax in axes[1][len(shown):]:
        ax.set_visible(False)
    fig.text(0.01, 0.005, "Dashed = target route (first seed). x = flight ended in failure. Tracks for the first four routes; all routes are in report.md.",
             fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out / "evaluation.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)


def report(run, m, g, s, out):
    lines = [f"# Run report: {run.name}", ""]
    if m is not None:
        tail = slice(max(0, len(m["update"]) - max(1, len(m["update"]) // 20)), None)
        lines += [
            "## Training (mean of the last 5% of updates)", "",
            f"- updates: {int(m['update'][-1])}, env steps: {m['steps'][-1] / 1e6:.2f} M, "
            f"throughput: {m['steps_per_second'][tail].mean():.0f} steps/s",
            f"- curriculum difficulty: {m['difficulty'][-1]:.1f} (1.0 = full)",
            f"- reward/step: {m['reward'][tail].mean():.3f}, finished-episode return: "
            f"{m['mean_finished_return'][tail].mean():.1f}",
            f"- cross-track: {m['cross_track_m'][tail].mean():.2f} m, altitude error: "
            f"{m['altitude_error_m'][tail].mean():.2f} m, outside alpha envelope: "
            f"{m['alpha_outside_fraction'][tail].mean():.1%}",
            f"- approx KL: {m['approx_kl'][tail].mean():.4f}, entropy: {m['entropy'][tail].mean():.3f}",
            "",
        ]
    if g is not None:
        last = {k: g[k][-1] for k in g}
        lines += ["## Last curriculum gate", "", "| metric | value | threshold | pass |", "| --- | --- | --- | --- |"]
        for key, label, limit, direction in (row for row in GATE if row[0] in last):
            ok = last[key] < limit if direction == "below" else last[key] > limit
            lines.append(f"| {label} | {last[key]:.3g} | {direction} {limit:g} | {'yes' if ok else 'no'} |")
        lines += ["", f"Gates passed: {int(g['passed'].sum())} of {len(g['passed'])}.", ""]
    if s is not None:
        lines += ["## Evaluation by route", "",
                  "| route | flights | completed | failed | mean duration (s) | mean progress (m) | mean cross-track (m) | mean return |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for p in dict.fromkeys(s["path"]):
            k = s["path"] == p
            lines.append(
                f"| {p} | {k.sum()} | {s['completed'][k].mean():.0%} | {s['failed'][k].mean():.0%} | "
                f"{s['duration_s'][k].mean():.1f} | {s['progress_m'][k].mean():.1f} | "
                f"{s['mean_cross_track_m'][k].mean():.2f} | {s['return'][k].mean():.1f} |")
        lines.append("")
    (out / "report.md").write_text("\n".join(lines))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="training output directory")
    ap.add_argument("--eval", type=Path, help="evaluation directory (default <run>/eval)")
    args = ap.parse_args()
    out = args.run / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})
    evaluation = args.eval or args.run / "eval"
    m = read_csv(args.run / "metrics.csv")
    g = read_csv(args.run / "eval.csv")
    s = read_csv(evaluation / "summary.csv")
    flights = read_csv(evaluation / "flights.csv")
    route_points = read_csv(evaluation / "routes.csv")
    meta = evaluation / "meta.json"
    route_spacing = json.loads(meta.read_text())["route_spacing_m"] if meta.exists() else 10.0
    if m is None:
        ap.error(f"no metrics.csv in {args.run}")
    training_plots(m, out)
    if g is not None:
        curriculum_plots(g, out)
    if s is not None:
        evaluation_plots(s, flights, route_points, out, route_spacing)
    print(report(args.run, m, g, s, out))
    print("Wrote", out, flush=True)


if __name__ == "__main__":
    main()
