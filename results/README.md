# Results

## ppo-20261005-002307 (best policy so far)

Update 190, 24.9 M environment steps, curriculum difficulty 0.2 (gentle
curves, tightest turn ~125 m). Trained with the height penalties, the stricter
curriculum gate and the exploration floor; before the sideways fix below.

| file | what |
| --- | --- |
| `checkpoint.pkl` | the policy (load with `rl.train.load_checkpoint`; the env config is inside) |
| `config.json`, `metrics.csv`, `eval.csv`, `endings.csv` | training config and logs |
| `analysis/` | `python -m rl.analyze` plots and report |
| `routes/` | `python -m rl.route_report`: 20 flights per route type for 60 s (tracks, sideways and height error over time with their average, error against curvature, `summary.csv`) |
| `videos/` | `python -m rl.record_flight`: 1 km straight route (level 0) and a difficulty-0.2 route, each sped up to 30 s |

Replay it: `docker/train.sh --viewer python -m rl.record_flight results/ppo-20261005-002307/checkpoint.pkl --difficulty 0.2 | ffmpeg ...`
(see the module docstring), or point `viewer.watch_training` at a copy in `runs/`.

### What it does well, and why it stopped improving

On the same 16 routes for 60 s (`run_comparison_60s.csv`) it was the best of
all runs: every flight survived, height error 0.4 m, sideways error 6.1 m.
Height is solved. Sideways is not: on a straight route all 20 flights drift to
the same side and settle ~9 m off, parallel to the route (`routes/cross_track.png`),
so the curriculum gate (sideways error under 3 m) stopped passing at 0.2.

Causes, fixed in `rl/rl_env.py` for later runs:
1. Nothing rewarded returning to the line: the pass reward is ~0 beyond ~5 m and
   progress shaping barely changes with a sideways offset. Now a sideways
   penalty applies at every difficulty (`lateral_penalty`).
2. The early heading reward rewarded flying parallel to the route segment, so
   turning back to the line scored worse. It now rewards heading toward the
   next route point (`heading_lookahead_points`).
3. The policy only saw route points 20-80 m ahead. It now also sees the target
   and the next point (`preview_index = (0, 1, 2, 4, 8)`).

Climbs, descents and turns tighter than ~100 m are untrained at difficulty 0.2
(`routes/tracks.png`).

## run_comparison_60s.csv

Every trained run's final policy on the same 16 routes (difficulty 0.2, full
length) for 60 s (`python -m rl.compare_tracking`). Older runs were trained on
an earlier canopy model or observation layout and do not fly in the current
simulator.
