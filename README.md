# robo-paramotor

MuJoCo model of a 1 m PEEK paramotor, with a JAX/MJX port for PPO training.

| Folder | Contents |
| --- | --- |
| `model/` | Aircraft definition: XML generator, generated `paramotor.xml` / `scene.xml`, parameters, aerodynamics, control helpers, sensor spec |
| `mjx/` | JAX/MJX simulation of the same model |
| `rl/` | PPO environment, route generation, training, evaluation, per-run analysis, multi-run rliable comparison, `requirements-rl.txt` |
| `viewer/` | Interactive MuJoCo viewer and the `view.sh` launcher |
| `tests/` | Aero, reporting, and RL test suites |
| `configs/` | Hyperparameter overrides (JSON) passed to training with `--config` |
| `docker/` | GPU training image and `train.sh` (train + evaluate + analyze in one command) |
| `data/flights/` | Logged viewer flights (odometry CSVs) |
| `docs/` | `MODEL_NOTES.md`, `RL.md`, the model-alignment report, papers, images |
| `todo/` | Current plan (`RL_TODO_v2.md`); superseded plans in `todo/archive/` |

Run everything from the repo root. Python modules are run with `python -m`:

```bash
python -m model.build_paramotor          # regenerate model/paramotor.xml + scene.xml
viewer/view.sh                           # open the interactive viewer
python -m pytest                         # all tests
docker/train.sh --test                   # all tests, inside the Docker image
docker/train.sh --name first --headed    # train + a window replaying each checkpoint's best flight
docker/train.sh --viewer                 # interactive MuJoCo viewer, from the Docker image
python -m rl.train --smoke --output runs/smoke
docker/train.sh --name first --updates 2000   # train + evaluate + plots in Docker -> runs/first/
docker/train.sh --name base --seeds 5         # 5 seeds -> runs/base/seed0..4
docker/train.sh --compare base high_lr        # rliable comparison -> runs/compare/
```

See [docs/MODEL_NOTES.md](docs/MODEL_NOTES.md) for the model and
[docs/RL.md](docs/RL.md) for training.
