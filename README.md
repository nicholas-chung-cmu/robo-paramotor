# robo-paramotor

MuJoCo model of a 1 m PEEK paramotor, with a JAX/MJX port for PPO training.

| Folder | Contents |
| --- | --- |
| `model/` | Aircraft definition: XML generator, generated `paramotor.xml` / `scene.xml`, parameters, aerodynamics, control helpers, sensor spec |
| `mjx/` | JAX/MJX simulation of the same model |
| `rl/` | PPO environment, route generation, training, evaluation, `requirements-rl.txt` |
| `viewer/` | Interactive MuJoCo viewer and the `view.sh` launcher |
| `tests/` | Aero, reporting, and RL test suites |
| `docker/` | GPU training image (`Dockerfile`, `compose.yaml`) |
| `data/flights/` | Logged viewer flights (odometry CSVs) |
| `docs/` | `MODEL_NOTES.md`, `RL.md`, the model-alignment report, papers, images |
| `todo/` | Current plan (`RL_TODO_v2.md`); superseded plans in `todo/archive/` |

Run everything from the repo root. Python modules are run with `python -m`:

```bash
python -m model.build_paramotor          # regenerate model/paramotor.xml + scene.xml
viewer/view.sh                           # open the interactive viewer
python -m pytest                         # all tests
python -m rl.train --smoke --output runs/smoke
docker compose -f docker/compose.yaml run --rm rl python -m rl.train --smoke --output runs/smoke
```

See [docs/MODEL_NOTES.md](docs/MODEL_NOTES.md) for the model and
[docs/RL.md](docs/RL.md) for training.
