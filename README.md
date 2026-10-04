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
| `todo/` | Open items (`TODO.md`), the phased plan (`RL_TODO_v2.md`); superseded plans in `todo/archive/` |

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

## Rewards

What the policy is trained on (`ParamotorEnv.step` in `rl/rl_env.py`). All
terms use the true simulated state, not the noisy sensors:

| Term | Value | When |
| --- | --- | --- |
| Point passed | exp(−(max(d − 2 m, 0) / 2 m)²): +1 within 2 m, 0.37 at 4 m, 0.02 at 6 m | each route point (10 m apart), when the vehicle crosses the plane through it perpendicular to the route; d is the 3D miss distance (`success_radius_m`, `pass_sigma_m`) |
| Progress shaping | +0.1 per metre (`progress_reward_per_m`) | every step: the drop in distance-to-go (distance to the target point + route length left after it); moving away is negative |
| Alpha range | +0.005 × fraction of wing strips inside the angle-of-attack range (`envelope_reward`) | every step; leaving the range no longer ends the episode |
| Smoothness | −`action_change_penalty` × ‖action − previous action‖², currently 0 (was 0.02) | every step |
| Route completed | +10 | crossing the last point's plane, however accurately (ends the episode) |
| Failure | −10, replaces that step's reward | ground contact, more than 35 m off the route sideways or 25 m vertically, or non-finite physics (ends the episode) |

A non-finite reward also counts as −10. Reaching the 300 s time limit ends the
episode with no penalty, and PPO bootstraps the value there. PPO discounts with
γ = 0.997 (about a 13 s horizon at 25 Hz). See [docs/RL.md](docs/RL.md) for details.

See [docs/MODEL_NOTES.md](docs/MODEL_NOTES.md) for the model and
[docs/RL.md](docs/RL.md) for training.
