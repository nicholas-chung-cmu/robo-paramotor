# Paramotor PPO

This trains a feedforward policy with JAX, MuJoCo MJX, Flax and Optax. The existing
interactive simulator still works independently. The JAX port uses the accepted
strip aerodynamics and mechanical brake tendons; it does not add the removed
brake-panel lift or drag.

## Install

Use a separate environment so the viewer's existing dependencies stay intact:

```bash
python3 -m venv .venv-rl
source .venv-rl/bin/activate
python -m pip install -r rl/requirements-rl.txt
python -m pip install 'jax[cuda13]==0.11.2'
python -c 'import jax; print(jax.devices())'
```

Run on Linux or WSL2 with an NVIDIA driver that supports CUDA 13. The CUDA wheels
include the runtime libraries. If your driver only supports CUDA 12, install
`jax[cuda12]==0.11.2` instead, in a fresh environment. Check the current
[JAX installation requirements](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu)
for your driver. Do not install both extras in one environment. Omitting the CUDA
extra provides a CPU installation for debugging. `JAX_PLATFORMS=cpu` explicitly
selects the CPU. GPU execution should print `CudaDevice`, not `CpuDevice`.

MJX may print that optional `warp` imports are unavailable. This implementation
explicitly uses the JAX backend, so Warp is not a required dependency.

MuJoCo and MJX are pinned together. `mjx/paramotor_mjx.py` uses a few private MJX
forward-stage functions to inject aerodynamic forces at the correct point in the
solver. Run the parity tests before upgrading either package.

## Docker

The container is the portable way to move between GPU machines. The host needs
only an NVIDIA driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html);
the JAX CUDA wheels inside the image bring their own CUDA runtime.

The one-command path is `docker/train.sh`. It builds the image if needed, then
trains, evaluates on the fixed routes and writes plots, all into `runs/<name>/`:

```bash
docker/train.sh --smoke                                  # ~1 min end-to-end check
docker/train.sh --name first --num-envs 256 --updates 2000
docker/train.sh --name first --resume --updates 1000     # continue the same run
docker/train.sh --analyze first                          # re-plot, also mid-training
docker/train.sh --name baseline --seeds 5 --updates 500  # runs/baseline/seed0..4
docker/train.sh --name high_lr --seeds 5 --updates 500 --config configs/high_lr.json
docker/train.sh --compare baseline high_lr               # rliable statistics + plots
```

Options it does not recognise are passed to `python -m rl.train`. The steps it
runs can also be done by hand:

```bash
# Host check: the driver is visible to Docker.
docker run --rm --gpus all ubuntu nvidia-smi

# Build once per machine (CUDA 13 default; JAX_CUDA=cuda12 for older drivers).
# All commands run from the repo root.
docker compose -f docker/compose.yaml build
docker compose -f docker/compose.yaml run --rm rl   # prints [CudaDevice(id=0)]

# Every command from "Run" below works the same way, prefixed:
docker compose -f docker/compose.yaml run --rm rl python -m pytest tests/test_rl.py -q
docker compose -f docker/compose.yaml run --rm rl python -m rl.train --smoke --output runs/smoke
docker compose -f docker/compose.yaml run --rm rl python -m rl.train --num-envs 256 --updates 1000 --output runs/first
```

The repo is mounted at `/workspace`, so code edits take effect without a rebuild
and `runs/` is written to the host. Rebuild only after changing
`rl/requirements-rl.txt`. Without a GPU, build the CPU image and run it without
compose's GPU reservation:

```bash
docker build -f docker/Dockerfile -t paramotor-rl:cpu --build-arg JAX_CUDA=cpu .
docker run --rm -v "$PWD:/workspace" paramotor-rl:cpu python -m tests.test_aero
```

The container is headless, so `rl.evaluate --view` and `viewer/view.sh` must run on the
host; copy the checkpoint out of `runs/` and view it there.

## Run

Run everything from the repo root; modules are invoked with `python -m`.

```bash
# Check physics, sensor timing, route geometry, and PPO math.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_rl.py -q
python -m tests.test_aero

# Compile and exercise rollout, optimization, and checkpoint writing.
python -m rl.train --smoke --output runs/smoke

# Initial training run. Start small on a laptop; increase after measuring memory.
python -m rl.train --num-envs 64 --updates 1000 --output runs/first

# Continue for another 1000 updates, keeping weights, optimizer and curriculum.
python -m rl.train --resume runs/first/checkpoint.pkl --updates 1000 --output runs/first

# Repeatable evaluation: seven routes, three seeds each.
python -m rl.evaluate runs/first/checkpoint.pkl --output runs/eval-first

# Watch one recorded evaluation flight, with its reference path overlaid.
python -m rl.evaluate runs/first/checkpoint.pkl --path left --episodes 1 --view
```

The test command disables unrelated globally installed pytest plugins (for example,
ROS launch-testing plugins).

The first rollout compiles and can take substantially longer than later updates.
The smoke run also exercises episode resets and the curriculum gate. It only checks execution; it does not produce a useful trained pilot.
Training writes `config.json`, `metrics.csv`, `eval.csv` (curriculum gate results), and `checkpoint.pkl`. Evaluation
writes per-step `flights.csv`, per-flight `summary.csv` and the target `routes.csv`, including failures and
how long each flight survived. Compare returns together with distance traveled,
tracking error, failure rate and time outside the aerodynamic envelope.

The defaults use 64 environments and 128 control steps per PPO rollout. This is a
starting point for a 6–8 GB laptop GPU, not a guaranteed memory or speed budget.
Reduce `--num-envs` to 16 or 32 if needed; try 128/256 once the smaller run works.
JAX preallocation is disabled by the entry points. Compiled programs are cached
under `runs/.jax_cache` so repeated launches can reuse them. Set
`JAX_COMPILATION_CACHE_DIR` to choose a different location. Changing environment count,
rollout length or observation shape triggers recompilation. The existing 2000 Hz
physics timestep is retained because the suspension is stiff. A small batch can
underutilize a GPU; benchmark after compilation.

Resume starts fresh physical episodes. It restores policy, optimizer, RNG and
curriculum, but does not serialize the complete simulation batch, so it is not
an exact continuation of the interrupted trajectories. Checkpoints use Python
pickle: load your own trusted checkpoint files.

## Configuration

Defaults live in `EnvConfig` in `rl/rl_env.py` and `PPOConfig` in `rl/train.py`. Override
only what you need through JSON, without editing code:

```json
{
  "env": {
    "control_hz": 25,
    "sensor_hz": 100,
    "gps_hz": 5,
    "history_hz": 25,
    "history_seconds": 1.0,
    "episode_seconds": 60.0,
    "curriculum": true,
    "thrust_max": 1.0,
    "noise_std": {"gyro": [0.0, 0.0, 0.0], "accel": 0.0, "gps_pos_xy": 0.0},
    "bias_std": {"gyro": 0.0},
    "bias_walk_std": {"gyro": 0.0},
    "gps_latency_s": 0.0,
    "gps_dropout": 0.0
  },
  "ppo": {"num_envs": 64, "rollout_steps": 128, "updates": 1000}
}
```

Save this as e.g. `my_training.json`, then pass `--config my_training.json`.
CLI batch-size, update-count and seed options override that file. Resume uses the
configuration stored in the checkpoint; CLI overrides still apply.

## Observations and actions

The policy sees only what the vehicle can measure. `model/sensors.csv` lists every
sensor: part, units, rate, white-noise σ, per-episode bias σ, the datasheet figure
each value comes from, and the channels deliberately left out. It is the single
source of truth: `EnvConfig.noise_std` and `bias_std` default to its values, and
`tests/test_rl.py` checks that the CSV and the environment agree.

| Frame entry | Source | Rate | Size |
| --- | --- | --- | --- |
| estimated attitude, 6-D (first two columns of R) | estimator from IMU + compass | 100 Hz | 6 |
| `gyro` | LSM6DSOX | 100 Hz | 3 |
| `accel` | LSM6DSOX | 100 Hz | 3 |
| `mag` | QMC5883L | 100 Hz | 3 |
| GNSS horizontal position | GM10 Pro V3 (u-blox M10) | 5 Hz, held | 2 |
| GNSS velocity | GM10 Pro V3 | 5 Hz, held | 3 |
| `prop_omega` | ESC telemetry | 100 Hz | 1 |
| `arm_pos_L`, `arm_pos_R` | servo telemetry | 100 Hz | 2 |
| barometric altitude | BMP581 | 100 Hz | 1 |
| GPS age, fresh-fix flag | | | 2 |

Not observed: true body velocity (`vel_body`; no airspeed sensor), world angular
velocity (`pod_angvel`; duplicates the gyro), brake-line lengths (no line
sensors), and GNSS altitude (no published vertical accuracy; the barometer
replaces it). Battery voltage and current are on the BOM but not modelled.

Sensors are sampled at 100 Hz. The policy receives one frame every 40 ms
(`history_hz = 25`) for the last second, plus five route preview points and the
previous action/filter velocity: 670 inputs with the defaults. History is filled
with the first reading at reset. The network has separate actor and critic MLPs,
each with two 128-unit tanh layers. It has no recurrent state.

Noise is white per sample; bias is drawn once per episode (`bias_std`) and can
random-walk (`bias_walk_std`, channel units/√s, default off: no datasheet gives
it). Keys of all three dictionaries are `sensors.csv` channel names, and each
accepts a scalar or one value per component. Unlisted values are zero, so
`"noise_std": {}` turns noise off. The attitude estimate is the true attitude
rotated by a small world-frame error (`attitude_roll_pitch`, `attitude_yaw`); it
is an estimator emulation, not an AHRS. GNSS values are held between fixes.
Optional latency uses a sample buffer; optional dropout holds the last
successful fix and increases its age. GPS latency is rounded to the sensor
period. Noise is applied before normalization and before GPS sample/hold.

Actions are `[thrust, left_brake, right_brake]` in `[-1, 1]`. They map to
`[0, thrust_max]` N and `[0, 3]` rad. `thrust_max` is 1.0 N, the hardware thrust
clamp, and equals the XML actuator range. Under power the propeller reaction
torque turns the vehicle; holding a line is left to the policy. The brake command
follows the same critically damped filter as the viewer, reaching about 99% of a
full step in one second. The physical servo and tendon dynamics still run
underneath. Propeller spin is synchronized when thrust changes, using the
viewer's thrust/RPM relation. `model/paramotor_control.py` contains this shared
actuator logic.

## Paths, reward and curriculum

Routes are spatial curves with no arrival times. Projection searches near the
previous path segment, avoiding jumps between branches at a figure-eight crossing.
The policy gets points 5, 10, 20, 40 and 60 m ahead of its estimated progress.
These are relative to measured GPS position and rotated into a horizontal frame
aligned with measured yaw. Each is divided by its lookahead distance. Ground truth
is used for rewards and termination, not for these guidance observations.

Random routes integrate smoothly changing horizontal curvature and vertical grade.
At full difficulty, curvature is bounded by 0.04/m (25 m minimum radius) and grade
by ±0.10 (vertical change / horizontal distance). At 6 m/s, that curvature is about
14°/s, below the turning rates measured in the recordings. These conservative
geometric limits are starting points; they do not guarantee every combined motion
is dynamically feasible. Fixed evaluation routes are straight, 60 m left/right
circles, an S-turn, figure eight, climb and descent. Climb/descent approach 8%/15%
grade then flatten, so the descent route stays above the ground from the default
launch altitude. `--path random` evaluates seeded random routes separately.

Per-step reward is forward path progress divided by nominal forward distance per
control step, multiplied by an exponential tracking-error factor. It also subtracts
quadratic horizontal/vertical errors and command changes. Progress is signed, so
flying backward does not earn progress reward. Failure gives −10 and reaching the
route endpoint gives +10. Following the route matters continuously; the policy is
not paid for waiting at a waypoint.

Failures include ground crossing, excessive tracking error, canopy below the pod,
nonfinite dynamics, or remaining outside the calibrated angle-of-attack interval
for over one second. A 60-second time limit truncates the episode. PPO bootstraps
value at time limits, but stops advantage propagation across all episode resets.
The aerodynamic coefficient clamp remains the accepted model's clamp; it is not a
physical stall model.

The curriculum begins with straight, level routes, gradually increases horizontal
curvature, then adds vertical variation above difficulty 0.5. Every 25 updates, a
separate deterministic, fixed-seed ten-second evaluation checks tracking, survival,
forward progress and aerodynamic-envelope compliance. Passing advances difficulty
by 0.2 for newly reset episodes. Set `curriculum: false` to train at full difficulty.
This gate is an early training aid; the longer fixed evaluation suite is the useful
comparison for saved policies.

## Code map

- `mjx/paramotor_mjx.py`: JAX aerodynamics and MuJoCo forward/integration pipeline.
- `model/paramotor_control.py`: shared thrust/spin and brake smoothing.
- `rl/routes.py`: route generation, local projection and lookahead sampling.
- `rl/rl_env.py`: reset/step, sensors, history, rewards and termination.
- `rl/train.py`: feedforward PPO, curriculum, logging and checkpoints.
- `rl/evaluate.py`: fixed-route evaluation, CSV export and optional replay.
- `tests/test_rl.py`: dynamics parity and learning-environment regression tests.

There is no Gym/Brax wrapper or separate learner framework: `vmap` batches the
functional environment, `lax.scan` runs physics/rollouts, and Optax updates the PPO
network. The initial version randomizes starting position and configured sensors;
wind, mass and aerodynamic coefficient randomization are not enabled.

## Validation performed

The October 2026 aero review corrects force references and reaction-torque routing;
see `MODEL_NOTES.md` section 6. All 10 RL checks pass on CPU with those changes,
but 3 powered-flight acceptance checks fail. Powered trim is not validated.
The GPU execution results below describe the earlier model revision.

Previously validated on the RTX 3060 Laptop GPU (6 GB): 36 existing aero checks, 10 RL
regression tests, PPO smoke training, checkpoint resume with 64 environments,
and CSV evaluation on all seven fixed routes. The normal 64-environment,
128-step, four-epoch configuration also completed two updates (16,384 control
transitions). Its second update ran at about 320 aggregate policy steps/second;
each policy step contains 80 physics steps. Compilation time and other hardware
will change throughput. These short runs validate execution, not learned flight
performance. The optional GUI replay's geometry API was checked; the viewer
window still needs a manual visual check.

Training uses 10 constraint-solver iterations and 5 line-search iterations;
both are configurable. Native/MJX numerical parity checks use matching solver
settings, in addition to checking the aerodynamic forces themselves.

## Tracking and analysis

`python -m rl.analyze runs/<name>` (or `docker/train.sh --analyze <name>`) reads
whatever the run has written so far and produces `runs/<name>/analysis/`:

- `training.png`: one point per PPO update, from the training rollouts themselves.
  Reward, finished-episode return, cross-track and altitude error, envelope
  violations and curriculum difficulty show whether the pilot is improving.
  Policy/value loss, entropy and approximate KL show whether PPO is healthy.
- `curriculum.png`: the fixed-seed gate run every `eval_every` updates, with
  each pass threshold drawn. Difficulty advances by 0.2 when all five pass.
- `evaluation.png`: per-route completion, failure, cross-track and progress from
  `rl.evaluate`, with top-down tracks drawn over the target route.
- `report.md`: the same headline numbers as text, to paste into notes.

Read them in this order. Training reward rises at fixed difficulty but drops
each time the curriculum steps up, so judge progress against the difficulty
trace, not reward alone. The gate flies only 10 s (250 steps) per seed, so a
policy can pass it and still fail the 60 s evaluation flights. The evaluation
is the real test.

## Comparing configurations (rliable)

One run per configuration is not enough to say one beats another: RL results
vary a lot between seeds. Train each configuration with several seeds (5 is a
reasonable minimum), then compare them with
[rliable](https://github.com/google-research/rliable) (Agarwal et al.,
NeurIPS 2021):

```bash
python -m rl.compare runs/baseline runs/high_lr        # or docker/train.sh --compare ...
```

Each run is scored per evaluation route as the distance flown along the route
divided by what cruise speed covers in the evaluation window, so 1.0 means it
flew the whole minute at speed without leaving the route. Routes are rliable's
tasks. `runs/compare/<a>-vs-<b>/` then holds:

- `aggregates.png`: IQM, median, mean and optimality gap, each with a 95%
  stratified-bootstrap confidence interval. IQM (the mean of the middle 50% of
  scores) is the headline number: robust to one lucky or crashed seed.
- `improvement.png`: probability that configuration X beats Y on a random
  route. A CI that excludes 0.5 is a real difference.
- `profiles.png`: fraction of (run, route) scores above each threshold; a
  curve that lies entirely above another dominates it.
- `learning.png`: IQM across seeds of training reward, curriculum difficulty
  and cross-track error against environment steps, i.e. how fast each improves.
- `report.md`: all of the numbers, plus per-route means and the environment
  steps each configuration needed to reach full difficulty.

Evaluate every run with the same `rl.evaluate` settings (the defaults) so they
fly identical routes; `rl.compare` refuses runs evaluated on different routes.
Put hyperparameter changes in a JSON file under `configs/` and pass it with
`--config`, so each configuration is recorded in its runs' `config.json`.
