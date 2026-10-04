# Paramotor PPO

This trains a closed-loop control policy with JAX, MuJoCo MJX, Flax and Optax:
every 40 ms it maps the sensor history and route preview to thrust and brake
commands. The network is a plain MLP with no recurrent state. The existing
interactive simulator still works independently. The JAX port uses the accepted
strip aerodynamics on the deformable canopy (a flex shell, see
docs/MODEL_NOTES.md §7) and mechanical brake tendons; it does not add the
removed brake-panel lift or drag.

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

Physics runs on MJX's MuJoCo Warp backend (`warp-lang`, pinned via
`mujoco-mjx[warp]` in `rl/requirements-rl.txt`). Each substep is one Warp `step`
call, including the canopy skin's bending and edge constraints. Warp runs on the
GPU, and on the CPU for `--cpu` debugging. With the deformable canopy it runs
about 4.1k control steps/s at 4096 envs (RTX 5080), roughly 11× slower than the
old rigid canopy (47k), so one PPO update takes about 2 minutes.

The aerodynamic forces are computed in JAX from `qpos`/`qvel` alone (canopy
vertices are world-axis slide bodies, the pod a free body), with the same mesh
geometry functions as the native model, and applied as `xfrc_applied`. That
code avoids matrix products, and the pod's use full float32 precision, because
GPU matmuls default to TF32 (about 1e-3 relative error).
MuJoCo, MJX and Warp are pinned together; run the parity tests before upgrading.
`docker/train.sh` keeps Warp's compiled kernels in `runs/.warp_cache`.

## Docker

The container is the portable way to move between GPU machines. The host needs
only an NVIDIA driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html);
the JAX CUDA wheels inside the image bring their own CUDA runtime.

The one-command path is `docker/train.sh`. It builds the image if needed, then
trains, evaluates on the fixed routes and writes plots, all into `runs/<name>/`:

```bash
docker/train.sh --smoke                                  # ~1 min end-to-end check
docker/train.sh --name first --updates 2000
docker/train.sh --name first --resume --updates 1000     # continue the same run
docker/train.sh --analyze first                          # re-plot, also mid-training
docker/train.sh --name baseline --seeds 5 --updates 500  # runs/baseline/seed0..4
docker/train.sh --name high_lr --seeds 5 --updates 500 --config configs/high_lr.json
docker/train.sh --compare baseline high_lr               # rliable statistics + plots
docker/train.sh --test                                   # test suite, live output
docker/train.sh --name first --headed                    # train + watch the best flight per checkpoint
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
docker compose -f docker/compose.yaml run --rm rl python -m rl.train --updates 1000 --output runs/first
```

The repo is mounted at `/workspace`, so code edits take effect without a rebuild
and `runs/` is written to the host. Rebuild only after changing
`rl/requirements-rl.txt`. Without a GPU, build the CPU image and run it without
compose's GPU reservation:

```bash
docker build -f docker/Dockerfile -t paramotor-rl:cpu --build-arg JAX_CUDA=cpu .
docker run --rm -v "$PWD:/workspace" paramotor-rl:cpu python -m tests.test_aero
```

On a Linux desktop (X11 or XWayland) the container can also open windows.
Windows share the host's X socket and this session's X auth cookie (no
`xhost +` needed), and the NVIDIA runtime injects its OpenGL driver, so they
render on the GPU:

```bash
docker/train.sh --name first --headed --updates 2000   # train + watch it learn
docker/train.sh --watch first                    # watch a run that is already training
docker/train.sh --viewer                         # interactive MuJoCo viewer (free flight)
docker/train.sh --viewer --sweep --zoom 12       # any viewer/view.sh options
docker/train.sh --viewer python -m rl.evaluate runs/first/checkpoint.pkl \
    --path left --episodes 1 --view              # replay one evaluation flight
```

The training watcher (`viewer/watch_training.py`) waits for each new
checkpoint, flies the latest policy on 32 fixed-seed routes at the run's current
curriculum difficulty, and replays the flight that got furthest along its route,
with the route drawn and an overlay: update, environment steps, difficulty,
the best and median distance, and whether this is a new best. It evaluates in a
background thread on the GPU, sharing it with training (a few percent of
training throughput), and keeps the previous best flight playing meanwhile.
With `--seeds` it follows whichever seed is training. The window stays open
after training ends; close it to stop the watcher.

A `--view` replay writes its one flight to `<run>/replay/`, never over
`<run>/eval/`. On macOS, run `viewer/view.sh` on the host instead (the viewer
needs `mjpython` there, and Docker Desktop has no display).

## Run

Run everything from the repo root; modules are invoked with `python -m`.

```bash
# Check physics, sensor timing, route geometry, and PPO math.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_rl.py -q
python -m tests.test_aero

# Compile and exercise rollout, optimization, and checkpoint writing.
python -m rl.train --smoke --output runs/smoke

# Initial training run (4096 parallel environments by default).
python -m rl.train --updates 1000 --output runs/first

# On a laptop GPU, start smaller and increase after measuring memory.
python -m rl.train --num-envs 256 --updates 1000 --output runs/first

# Continue for another 1000 updates, keeping weights, optimizer and curriculum.
python -m rl.train --resume runs/first/checkpoint.pkl --updates 1000 --output runs/first

# Repeatable evaluation: seven routes, 20 seeds each -> runs/first/eval/.
python -m rl.evaluate runs/first/checkpoint.pkl

# Many seeds' checkpoints at once, 100 flights per route each (one batch,
# about the time of a single checkpoint). A progress bar shows simulated time,
# flights still airborne and failures.
python -m rl.evaluate runs/base/seed*/checkpoint.pkl --episodes 100

# Watch one recorded evaluation flight, with its reference path overlaid.
python -m rl.evaluate runs/first/checkpoint.pkl --path left --episodes 1 --view
```

The test command disables unrelated globally installed pytest plugins (for example,
ROS launch-testing plugins). Without a local environment, run the suite in the
image with live output: `docker/train.sh --test` (pytest arguments may follow,
e.g. `docker/train.sh --test tests/test_rl.py -k gps`).

Evaluation cost is set by the number of sequential physics steps (the episode length of
flight), not by the number of flights, so more flights per route are nearly
free and give tighter statistics. Every flight's summary goes to `summary.csv`;
only the first `--trace` (default 3) episodes per route are recorded step by
step in `flights.csv`, which keeps thousands of flights cheap. Evaluation stops
early once every flight has ended.

The first rollout compiles and can take substantially longer than later updates.
The smoke run also exercises episode resets and the curriculum gate. It only checks execution; it does not produce a useful trained pilot.
Training writes `config.json`, `metrics.csv`, `eval.csv` (curriculum gate results), and `checkpoint.pkl`. Evaluation
writes per-step `flights.csv`, per-flight `summary.csv` and the target `routes.csv`, including failures and
how long each flight survived. Compare returns together with distance traveled,
tracking error, failure rate and time outside the aerodynamic envelope.

The defaults use 4096 environments and 128 control steps per PPO rollout, sized
for a 16 GB desktop GPU (RTX 5080). Each step's physics is tiny, so a small batch
leaves the GPU mostly idle: time per update is set by the 10,240 sequential
physics substeps per rollout, and more environments add data at little extra
time until the GPU saturates. On a 6–8 GB laptop GPU use `--num-envs 256` or
lower. One update now collects 4096 x 128 = 524,288 samples, so compare runs by
environment steps, not by update count.
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
    "episode_seconds": 300.0,
    "curriculum": true,
    "thrust_max": 2.0,
    "noise_std": {"gyro": [0.0, 0.0, 0.0], "accel": 0.0, "gps_pos_xy": 0.0},
    "bias_std": {"gyro": 0.0},
    "bias_walk_std": {"gyro": 0.0},
    "gps_latency_s": 0.0,
    "gps_dropout": 0.0
  },
  "ppo": {"num_envs": 4096, "rollout_steps": 128, "updates": 1000}
}
```

Save this as e.g. `my_training.json`, then pass `--config my_training.json`.
CLI batch-size, update-count and seed options override that file. Resume uses the
configuration stored in the checkpoint; CLI overrides still apply.

## Observations and actions

The policy sees only what the vehicle can measure. `model/sensors.csv` lists every
sensor: part, units, rate, white-noise σ, per-episode bias σ, the datasheet figure
each value comes from, and the channels deliberately left out. It is the single
source of truth for noise: `sensor_spec.defaults("noise_std")` and
`("bias_std")` read it, and `tests/test_rl.py` checks that the CSV and the
environment agree. **Noise is temporarily off by default**: `EnvConfig.noise_std`
and `bias_std` default to `{}` while the heading-frame observations are tuned
(see `todo/TODO.md`). Pass the `sensor_spec` defaults to turn it back on.

Every vector in a frame is in the **heading frame**: world axes rotated by the
estimated yaw, z up. The route preview uses the same frame, so the policy sees
the same inputs whichever way it points. For that reason absolute GNSS x/y and
the compass are not in the frame (the compass still feeds the yaw estimate).

| Frame entry | Source | Rate | Size |
| --- | --- | --- | --- |
| estimated roll/pitch, 6-D (first two columns of R with yaw removed) | estimator from IMU + compass | 100 Hz | 6 |
| `gyro` (body frame) | LSM6DSOX | 100 Hz | 3 |
| `accel` specific force, gravity included (`R·accel`, heading frame) | LSM6DSOX + attitude | 100 Hz | 3 |
| GNSS velocity (heading frame) | GM10 Pro V3 (u-blox M10) | 5 Hz, held | 3 |
| `prop_omega` | ESC telemetry | 100 Hz | 1 |
| `arm_pos_L`, `arm_pos_R` | servo telemetry | 100 Hz | 2 |
| barometric altitude | BMP581 | 100 Hz | 1 |
| GPS age, fresh-fix flag | | | 2 |

Not observed: true body velocity (`vel_body`; no airspeed sensor), world angular
velocity (`pod_angvel`; duplicates the gyro), brake-line lengths (no line
sensors), and GNSS altitude (no published vertical accuracy; the barometer
replaces it). Battery voltage and current are on the BOM but not modelled.

Sensors are sampled at 100 Hz. The policy receives one frame every 40 ms
(`history_hz = 25`) for the last second, plus three route preview points (20, 40
and 80 m ahead along the route, relative to the vehicle and divided by their
distance) and the previous action/filter velocity: 539 inputs with the defaults.
Routes are ~1 km, stored as 101 points 10 m apart (`route_points`,
`route_spacing_m`); projection and preview interpolate linearly between them.
Checkpoints saved before these fields existed load with their original 513
points 2 m apart. History is filled
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

Actions are `[thrust, brake, diff]` in `[-1, 1]`. Thrust maps to `[0, thrust_max]` N.
`brake` maps to `b` in `[0, 1]` and `diff` is used as is (positive = right brake).
They mix into `left = clip(b - diff, 0, 1)` and `right = clip(b + diff, 0, 1)`,
each scaled to `[0, 3]` rad of line travel. `thrust_max` is 2.0 N, the hardware thrust
clamp, and equals the XML actuator range. Under power the propeller reaction
torque turns the vehicle; holding a line is left to the policy. The brake command
follows the same critically damped filter as the viewer, reaching about 99% of a
full step in one second. The physical servo and tendon dynamics still run
underneath. Propeller spin is synchronized when thrust changes, using the
viewer's thrust/RPM relation. `model/paramotor_control.py` contains this shared
actuator logic.

## Paths, reward and curriculum

Routes are spatial curves with no arrival times: 101 points, 10 m apart. The
vehicle works through them in order. The target is the first point not yet passed.
It is passed when the true position crosses the plane through the point,
perpendicular to the route there (the tangent from the previous to the next
point), however far off the vehicle is, so a miss never stalls the route. The
miss distance is the closest approach to the point during that step. The policy gets the points 2, 4 and 8
indices past the target (`preview_index`, so 20, 40 and 80 m at 10 m spacing),
clamped to the last point. They are relative to measured position (GPS x/y and
barometer) and rotated into a horizontal frame aligned with measured yaw. Each is
divided by its look-ahead distance. Ground truth is used for passing points,
rewards and termination, not for these guidance observations. There is no
projection or search window. Cross-track and altitude error, for metrics and
failure, are measured against the segment into the target.

Random routes integrate smoothly changing horizontal curvature and vertical grade.
At full difficulty, curvature is bounded by 0.04/m (25 m minimum radius) and grade
by ±0.10 (vertical change / horizontal distance). At 6 m/s, that curvature is about
14°/s, below the turning rates measured in the recordings. These conservative
geometric limits are starting points; they do not guarantee every combined motion
is dynamically feasible. Fixed evaluation routes are straight, 60 m left/right
circles, an S-turn, figure eight, climb and descent. Climb/descent approach 8%/15%
grade then flatten, so the descent route stays above the ground from the default
launch altitude. `--path random` evaluates seeded random routes separately.

Each pass earns a **graded reward** on the 3D miss distance d:
`exp(-(max(d - success_radius_m, 0) / pass_sigma_m)²)`. That is 1 within 2 m,
0.78 at 3 m, 0.37 at 4 m and 0.02 at 6 m with the defaults (both 2 m). The
reward also has -10 on failure, +10 for passing the last point's plane (however
accurately), a small **alpha-range reward** (`envelope_reward`, 0.005 per control
step times the fraction of spanwise strips inside the aero model's
angle-of-attack interval, at most 0.125/s), and **progress shaping**:
`progress_reward_per_m` (0.1) times the drop in distance-to-go each step.
Distance-to-go is the true 3D distance to the target point plus the route length
remaining after it. It is nearly continuous when a point is passed accurately;
on a wide miss it drops when the target advances, so shaping alone does not
punish misses (the graded pass reward does). It is a plain difference with no
terminal term: over a flight it sums to 0.1 × metres gained, so ending the
episode early earns nothing extra. A full 1 km route is worth about 100 from
shaping, next to up to 100 from passes (points 1 to 100). Set it to 0 to train on
the pass rewards alone. The penalty on command changes (`action_change_penalty`,
was 0.02 times the squared action change) is off by default.

Failures include ground crossing, excessive tracking error (35 m horizontal or
25 m vertical from the target segment), or nonfinite dynamics. The canopy
dropping below the pod no longer ends an episode either. Leaving the calibrated angle-of-attack interval no longer
ends an episode; it only forgoes the alpha-range reward (each spanwise strip,
its incidence averaged over the chord by area). A 300-second time limit truncates the episode. PPO bootstraps
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

- `mjx/paramotor_mjx.py`: JAX aerodynamics and the MJX (Warp) physics step.
- `model/paramotor_control.py`: shared thrust/spin and brake smoothing.
- `rl/routes.py`: route generation, local projection and lookahead sampling.
- `rl/rl_env.py`: reset/step, sensors, history, rewards and termination.
- `rl/train.py`: PPO for the control policy (MLP, no recurrent state), curriculum, logging and checkpoints.
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
trace, not reward alone. The gate flies each seed's whole episode at the
current difficulty (to the end of the route, a failure, or the time limit), but
only on 8 fixed seeds, so the evaluation is still the real test.

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
