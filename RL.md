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
python -m pip install -r requirements-rl.txt
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

MuJoCo and MJX are pinned together. `paramotor_mjx.py` uses a few private MJX
forward-stage functions to inject aerodynamic forces at the correct point in the
solver. Run the parity tests before upgrading either package.

## Run

```bash
# Check physics, sensor timing, route geometry, and PPO math.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest test_rl.py -q
python test_aero.py

# Compile and exercise rollout, optimization, and checkpoint writing.
python train.py --smoke --output runs/smoke

# Initial training run. Start small on a laptop; increase after measuring memory.
python train.py --num-envs 64 --updates 1000 --output runs/first

# Continue for another 1000 updates, keeping weights, optimizer and curriculum.
python train.py --resume runs/first/checkpoint.pkl --updates 1000 --output runs/first

# Repeatable evaluation: seven routes, three seeds each.
python evaluate.py runs/first/checkpoint.pkl --output runs/eval-first

# Watch one recorded evaluation flight, with its reference path overlaid.
python evaluate.py runs/first/checkpoint.pkl --path left --episodes 1 --view
```

The test command disables unrelated globally installed pytest plugins (for example,
ROS launch-testing plugins).

The first rollout compiles and can take substantially longer than later updates.
The smoke run also exercises episode resets and the curriculum gate. It only checks execution; it does not produce a useful trained pilot.
Training writes `config.json`, `metrics.csv`, and `checkpoint.pkl`. Evaluation
writes per-step `flights.csv` and per-flight `summary.csv`, including failures and
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

Defaults live in `EnvConfig` in `rl_env.py` and `PPOConfig` in `train.py`. Override
only what you need through JSON, without editing code:

```json
{
  "env": {
    "control_hz": 25,
    "sensor_hz": 100,
    "gps_hz": 5,
    "history_seconds": 1.0,
    "episode_seconds": 60.0,
    "curriculum": true,
    "thrust_max": 1.7,
    "noise_std": {"gyro": [0.0, 0.0, 0.0], "accel": 0.0, "pod_pos": 0.0},
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

All 13 XML sensors (30 values) are retained in XML order. Each 100 Hz observation
frame adds GPS age and a fresh-fix flag. The policy receives the last second of
these frames, five route preview points and the previous action/filter velocity:
3,220 inputs with the defaults. History is filled with the first reading at reset.
The network has separate actor and critic MLPs, each with two 128-unit tanh layers.
It has no recurrent state.

| Sensor | Units | Update rate |
| --- | --- | --- |
| `pod_quat` | quaternion, wxyz | 100 Hz |
| `gyro` | rad/s, IMU frame | 100 Hz |
| `accel` | m/s², IMU specific force | 100 Hz |
| `mag` | model magnetic-field units, IMU frame | 100 Hz |
| `vel_body` | m/s, IMU frame | 100 Hz |
| `pod_pos` | m, world coordinates | 5 Hz GPS |
| `pod_vel` | m/s, world coordinates | 5 Hz GPS |
| `pod_angvel` | rad/s, world frame | 100 Hz |
| `prop_omega` | rad/s | 100 Hz |
| `arm_pos_L`, `arm_pos_R` | rad | 100 Hz |
| `brake_len_L`, `brake_len_R` | m | 100 Hz |

GPS values are held between fixes. Optional latency uses a sample buffer; optional
dropout holds the last successful fix and increases its age. Noise is generated
with explicit JAX RNG keys, independently per environment. `noise_std` is white
noise standard deviation **per reading**, `bias_std` draws a constant initial bias
per episode, and `bias_walk_std` is bias random walk in sensor units/√second.
Each sensor accepts a scalar or one value per component. Unlisted values are zero.
Convert datasheet noise densities to per-sample values using your sensor bandwidth
before filling them in. GPS latency is rounded to the fast sensor sampling period.
Reset bootstraps the buffer with an initial fix.

Noise is applied before normalization and before GPS sample/hold. Quaternion
component noise is renormalized; it is a configurable approximation, not an AHRS
model. The XML also supplies orientation, body velocity and world angular velocity
as ideal derived channels. They remain available as requested; deployment needs
corresponding onboard estimates. No barometer channel is invented because none is
currently defined in the XML.

Actions are `[thrust, left_brake, right_brake]` in `[-1, 1]`. They map to
`[0, thrust_max]` N and `[0, 3]` rad. The brake command follows the same critically
damped filter as the viewer, reaching about 99% of a full step in one second.
The physical servo and tendon dynamics still run underneath. Propeller spin is
synchronized when thrust changes, using the viewer's thrust/RPM relation.
`paramotor_control.py` contains this shared actuator logic. The 1.7 N default is
based on the recorded climb flight; it can be lowered in configuration and cannot
exceed the XML's 2 N actuator limit.

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

- `paramotor_mjx.py`: JAX aerodynamics and MuJoCo forward/integration pipeline.
- `paramotor_control.py`: shared thrust/spin and brake smoothing.
- `paths.py`: route generation, local projection and lookahead sampling.
- `rl_env.py`: reset/step, sensors, history, rewards and termination.
- `train.py`: feedforward PPO, curriculum, logging and checkpoints.
- `evaluate.py`: fixed-route evaluation, CSV export and optional replay.
- `test_rl.py`: dynamics parity and learning-environment regression tests.

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
