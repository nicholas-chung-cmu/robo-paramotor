# TODO

Open work items. The phased model → training plan is in
[RL_TODO_v2.md](RL_TODO_v2.md); this list is for items that came up along the way.
Each item says why it matters and how to tell it is done.

## Speed

- [ ] **Try the MuJoCo Warp backend for MJX.** We use the JAX backend
      (`mjx.put_model(..., impl="jax")` in `mjx/paramotor_mjx.py`) because the
      aero forces are injected between MJX's private forward stages
      (`fwd_velocity` → aero → `fwd_actuation`/`fwd_acceleration`). Warp runs
      the step as hand-written GPU kernels, with CUDA graphs cutting the
      per-kernel launch overhead. That overhead is our bottleneck: each update
      runs 10,240 tiny physics substeps in sequence.
      Steps:
      1. Pin `warp-lang` and `mujoco-warp` to versions matching MuJoCo 3.13.
      2. Confirm Warp supports what the model uses: the `implicitfast`
         integrator, limited tendons, and position actuators.
      3. Re-plumb the aero: compute the wrench in JAX from the current
         positions and velocities and pass it as `xfrc_applied` to Warp's
         step, or port `ParamotorMJX.wrench` to a Warp kernel.
      **Done when:** the native-parity tests in `tests/test_rl.py` pass on the
      Warp backend, and a steps/s benchmark at equal `num_envs` shows the gain
      (or shows there is none).
- [ ] **Timestep study** (RL_TODO_v2 item 2.6). 0.5 ms → 1 ms halves every
      run's cost. It is the largest remaining lever, and it compounds with Warp.
- [ ] **Measure environment-count scaling.** Run 256 / 1024 / 2048 / 4096 /
      8192 envs, recording steps/s and peak GPU memory. The default is now 4096
      but has not been benchmarked. Also record the 16 GB ceiling.
- [ ] **Merge the sensor pass into the next substep's forward** (#4 from the
      speed review). `sample()` re-runs a full `forward()` every 20 substeps.
      Three of the four samples per control step could reuse the next
      substep's forward. The last one must stay, so the observation never
      sees the next action. Saves ~3–4% now, ~8% at a 1 ms timestep. Needs
      the sensor-timing and parity tests to pass. Best done after the
      timestep study.
- [ ] **Benchmark the batched evaluator** against the old one (1 checkpoint ×
      21 flights vs 3 checkpoints × 700 flights). This was interrupted.
- [ ] Optionally, train several seeds inside one JAX program (vmap over seeds),
      so they compile once and share the GPU.

## PPO at 4096 environments

- [ ] **Re-tune the batch split.** One update now collects 524,288 samples but
      takes only 16 gradient steps (4 minibatches × 4 epochs). Try the
      RL_TODO_v2 5.1 settings: ~32 minibatches and `rollout_steps` 20–32.
- [ ] **Scale the gate and checkpoint intervals** to environment steps, not
      updates. At 4096 envs, `eval_every=25` is every 13 M samples.
- [ ] **Stagger episode starts.** All envs launch together, so the first ~12
      updates see only the first seconds of flight. Randomize each env's
      starting step.

## Evaluation and curriculum

- [ ] **Lengthen the curriculum gate**, or add a longer check. It flies only
      10 s, so a policy can pass it and still fail 60 s flights by sinking or
      drifting.
- [ ] **Make route completion reachable.** Evaluation routes are ~1,000 m,
      but a 60 s flight at 6 m/s covers ~360 m, so `completed` is always 0.
      Either shorten the routes or lengthen the flights.

## Model and observations

- [ ] **Barometer noise at 20 Hz.** `sensors.csv` still uses the OSR ×8 noise
      figure, the highest setting that supports 100 Hz. At 20 Hz the BMP581
      allows a higher OSR with lower noise. Take the right row from the
      datasheet.
- [ ] **Policy memory.** The policy sees a 1 s sensor history and has no
      recurrent state. Options: a longer `history_seconds`, or an LSTM/GRU
      policy (a bigger PPO change).

## Tooling

- [ ] Optional W&B or TensorBoard logging behind a flag, for live curves
      and comparing many runs.
