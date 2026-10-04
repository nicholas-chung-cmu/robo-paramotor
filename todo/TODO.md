# TODO

Open work items. The phased model → training plan is in
[RL_TODO_v2.md](RL_TODO_v2.md); this list is for items that came up along the way.
Each item says why it matters and how to tell it is done.

## Reward

- [ ] **Evaluate the reward changes.** Two changes replaced the strict sparse
      reward (+1 only within 2 m, and a missed point was never cleared):
      progress shaping (`progress_reward_per_m`, 0.1 per metre of
      distance-to-go closed), and plane-crossing passes with a graded reward,
      exp(-(max(d - 2 m, 0) / 2 m)²) on the miss distance d. Compare against the
      old reward (shaping 0) on completion rate, miss distance per point and
      envelope failures. Still worth considering: speed or energy terms (speed
      made good, thrust use).
      **Done when:** a same-seed comparison shows the new reward trains at
      least as reliably, with smaller miss distances.
- [ ] **Log miss distance.** `completed` now means every plane was crossed, not
      every point hit. Add the per-pass miss distance (or a within-2 m hit
      count) to the step metrics so evaluation and `rl.analyze` report accuracy.

## Speed

- [x] **MuJoCo Warp backend for MJX.** Done, and the JAX backend is removed.
      The aero wrench is computed in JAX from
      `qpos`/`qvel` and passed as `xfrc_applied` to Warp's `step`. Native
      parity passes. At 4096 envs the env step is about 6.4×
      faster (47k vs 7.3k control steps/s, measured with another run sharing the GPU).
- [ ] **Benchmark full training on Warp** without another job on the GPU:
      s/update at 4096 and 8192 envs, and whether PPO (not physics) is now the
      bottleneck. Also try Warp's `graph_mode` options in `mjx.put_model`.
- [ ] **Skip sensors in the Warp substeps.** Warp's `step` always computes
      sensordata (including the accelerometer's extra pass), but only every
      20th substep's sensors are used. Look for a cheaper path, such as a
      model with sensors disabled for `step`.
- [ ] **Timestep study** (RL_TODO_v2 item 2.6). 0.5 ms → 1 ms halves every
      run's cost. It is the largest remaining lever, and it compounds with Warp.
- [ ] **Measure environment-count scaling.** Run 256 / 1024 / 2048 / 4096 /
      8192 envs, recording steps/s and peak GPU memory. The default is now 1024.
      With the closed-airfoil canopy (585 DOF) the Newton solver runs out of
      memory on the 16 GB RTX 5080 at 4096 envs (15.8 GB peak) and 3072
      (15.3 GB). 2048 ran one update (774 s, 339 steps/s) and then failed to
      allocate Newton's per-step 2048 × 592 × 592 matrix (2.9 GB) on update 2.
      It factors a dense nv × nv matrix per env, so memory grows with nv². CG
      fit 4096 envs in 11.2 GB (1,340 steps/s, but it hit its 10-iteration
      cap); kept Newton.
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

- [ ] **Re-tune the batch split.** Minibatches are now 32 (was 4): 128
      gradient steps per update instead of 16, at no measurable cost (8.7 s per
      update at 4096 envs either way; approx_kl about 0.004 over the first 6
      updates). Still open: compare learning curves against 4 minibatches on
      the same seed, and try the RL_TODO_v2 5.1 `rollout_steps` of 20–32.
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

- [ ] **Turn sensor noise back on.** Noise and bias are temporarily off by
      default (`EnvConfig.noise_std` and `bias_std` are `{}` in
      `rl/rl_env.py`) while the heading-frame observations are tried noise-free.
      Restore the defaults to `sensor_spec.defaults("noise_std")` /
      `("bias_std")` and the matching assert in
      `test_sensor_csv_is_complete_and_sourced`. Note the heading-frame
      acceleration now carries attitude error: a 1.15° tilt bias rotates about
      0.2 m/s² of gravity into the horizontal channels.
      **Done when:** the defaults match `sensors.csv` again and a noisy run
      trains about as well as the noise-free one.

- [ ] **Barometer noise at 20 Hz.** `sensors.csv` still uses the OSR ×8 noise
      figure, the highest setting that supports 100 Hz. At 20 Hz the BMP581
      allows a higher OSR with lower noise. Take the right row from the
      datasheet.
- [ ] **Policy memory.** The policy sees a 1 s sensor history and has no
      recurrent state. Options: a longer `history_seconds`, or an LSTM/GRU
      policy (a bigger PPO change).

## Deformable canopy (option D)

- [ ] **The canopy does not hold its shape.** With the flex skin, free flight
      from 6 m/s folds the chord within 0.2 s and spirals down (~5 m/s sink);
      `test_unpowered_glide` fails on this (L/D 0.22). A, B and C lines at
      every station do not fix it: the unsupported leading edge curls back
      (~4 cm radius at q ≈ 22 Pa). See docs/MODEL_NOTES.md §7. Candidate design
      changes: nose battens/rods along the leading edge (stiffened vertices or
      a stiffer leading-edge strip), more line attachments per station
      including near the leading edge, a thicker skin. Training cannot be
      meaningful until a design flies.
      **Done when:** a rigged design glides with L/D 2–5 and `test_unpowered_glide`
      passes without loosening it.
- [ ] **Aerodynamics of a flapping membrane.** Cell strips are flat-plate
      lift/drag; they are not a membrane pressure model, so collapse details are
      approximate. Option E (an external aero solver over the deformed shape)
      is the higher-fidelity path once a real wing exists.
- [ ] **Training cost.** About 11× slower than rigid (4.1k vs 47k control
      steps/s at 4096 envs; ~2 min per update). Levers: a coarser mesh (fewer
      chord stations), the timestep study, fewer solver iterations for the
      edge constraints, and skipping unused sensor passes.

## Physics priors for the policy

The policy is a plain MLP trained with model-free PPO; physics reaches it only
through the simulator and the observation/action design. These add cheap
priors without changing what runs on the vehicle. Model-based RL and a
residual policy on a classical controller were considered and ruled out.

- [ ] **Physically meaningful inputs.** Add per-frame features the network
      otherwise has to learn to compute, all from sensors already observed:
      flight-path angle (`atan2(v_z, |v_xy|)` from GPS velocity), ground-track
      turn rate (change in GPS course), and specific energy (barometric
      height + |v|²/2g, plus its rate of change). They go in the heading frame
      like the rest of the frame, so `frame_size`/`obs_size` and the
      layout test in `tests/test_rl.py` change, and old checkpoints stop loading.
      **Done when:** a same-seed comparison against the current observation
      shows faster learning or a lower envelope-failure rate (or no change, and
      the features are dropped).
- [ ] **Privileged critic** (asymmetric actor–critic; this is RL_TODO_v2 item
      5.2). The actor keeps the sensor observation; the critic also gets true
      state it is not allowed to deploy with: true angle of attack per panel
      (`physics.raw_alpha`), true velocity and body rates, brake line lengths,
      true position error to the route, and (once added) wind. Needs a second
      observation vector in `State`, a critic input in `ActorCritic`, and the
      rollout/GAE plumbing in `rl/train.py`. The deployed policy is unchanged.
      **Done when:** it trains at least as well as the shared-observation
      critic on the same seeds, ideally with less variance across seeds.
- [ ] **Left/right mirror augmentation.** Not implemented yet: nothing in
      `rl/` mirrors observations, actions or routes. (Random routes already
      contain both turn directions, since curvature phase is random, but that
      is not augmentation.) Mirror map: negate lateral components (y of
      heading-frame vectors and of the route preview, roll and yaw rates,
      lateral accel), swap left/right arm angles, negate the brake
      differential. Caveat: propeller reaction torque makes the vehicle
      asymmetric, so a mirrored transition is not exactly physical. Prefer a
      symmetry loss on the policy, penalizing
      ‖π(mirror(o)) − mirror(π(o))‖², with a small weight, over adding mirrored
      samples to PPO's on-policy batch. Check the asymmetry first: fly mirrored
      routes with a fixed policy and compare.
      **Done when:** a mirror test (observation and action maps applied twice
      are the identity) passes, and left vs right evaluation routes perform
      closer than without it.

## Tooling

- [ ] Optional W&B or TensorBoard logging behind a flag, for live curves
      and comparing many runs.
