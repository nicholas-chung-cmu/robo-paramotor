# TODO — what the paramotor MJCF needs before a PPO policy can control it

> Implementation update: the JAX/MJX feedforward PPO environment, training and
> evaluation entry points now live in this repo. See [RL.md](RL.md) for the active
> design, configuration and commands. The checklist below is historical planning;
> its CPU/Gym wrapper, observation and brake-model proposals are superseded where
> they differ from that implementation.

Scope: everything between the model as it stands today (`paramotor.xml` +
`paramotor_aero.py`) and a PPO run that produces a usable control law.
Grouped by whether it *blocks* training, *invalidates* training, or improves it.
Source of truth for the model is `MODEL_NOTES.md`; hardware is
`../itemized_spec.md`.

Measured baseline, 2026-09-23: `nq=17 nv=15 nu=3 nsensordata=30`,
`dt=5e-4`, **3.6x realtime single-threaded** with strip aero on.

---

## 0. Blockers — nothing trains until these exist

- [ ] **Gymnasium env wrapper** (`paramotor_env.py`). There is no env today, only
      a viewer. Needs `reset()`, `step()`, `observation_space`, `action_space`,
      `render()`. Build it around the same three lines the viewer uses
      (`MjModel` + `MjData` + the aero callback), not around `view_paramotor.py`.
- [ ] **Decide the global-callback problem.** `mujoco.set_mjcb_passive()` is a
      *process-global* hook, and `ParamotorAero` is bound to one `MjModel` at
      construction (`self.canopy`, `self.panels` are ids into that model). Two
      envs in one process will feed the wrong model to the wrong callback.
      Options, pick one and write it down:
      - `SubprocVecEnv` — one model per process. Simplest, works today.
      - A dispatcher callback keyed on `model` identity (`id(model)` → aero
        instance) installed once at import.
      - Call the aero object manually before `mj_step` and write into
        `qfrc_applied`. Loses the "it is a passive force" property.
- [ ] **Control decimation.** Physics is 2000 Hz; the policy must not be. Pick a
      control rate (**50 Hz → `frame_skip=40`** is the natural first choice, it
      matches a realistic outer loop) and step `frame_skip` times per `step()`.
      Without this, an episode is 100k+ policy steps and PPO is hopeless.
- [ ] **Trim / initial state.** The model ships with **no `<key>` keyframe**.
      `view_paramotor.py` reaches trim by launching at 6 m/s and settling ~10 s
      of sim — far too slow to pay per episode. Add a `<keyframe>` to
      `build_paramotor.py` holding the settled glide state (qpos, qvel, and
      `prop_spin` velocity consistent with thrust), then `reset()` is
      `mj_resetDataKeyframe` + randomization.
- [ ] **Port `sync_prop()` out of the viewer.** It is currently inlined in
      `view_paramotor.py` and `MODEL_NOTES.md` §5 tells every consumer to copy
      it. Thrust and rotor speed are one physical thing; a policy that writes
      `ctrl[thrust]` without also setting `qvel[prop_spin]` gets force with no
      angular momentum and therefore **no gyroscopic moment**. Move it to a
      shared module and call it inside `step()`.
- [ ] **Observation vector.** The 13 sensors give 30 numbers, all ground truth.
      Decide what the policy actually sees, and keep it to what the real vehicle
      has (LSM6DSOX, BMP581, GNSS, compass — see the `<sensor>` block). Do not
      feed it `framepos`/`framelinvel` of the pod unless GNSS is genuinely in the
      loop at that rate. Quaternion → 6D rotation representation.
- [ ] **Action vector.** `nu=3`: `thrust`, `servo_pos_L`, `servo_pos_R`. Note
      `MODEL_NOTES.md` §5 still claims 4 actuators including `propeller_speed` —
      **the built XML has 3**. Fix the doc or the generator. Normalize actions to
      [-1,1] and map onto `ctrlrange` explicitly.
- [ ] **Reward and task.** Nothing defines what "control" means yet. Pick the
      first task concretely — heading hold, altitude hold, or waypoint capture —
      and write the reward before writing the env, not after.
- [ ] **Termination conditions.** Collision is disabled everywhere
      (`contype=0 conaffinity=0`), so there is no ground contact to terminate on.
      Terminate on altitude below the ground plane (z = -18 m in `scene.xml`),
      on canopy tumble (|α| outside the envelope, or canopy-below-pod), on line
      slackness, and on a time limit.

## 1. Physics that will silently invalidate a trained policy

//this is an acceptable approximation and will be how we build the physical model 
- [ ] **Brakes produce no aerodynamic effect.** `brake_wrench_frd()` returns
      zero. The only thing a brake command does today is rotate the rigid canopy
      geometrically through the tendons. A PPO policy will learn *that* mechanism,
      which is not the real one. This is the single biggest correctness gap for a
      controls task. Either identify C_Lδa / C_Ddelta_a / C_lδa / C_nδa on the
      vehicle, or implement model **B** from `MODEL_NOTES.md` §7 (per-panel
      spanwise twist DOFs) so brake authority comes out of geometry. §7 already
      recommends B as the next step.
      - Note: if panels get their own DOFs, the strip fast path **raises** —
        `ParamotorAero.__init__` explicitly rejects `body_dofnum != 0` panels.
        `_apply_strip` must move to `mj_objectVelocity` per panel first.

//there is a hardware clamp that is acceptable
- [ ] **Thrust above ~1.0 N departs.** Documented open issue (§6): above ~1 N the
      suspension goes fully slack, the canopy tumbles, α hits ±180°, and the
      linear no-stall C_L means nothing there. A PPO policy *will* find this and
      exploit it. Mitigate in this order:
      1. Clamp the thrust action to the validated envelope (0–1.0 N) for the
         first runs, and assert the clamp in the env.
      2. Add a stall model so α excursions are penalized by physics, not by
         reward shaping.
      3. Re-trim the rigging.
- [ ] **α clamping is invisible to the learner.** `aero.envelope_report()` counts
      excursions; the policy never sees them. Surface `n_alpha_clamped` /
      `alpha_raw_min/max` per episode into `info` and log it. A run where the
      clamp fires often is a run trained on fiction.
- [ ] **No actuator lag.** Motor τ≈0.45 s (spec §3.6) and servo dynamics are not
      modelled; `<position>` servos respond instantly up to `forcerange`. A
      policy trained on instant actuators is the classic sim-to-real failure.
      Add first-order lag on thrust and use the `torque` servo mode with a
      realistic loop, or add `<general dyntype="filter">`.
- [ ] **Servo force cap is 8x the real hardware.** `forcerange = ±2.0 N·m` vs the
      BD10BL-CAN datasheet stall of **0.275 N·m** at 7.4 V (§5). Set
      `SERVO_TAU_CAP = 0.275` before training or the policy gets authority the
      vehicle does not have.
- [ ] **`SERVO_ARMATURE = 5e-5` is an identification parameter, not a datasheet
      value** (§5). It is doing real work numerically — without it the capstan
      constraint broke down. Get it from a bench step response, and put it in the
      domain-randomization list until then.
- [ ] **Bilateral brake coupling — brakes are pull-only in reality.** §10: "a
      released brake line does not go slack the way a real one does." A policy
      that learns to *push* a brake line is learning a sim artifact.

## 2. Throughput — 3.6x realtime is not enough

- [ ] **Budget the run.** At 50 Hz control, one policy step ≈ 5.5 ms wall. 5M
      steps ≈ **7.6 hours single-threaded**. Decide the budget before choosing
      the approach.
- [ ] **Vectorize.** `SubprocVecEnv` with 8–16 workers is the low-effort path and
      also resolves the global-callback problem above. Measure actual
      steps/second; the Python aero callback is the bottleneck, not MuJoCo.
- [ ] **Profile the aero callback.** It runs at 2000 Hz and does a per-panel
      numpy pass plus a Python `mj_applyFT` loop over bodies. Two cheap wins:
      hoist the `np.flatnonzero` loop, and evaluate whether aero can run at a
      lower rate than physics (it is smooth; every 4th step may be enough).
- [ ] **Decide on MJX, explicitly, and early.** MJX would give 100–1000x on GPU,
      but **`paramotor_aero.py` is not jittable as written** (Python loops,
      `mj_applyFT`, `mj_objectVelocity`, in-place numpy). Porting strip theory to
      JAX is a real project. Either commit to it now or commit to CPU
      vectorization now — do not half-do both.
- [ ] **Consider a larger timestep.** `dt=5e-4` with `implicit`. Test whether
      `1e-3` or `2e-3` is stable with the tendon constraints and the servo
      armature; that is a free 2–4x.

## 3. Sim-to-real — needed if the policy is ever meant to fly

- [ ] **Sensor noise, bias and delay.** All sensors read ground truth: MuJoCo
      3.13 dropped sensor noise, so the `noise` attribute parses and is ignored
      (see the `<sensor>` comment). Write an observation-corruption layer — white
      noise, gyro bias random walk, GNSS latency and rate (it is *not* 50 Hz),
      baro drift, magnetometer hard/soft iron.
- [ ] **Sensor rates.** GNSS is ~10 Hz, IMU ~200 Hz+, baro ~50 Hz. Feeding all of
      them at the control rate trains a policy on information it will not have.
- [ ] **Domain randomization.** The parameter file tags every coefficient
      `[PAPER]` / `[AR]` / `[ROBOT]` / `[PROVISIONAL]`. Randomize the
      `[PROVISIONAL]` and `[PAPER]` ones in particular — `CD0`, `Cm0`, `Cma`,
      `Cnr` are explicitly flagged "flying on these is fine; sizing anything on
      them is not." Also: all-up mass, battery position on its rail (§1 says it
      trims the CG), `k_T`, servo armature, wind.
- [ ] **Wind and turbulence.** `--wind` in the viewer is a launch speed, not a
      wind field. Aero uses body velocity directly; add a wind vector to the
      relative-velocity computation in `_apply_strip` / `_apply_lumped` and
      randomize it per episode, plus gusts.
- [ ] **`k_T` is anchored on an assumed 10 krpm static bench point** (§5).
      Re-fit from a thrust stand; until then randomize it.
- [ ] **Deployment path.** Decide now how the policy leaves Python: control rate
      on the MCU, ONNX vs hand-written inference, the CAN servo interface, and
      what the fallback is when inference misses its deadline.

## 4. Infrastructure

- [ ] **Seeding and determinism.** Seed MuJoCo state, the aero randomization and
      the policy separately; verify a fixed seed reproduces an episode exactly.
- [ ] **A scripted baseline controller.** PPO needs something to beat. A PID on
      heading via differential brake is enough, and it doubles as a check that
      the brake channel has any authority at all (see §1).
- [ ] **Episode logging.** Per-episode: reward terms broken out, α excursion
      count, line slack events, thrust saturation, servo torque saturation.
- [ ] **Extend `test_aero.py` to the env.** The suite's 38 assertions cover the
      aero layer. Add: reset is deterministic, obs is finite and in range,
      actions clip to `ctrlrange`, the aero callback is bound to the right model
      under vectorization, and the energy/mass assertions still hold after any
      generator change.
- [ ] **Regenerate, do not hand-edit.** Every XML change above goes through
      `build_paramotor.py`. `_sanitize_comments()` and the minidom gate in
      `main()` will reject an invalid model — keep it that way.

---

## Suggested order

1. §0 in full, with thrust clamped to 1.0 N and a rigid canopy — gets a training
   loop running against a model you already trust.
2. §2 vectorization — make the loop fast enough to iterate.
3. §1 brakes (model B) — this is the one that decides whether the learned policy
   is about the real vehicle or about a tendon artifact.
4. §3 sim-to-real, once a policy exists worth transferring.
