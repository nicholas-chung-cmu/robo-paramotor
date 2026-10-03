# RL_TODO v2 — phased plan: paramotor model → MJX PPO training loop

Supersedes `RL_TODO.md` (v1). v1 listed *what is missing*; v2 orders it into
phases with exit gates, and folds in the decisions made in the v1 margin notes.
Source of truth for the model is `MODEL_NOTES.md`; hardware is
`../itemized_spec.md`.

**Rule for every phase: do not start phase N+1 until phase N's exit gate passes.**

---

## Decisions already made (from v1 review notes)

| # | Decision | Consequence |
|---|---|---|
| D1 | **Use MJX.** | Aero must be rewritten as pure JAX. `SubprocVecEnv`, the global-callback dispatcher, and the C++/Rust aero rewrite are all **dropped**. |
| D2 | **Brakes stay geometric** (no δ_a coefficients). | `brake_wrench_frd()` stays zero. Brake authority comes only from the tendons deflecting the deformable trailing edge (D4). |
| D3 | **Thrust is hardware-clamped to the validated envelope (0–1.0 N).** | No stall model or re-rig needed before training. The clamp must live in the model (`ctrlrange`), not only in the env. |
| D4 | **Deformable canopy (MODEL_NOTES §7 option D), adopted.** The rigid canopy is rejected. | Flex shell with PEEK bending and inextensible edges, Euler integrator, strip aero per skin cell, on MuJoCo Warp. About 11× slower than rigid. In the current rigging the skin does not hold its shape in flight (§7 finding). |

## Measured baseline (2026-10-01)

| Item | Value |
|---|---|
| Model | `nq=17 nv=15 nu=3 nbody=20 ntendon=10 nsensor=13 (30 values)`, `dt=5e-4`, `implicitfast` |
| `mjx.put_model(paramotor.xml)` | **succeeds**: no unsupported features (tendons, limits, implicitfast, sensors all accepted) |
| MJX vs C MuJoCo, 1 s, no aero, servos stepped to 1 rad | max qpos diff **7.3 mm**. **Root cause found:** when a `servo_pos` actuator saturates at its ±0.275 N·m `forcerange`, C MuJoCo and MJX treat the actuator's `kv` velocity derivative differently under `implicitfast`. Each of these removes the gap (≤1e-5 m): unsaturated command, `forcelimited` off, `kv=0`, or the Euler integrator. See 2.5. |
| MJX single env, Mac CPU | **0.36 ms / physics step**: slower than C MuJoCo. Mac CPU is for correctness only. |
| Python env | `.venv` reports `pyvenv.cfg` 3.12, but packages live in `lib/python3.14` and `python` is 3.14.5. mujoco 3.13.0, mujoco-mjx 3.13.0, jax 0.11.2, jax-metal 0.1.1. **No brax, no mujoco_playground, no warp.** |
| Already fixed since v1 | Servo `forcerange` is now ±0.275 N·m (datasheet stall). |
| Still open from v1 | No `<keyframe>`. Thrust `ctrlrange` is still 0–2.0. `sync_prop()` is still inlined in the viewer. `MODEL_NOTES` §5 still says 4 actuators; the XML has 3. |

---

## Phase 0 — Toolchain and compute (½ day)

Goal: an environment where MJX + Brax PPO run, and a known place to train.

- [ ] **Fresh venv on Python 3.12 or 3.13.** The current venv mixes 3.12 and
      3.14 metadata, and Brax/Playground support for 3.14 is unverified. Pin:
      `mujoco`, `mujoco-mjx` (same version, ≥3.13), `jax`, `brax`,
      `mujoco_playground`, `optax`, `flax`, `orbax-checkpoint`. Commit a
      `requirements-rl.txt` with exact versions.
- [ ] **Pick the training device.** `jax-metal` 0.1.1 is effectively abandoned
      and MJX is not supported on it. Treat the Mac as a **CPU dev box** (unit
      tests, parity tests, short smoke runs). Training goes on an **NVIDIA GPU**
      (CMU cluster / PSC Bridges-2 / cloud) with `jax[cuda12]`. Optionally
      evaluate the MuJoCo Warp backend (`mjx.put_model(..., impl="warp")`), which
      is NVIDIA-only.
- [ ] **Smoke test on the GPU box:** `put_model`, `jit(vmap(mjx.step))` over
      1024 envs, report steps/s. This is the throughput number every later
      decision gets checked against.
- [ ] Put the repo under the GPU box's path with the same requirements file.

**Exit gate:** `pytest test_aero.py` passes in the new venv on the Mac, and a
vmapped `mjx.step` runs on the GPU with a recorded steps/s figure.

---

## Phase 1 — Specification (no env code yet)

Goal: everything the env must implement, written down with numbers, **before**
writing it. This is the gate the v1 notes asked for: sensor rates, noise and
bias, domain randomization, and infrastructure must all be pinned here.
Deliverable: `rl_spec.py` (one dict per section, imported by the env and tests)
plus this file updated.

### 1.1 Task, action, reward, termination

- [ ] **First task: heading hold with altitude hold.** Random heading
      setpoint steps (±90°) every 5–10 s. Brakes steer, thrust holds altitude.
      Later tasks (waypoint capture, then path following) reuse the env with a
      different command generator.
- [ ] **Control rate 50 Hz.** At `dt=5e-4` that is `n_substeps=40`; at
      `dt=1e-3`, 20 (see 2.6). Episode length 30 s = 1500 policy steps.
- [ ] **Action, `nu=3`, normalized to [-1, 1]:**

  | ch | actuator | physical range | mapping |
  |---|---|---|---|
  | 0 | `thrust` | 0 – 1.0 N (D3) | affine, then first-order lag (1.3) |
  | 1 | `servo_pos_L` | 0 – 3.0 rad | affine; rate-limited at no-load speed (1.3) |
  | 2 | `servo_pos_R` | 0 – 3.0 rad | same |

- [ ] **Reward** (all terms logged separately in `metrics`):
      `r = w_h·exp(-(Δψ/σ_ψ)²) + w_z·exp(-(Δz/σ_z)²) − w_u·‖a_t − a_{t−1}‖² − w_T·thrust − w_α·1[α clamped] + r_alive`.
      Starting weights: `w_h=1.0, σ_ψ=0.35 rad, w_z=0.5, σ_z=2 m, w_u=0.05, w_T=0.05, w_α=0.5, r_alive=0.1`.
- [ ] **Termination:**
      - canopy-up axis · world-up < cos 60° (tumble), or canopy CoM below pod CoM
      - raw panel α outside [−30°, +45°] for more than 0.2 s (far outside the
        −8/+18° model envelope)
      - altitude loss > 15 m below the episode start (ground plane is at
        −18 m in `scene.xml`; collision is disabled, so there is no contact to
        detect)
      - any NaN/Inf in `qpos`/`qvel` (reset that env, count it in metrics)
      - time limit 30 s (truncation, not termination, for value bootstrapping)

### 1.2 Sensor model — rates, noise, bias, latency

All sensors in the MJCF read ground truth (MuJoCo 3.13 ignores `noise`). The
env applies a corruption layer per sensor. Values below are datasheet typicals
where stated: **verify against the delivered parts and replace with bench
measurements.** Sample-and-hold to the control rate: a sensor slower than
50 Hz repeats its last value.

| Sensor (part) | Quantity | Rate in sim | White noise (σ per sample) | Bias / drift | Latency |
|---|---|---|---|---|---|
| LSM6DSOX gyro | ω body, rad/s | 50 Hz to policy (decimated from ≥500 Hz in HW, spec §MCU) | noise density 3.8 mdps/√Hz (datasheet, HP mode). At 50 Hz with on-chip LPF use σ ≈ 0.003 rad/s; randomize ×[0.5, 3] for vibration | turn-on bias U(±1 °/s) per episode (zero-rate level ±1 dps typ.), random walk σ 1e-4 rad/s/√s | 0–1 ctrl step |
| LSM6DSOX accel | specific force, m/s² | 50 Hz | 70 µg/√Hz (datasheet) → σ ≈ 0.01 m/s²; ×[1, 5] for prop vibration | turn-on bias U(±20 mg) per axis (zero-g offset typ.) | 0–1 step |
| BMP581 baro | altitude, m | 50 Hz | σ 0.1 m after on-chip OSR/IIR (≈1 Pa) | offset U(±3 m) per episode (±30 Pa absolute accuracy) + drift 0.02 m/s random walk; prop-wash/dynamic-pressure error k·V², k ∈ U(0, 0.02) | 1–2 steps |
| GM10 Pro V3 (u-blox M10 class) GNSS | position NED, m; velocity NED, m/s | **10 Hz** (spec: "up to 10 Hz") | pos σ 1.5 m horiz / 3 m vert, as a slow Gauss–Markov process (τ ≈ 30 s), not white; vel σ 0.05–0.1 m/s | — | **100 ms** (5 steps); randomize 60–150 ms |
| QMC5883L compass | heading, rad | 50 Hz (sensor ODR up to 200 Hz) | σ 1° | hard-iron heading offset U(±5°) per episode (motor and power wiring nearby, spec §GNSS) | 0–1 step |
| BD10BL-CAN servo telemetry | arm angle, rad | 50 Hz | σ 0.005 rad | — | 1 step (CAN) |
| AM32 ESC telemetry | rpm | 10 Hz | σ 1 % | — | 1–2 steps |

- [ ] **Attitude source.** Decide: (a) policy gets raw gyro/accel/mag and must
      learn to estimate, or (b) env emulates the onboard estimator's output
      (attitude with error σ 1–2° roll/pitch, 3–5° yaw, slow-correlated).
      **Recommend (b)** for the first runs: it matches the deployed
      architecture (estimator 100–200 Hz, spec §MCU) and keeps the policy small.
      Represent attitude as 6-D rotation (first two columns of R) or gravity
      vector + heading, never a raw quaternion.
- [ ] **Observation vector** (target ≈ 25–35 dims, all normalized):
      attitude (6), gyro (3), baro altitude error to setpoint (1),
      baro-derived vertical speed (1), GNSS ground velocity in heading frame (2,
      held/delayed), heading error (sin, cos) (2), servo arm angles (2),
      prop rpm (1), last action (3), plus a short history (stack last 3 gyro and
      action samples). **No ground-truth `framepos`/`framelinvel`.** Those stay in
      `info` for the critic (asymmetric actor–critic, 5.2).

### 1.3 Actuator dynamics (were v1 §1; must be in the spec, not the reward)

| Item | Value | Source |
|---|---|---|
| Motor/prop thrust lag | first-order, τ = 0.45 s nominal, DR [0.3, 0.6] | MODEL_NOTES §6 (spec §3.6 figure) |
| Thrust ceiling | 1.0 N (D3) | MODEL_NOTES §6 envelope |
| Servo no-load speed | 0.09 s/60° = 11.6 rad/s @7.4 V; 0.12 s/60° = 8.7 rad/s @6 V | BD10BL-CAN datasheet (spec) |
| Servo stall torque | 0.275 N·m @7.4 V; 0.216 N·m @6 V | same |
| Servo command → motion latency | 1 control step (CAN) | assumption, measure |
| Prop spin | `ω = √(T/K_T)`, written to `qvel[prop_spin]` every substep | `sync_prop()` logic, now in JAX |

- [ ] Implement the thrust lag and the servo rate limit **in the env state**
      (a JAX filter on the commanded value), not via `dyntype="filter"`, so
      τ can be domain-randomized per env without rebuilding the model.

### 1.4 Domain randomization

Randomized per episode on reset. **Bold** = `[PROVISIONAL]` in
`paramotor_params.py`, so these get the widest ranges. Aero params are plain
JAX inputs (free). Mass and inertia changes require per-env `mjx.Model` fields
(`body_mass`, `body_inertia`, `body_ipos`) vmapped with `in_axes`, as Playground
does it.

| Parameter | Nominal | Range | Why |
|---|---|---|---|
| **CD0** | 0.15 | U[0.10, 0.22] | Re ≈ 0.87e5, laminar-separation dominated |
| CL0 | 0.40 | ×U[0.85, 1.15] | single-skin PEEK CL "not established" (spec) |
| CLa | 2.08 | ×U[0.85, 1.15] | AR-corrected, not measured |
| CDa | 0.92 | ×U[0.8, 1.2] | |
| **Cm0** | 0.018 | U[−0.01, 0.05] | rigging-dependent |
| **Cma** | −0.2 | U[−0.4, −0.1] | |
| **Cnr** | −0.0035 | U[−0.04, 0.0] | strip theory gives −0.038 natively; paper −0.0035 |
| Cmq | −2.0 | ×U[0.7, 1.3] | |
| CD0F·AF | 0.0154 m² | ×U[0.7, 1.5] | bluff box estimate |
| ρ | 1.225 | U[1.10, 1.25] | temperature/altitude |
| All-up mass | 325.4 g | battery ±10 g, airframe block ±15 g | battery mass unverified (spec), allowances |
| Battery x-position | 0.010 m | ±0.015 m | CG trim rail (spec §Mechanical) |
| `K_T` | anchored on 10 krpm guess | ×U[0.8, 1.2] | re-fit on thrust stand |
| `KM_KT` (drag-torque ratio) | 0.0105 m | ×U[0.8, 1.2] | |
| Thrust lag τ | 0.45 s | U[0.3, 0.6] | |
| Servo speed / torque cap | 7.4 V values | interpolate 6.0–8.4 V battery | battery sag |
| `SERVO_ARMATURE` | 5e-5 | log-U[2e-5, 1e-4] | identification parameter |
| Servo `kp`, `kv` | 2.0, 0.02 | ×U[0.8, 1.2] | |
| Wind (mean) | 0 | speed U[0, 3] m/s, random azimuth | |
| Gusts | 0 | OU process per axis, σ U[0, 0.8] m/s, τ 2 s | |
| Sensor noise/bias/latency | 1.2 table | as listed | |
| Initial state | trim keyframe | heading U(±π), roll/pitch ±5°, airspeed ±0.5 m/s, thrust U[0.3, 0.9] N | |

- [ ] Curriculum: Phase 5 starts with DR **off**, then ramps the ranges
      linearly to 100 % over the first ~30 % of training.
- [ ] Keep a fixed **nominal** eval set (DR off, no wind) and a fixed
      **stress** eval set (DR at range edges, 3 m/s wind) for every checkpoint.

### 1.5 Infrastructure spec

- [ ] **File layout**
      ```
      paramotor_aero_jax.py   JAX port of paramotor_aero.py (pure functions)
      paramotor_env.py        MjxEnv: reset/step/obs/reward, sensor + actuator layers
      rl_spec.py              every number in Phase 1, one place
      train_ppo.py            Brax PPO entry point, config via CLI/yaml
      eval_policy.py          rollouts on C MuJoCo + numpy aero, plots, video
      test_aero_jax.py        JAX-vs-numpy aero parity
      test_env.py             env invariants
      ```
- [ ] **Seeding:** one root `jax.random.PRNGKey(seed)`, split into
      `{reset, dr, sensor, command, policy}` streams per env. A fixed seed must
      reproduce an episode bit-for-bit on the same device.
- [ ] **Logging:** TensorBoard or W&B. Per eval: each reward term, episode
      length, termination reason histogram, α-clamp fraction, servo torque
      saturation fraction, thrust-at-ceiling fraction, NaN-reset count.
- [ ] **Checkpoints:** orbax every N updates, storing params, normalizer
      stats, `rl_spec` snapshot and git hash.
- [ ] **Regenerate, do not hand-edit.** Every XML change goes through
      `build_paramotor.py`, and its minidom gate stays.

**Exit gate:** `rl_spec.py` exists and covers every table above. Every sensor
number is either cited to a datasheet or marked `# ASSUMED, measure`. Reviewed
by the team.

---

## Phase 2 — MJX-native physics

Goal: one physics substep that is pure JAX and matches the C MuJoCo + numpy aero
reference.

- [ ] **2.1 Generator changes** (`build_paramotor.py`, then regenerate)
      - `THRUST_MAX = 1.0` → thrust `ctrlrange="0 1.0"` (D3).
      - Add `<keyframe name="trim">` holding the settled glide state: settle
        about 10 s at a nominal thrust using the existing CPU path, write
        qpos/qvel/ctrl, with `prop_spin` velocity consistent with thrust.
      - Fix the stale "applied through xfrc_applied" comment in `<option>` and
        `MODEL_NOTES` §5's "4 actuators" text.
- [ ] **2.2 Port aero to JAX** (`paramotor_aero_jax.py`)
      - Host-side, once: panel ids, panel areas, `arch_recovery`, body ids,
        computed by the existing numpy `ParamotorAero.__init__` and passed in as
        constants. The Python loops in `__init__` never need to be jitted.
      - Per substep: pure function `aero_xfrc(mx, dx, params, wind_w) -> (xfrc (nbody,6), metrics)`.
      - **Replace `mj_objectVelocity`** with freejoint state: for a free body,
        `qvel[0:3]` is the world linear velocity of the body origin and
        `qvel[3:6]` is the angular velocity in the body frame. So
        `ω_w = R·ω_local` and `v_com = v_origin + ω_w × (xipos − xpos)`. Exact,
        no Jacobians.
      - **Relative airspeed:** `v_air = v_body − v_wind`. Wind enters here from
        day one, not retrofitted (it was missing in v1).
      - **Replace `mj_applyFT` → `qfrc_passive`** with `xfrc_applied`, which
        acts at the body CoM (already verified by `test_xfrc_acts_at_com`).
        The env has no mouse, so the reason v1 avoided `xfrc_applied` does not
        apply. Pure moments go in `xfrc_applied[canopy, 3:6]`.
      - Branches (`if V < eps`, `np.allclose`) become `jnp.where` with a safe
        norm. In-place writes become `.at[].set()`. The `self.last` dict
        becomes a returned metrics pytree.
- [ ] **2.3 Substep function**
      ```
      substep(mx, dx, env_state):
          xfrc, m = aero_xfrc(mx, dx, aero_params, wind(t))
          dx = dx.replace(xfrc_applied=xfrc,
                          ctrl=actuator_layer(env_state),        # lag + rate limit
                          qvel=dx.qvel.at[prop_dof].set(sqrt(T/K_T)))
          return mjx.step(mx, dx), m
      ```
      Run under `jax.lax.scan` for `n_substeps`. Note the one-substep lag:
      aero is computed from the kinematics of the state *entering* the step,
      the same as the C callback. At 2 kHz this does not matter, but it
      matters if 2.6 raises `dt`.
- [ ] **2.4 Aero parity tests** (`test_aero_jax.py`): for random states
      (including V≈0, α beyond the clamp, sideslip, roll rates), JAX forces
      and moments equal numpy `ParamotorAero.wrench` to 1e-6 relative in
      float64. Port the physics assertions from `test_aero.py` that do not
      depend on the callback: static lift, trim airspeed, glide L/D ≈ 3.07,
      strip roll derivatives C_lp ≈ −0.221 and C_lβ ≈ −0.292, and the
      no-spiral-at-0.5-N check.
- [ ] **2.5 Trajectory parity**, MJX + JAX aero vs C MuJoCo + numpy aero: a
      10 s glide, a 0.8 N powered climb, and a brake sweep. Fix the known
      mismatch first: move the servo's `kv=0.02` from the `<position>`
      actuator onto the `arm_*` joint `damping` in the generator (set `kv=0`),
      so the damping is not tied to a force-clamped actuator. With `kv=0` the
      two simulators agree to 1.6e-9 m over 1 s. Then re-check that the servo
      step response still matches.
      **Gate:** trim airspeed, sink rate and bank angle within 2 % / 0.5°, and
      no qualitative divergence (spiral, tumble) in either.
- [ ] **2.6 Timestep study.** `n_substeps` is the dominant training cost. Try
      `dt ∈ {5e-4, 1e-3, 2e-3}` in MJX with the tendon limits and servo
      armature. Accept the largest dt that keeps 2.5 parity and keeps the
      brake-line tendon violation under 1 mm. Each doubling halves wall-clock
      time.
- [ ] **2.7 float32.** Training runs in float32 on GPU. Re-run 2.5 in float32
      and confirm no drift blow-up, particularly the `prop_spin` angle, which
      grows without bound. Wrap it, or drop its qpos from anything observed.

**Exit gate:** 2.4 and 2.5 pass in float32 at the chosen dt. The chosen dt and
the measured GPU steps/s per substep are recorded here.

---

## Phase 3 — The environment

Goal: `paramotor_env.py`, a Playground-style `MjxEnv` (`reset(rng)`,
`step(state, action)`, `observation_size`, `action_size`) that is fully
jittable and vmappable.

- [ ] `reset`: keyframe → DR sample (1.4) → initial-state perturbation →
      command sample → sensor bias draws → zeroed filter and delay buffers.
- [ ] `step`: action → denormalize → actuator layer → `scan(substep)` →
      sensor layer (multi-rate sample-and-hold via step counters, delay via
      ring buffers in `env_state`) → obs, reward, done, metrics.
- [ ] Command generator: heading setpoint steps, altitude setpoint = start
      altitude ± U(5 m).
- [ ] Auto-reset on done, and NaN guard (reset plus a metric, never propagate NaN).
- [ ] Privileged critic obs: true state, true wind, DR params.
- [ ] **`test_env.py`:**
      - reset is deterministic per key
      - obs is finite and within normalization bounds across 1000 random
        rollouts
      - actions saturate correctly at `ctrlrange`
      - zero-brake, nominal-thrust rollout stays near trim for 30 s
      - `vmap` over 64 envs with different DR gives different trajectories
      - a fixed-seed episode reproduces exactly
      - mass assertion (325.40 g nominal) still holds after regeneration
- [ ] A CPU-only **render helper** that replays a logged `qpos` trajectory
      through `mujoco.Renderer` with `scene.xml`, for videos.

**Exit gate:** `test_env.py` green, and `jit(vmap(step))` at 1024 envs on GPU
with a measured env-steps/s.

---

## Phase 4 — Baseline and budget

- [ ] **Scripted baseline in JAX:** a PID on heading error through
      differential brake, plus a PI on altitude through thrust. It proves the
      brake channel has usable authority under D2 (if PID cannot turn the
      vehicle, PPO will not either, so stop and revisit D2), and it is the
      score PPO must beat.
- [ ] Evaluate the baseline on the nominal and stress eval sets. Record the
      reward, heading RMS and altitude RMS.
- [ ] **Throughput budget.** From the Phase 3 steps/s, compute wall time for
      100M env steps. If it exceeds ~6 h on one GPU, revisit dt (2.6) or
      `num_envs` before training.

**Exit gate:** baseline numbers recorded, and the budget fits.

---

## Phase 5 — PPO training

- [ ] **5.1 Brax PPO starting config** (Playground defaults for
      locomotion-scale tasks, then tune): `num_envs=2048–4096`,
      `unroll_length=20`, `batch_size=256`, `num_minibatches=32`,
      `num_updates_per_batch=4`, `discounting=0.99`, `learning_rate=3e-4`,
      `entropy_cost=1e-3`, `reward_scaling=1`, `normalize_observations=True`,
      `action_repeat=1` (the substeps are already inside `step`), policy MLP
      (128,128,128), value MLP (256,256,256).
- [ ] **5.2 Asymmetric actor–critic:** actor sees the sensor obs (1.2), critic
      also sees the privileged obs.
- [ ] **5.3 Run ladder.** Each rung must beat the baseline before moving on.
      1. No DR, no wind, no sensor noise: does it learn at all?
      2. + sensor noise, delay and multi-rate.
      3. + actuator lag and DR curriculum.
      4. + wind and gusts.
      5. Full DR, 3 seeds, for the reported result.
- [ ] **5.4 Watch for sim exploits:** α-clamp fraction rising during
      training, thrust pinned at the 1.0 N ceiling, brake arms parked at range
      limits, or bang-bang servo chatter. Any of these gets a reward or physics
      fix, not more training.

**Exit gate:** full-DR policy beats the PID baseline on both eval sets across
3 seeds, and the α-clamp fraction is below 1 %.

---

## Phase 6 — Validation and the path to hardware

- [ ] **Cross-simulator check:** run the trained policy in C MuJoCo with the
      original numpy `ParamotorAero` callback (`eval_policy.py`). Any large gap
      is an MJX/JAX-port artifact the policy found.
- [ ] Robustness sweeps: success vs wind speed, vs CD0, vs mass, vs latency.
      Plot each.
- [ ] Videos of nominal and stress episodes.
- [ ] **Deployment path** (decide, do not build yet): policy MLP to C (hand
      written or ONNX → ST Edge AI) on the STM32N6 Cortex-M55 at 50 Hz in the
      low-priority control task; the obs pipeline must match `rl_spec.py`
      exactly; and a fallback to the PID baseline on inference deadline miss or
      estimator fault.
- [ ] List every `[PROVISIONAL]` parameter that the robustness sweeps show the
      policy is sensitive to. That list is the identification priority for
      flight tests.

---

## Not in this plan (explicitly deferred)

- Brake aerodynamic coefficients / spanwise-twist canopy (MODEL_NOTES §7 B) — D2.
- Stall model and rigging re-trim for thrust above 1.0 N — D3.
- Flexible canopy (`flexcomp`), ground contact, launch and landing.
- C++/Rust aero rewrite — superseded by D1.
