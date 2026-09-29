# 1 m PEEK paramotor — MuJoCo model notes

Working notes for whoever develops this model and the flight code alongside it.
Source of truth for hardware is `../itemized_spec.md`.

```
build_paramotor.py    generator -> paramotor.xml + scene.xml   (edit THIS, not the XML)
paramotor.xml         the model: geometry, mass, tendons, actuators
paramotor_params.py   aero COEFFICIENTS                        (edit for tuning)
paramotor_aero.py     aero MODEL, eqs. (8)-(18)                (edit for physics)
test_aero.py          38 assertions on the aero layer
view_paramotor.py     interactive viewer, tendons forced visible
```


```bash
# look at it (macOS needs mjpython: the viewer must own the main thread)
./view.sh                      # live, strip aero, 1.5 N, camera tracks the pod
./view.sh --thrust 0.8         # inside the validated envelope
./view.sh --aero lumped        # the paper's single-force model, for comparison
./view.sh --aero off           # no aero: confirms it falls ballistically
./view.sh --sweep              # brakes cycling
./view.sh --build              # regenerate the XML first
```

Anything driving the model itself needs the same three lines the viewer uses:

```python
import mujoco, paramotor_aero, paramotor_params
m = mujoco.MjModel.from_xml_path("paramotor.xml"); d = mujoco.MjData(m)
mujoco.set_mjcb_passive(paramotor_aero.ParamotorAero(m, paramotor_params.PEEK_1M))
```

Without that callback there is **no lift and no drag** and the vehicle simply
falls. The constructor refuses to run if `density`/`viscosity` are non-zero, so
the double-counting failure cannot happen silently.

Aero reaches MuJoCo through `qfrc_passive` (via `mj_applyFT`), *not*
`xfrc_applied`. `xfrc_applied` belongs to the mouse: a callback that clears it
each step eats Ctrl+drag on exactly the bodies you want to grab.

Every BOM item is a **block** with its budgeted mass set explicitly: the model is
a 3-D block diagram of the bill of materials. Compiled total **325.40 g** against
the itemised BOM's 325.40 g (the suite asserts it), 74.6 g under the ceiling.

---

## 5. Propulsion and actuator drive modes

Rotor follows `mujoco_menagerie/skydio_x2`: one site actuator produces thrust and
the propeller drag reaction together via `gear`, declared in one line with the
limits carried by a `<default>` class:

```xml
<motor    class="prop"      name="thrust"          site="propeller" gear="0 0 1 0 0 -0.0105"/>
<velocity class="spin"      name="propeller_speed" joint="prop_spin"/>
<position class="servo_pos" name="servo_pos_L"     joint="drum_L"/>
```

### Actuator count

**4 actuators — `thrust`, `propeller_speed`, `servo_pos_L`, `servo_pos_R`**.
`propeller_speed` is not a second motor; it is the propeller's spin state.

```
--servo-mode position  (default)  thrust + propeller_speed + servo_pos_L/R  -> 4
--servo-mode torque               thrust + propeller_speed + servo_tau_L/R  -> 4
--servo-mode both                 both servo channels per side              -> 6
```

With `both`, drive one channel and leave the other's `ctrl` at 0 — MuJoCo sums
actuators on the same joint. The torque channel exists for the ~200 Hz torque
loop the spec assumes, and applies exactly the commanded N·m.

Two deliberate numbers to revisit:

* `forcerange = ±2.0 N·m` implements the spec's "cap the torque rating at a very
  high amount by default". Datasheet stall at 7.4 V is **0.275 N·m** — set
  `SERVO_TAU_CAP = 0.275` for the real actuator.
* `SERVO_ARMATURE = 5e-5 kg·m²` stands in for gearbox-reflected rotor inertia.
  The bare drum is ~6e-8 kg·m², numerically explosive against any realistic
  torque cap; without it the servo oscillated at the force limit and the capstan
  constraint broke down by 1.7 mm. An **identification parameter, not a datasheet
  value** — get it from a bench step response.

### Propeller gyroscopics — native, via a real spin DOF

Generated *inside the model*, the way the skydio example keeps propeller physics
in the MJCF rather than a Python callback. The propeller carries a real spin
joint and takes its inertia straight from the disk geom:

```xml
<body name="propeller" pos="-0.1160 0 0.010">
  <joint name="prop_spin" type="hinge" axis="1 0 0" limited="false"/>
  <geom name="prop_disk" type="cylinder" size="0.10160 0.0015" mass="0.005" .../>
</body>
```

MuJoCo's own Coriolis terms then produce `M = -(ω_body × H)`. Verified against
theory at **0.00% error**, appearing on the pitch axis from a yaw rate as it must.

| thrust | rpm | H | gyro M at 1 rad/s | prop drag torque |
|---:|---:|---:|---:|---:|
| 1.0 N | 4988 | 0.0135 | 0.0135 N·m | 0.0105 N·m |
| 4.0 N | 9975 | 0.0270 | 0.0270 N·m | 0.0420 N·m |

Same order as the drag reaction, on a vehicle that pitches and yaws under brake.

* Inertia is the **solid disk** `m·R²/2 = 2.58e-5 kg·m²`, taken from the geom.
  At the design speed (~250 rev/s) the two-blade transverse asymmetry averages
  out, so an axisymmetric disk is the right idealisation. About the *spin* axis
  it is 1.5× a uniform two-blade rod (`m·D²/12 = 1.72e-5`), so H and the
  gyroscopic moment are slightly overstated — conservative for control design.
* **Bookkeeping:** drag torque reaches the airframe through the thrust actuator's
  `gear`, *not* the spin joint, so the velocity servo settles at ~0 torque and
  the reaction is not double counted.
* Thrust and rotor speed are one physical thing — `thrust` alone gives force with
  no angular momentum and so no gyroscopic moment. Use
  **the `sync_prop()` helper inlined in the scripts** (or `spin_up()` to skip the ~0.1 s
  spin-up). `k_T` is anchored on the static bench point at an assumed 10 krpm —
  **re-fit it from a thrust stand.**

**Blade centrifugal force is deliberately not modelled.** On a balanced
axisymmetric propeller the root tensions cancel exactly: zero net force and moment on
the airframe. It is a blade-root structural load — hub retention, the spec's
"positive retention compatible with the 3 mm shaft" — not vehicle dynamics.
A 1P imbalance shake force is no longer provided; add it locally if the IMU
isolation question; its unbalance figure is a guess.

---

## 6. Aerodynamics — explicit coefficient model

**The ellipsoid fluid model is gone.** Lift and drag now come from an explicit
coefficient model after Umenberger & Göktoğan 2012 (`pap151.pdf`), eqs. (8)–(18),
applied through `xfrc_applied`:

```
paramotor_aero.py     the model. Pure functions in the paper's FRD frame,
                      plus a ParamotorAero callback that binds to MuJoCo.
paramotor_params.py   PAPER_ACRA2012 and PEEK_1M, identical keys.
test_aero.py          32 assertions.
paramotor_model_alignment.pdf   why, and what was NOT transferred.
```

```python
aero = ParamotorAero(model, paramotor_params.PEEK_1M)
mujoco.set_mjcb_passive(aero)
```

`<option density="0" viscosity="0">` and no `fluidshape` anywhere: MuJoCo's own
fluid model would otherwise be added on top and every force counted twice. The
constructor refuses to run if either is non-zero.

**`CANOPY_FLUIDCOEF` is deleted, not translated.** The ellipsoid primitive has no
C_L0, no C_Lα and no rate derivatives — it is a different *function*, not a
different parameterisation of the same one.

**The paper's aircraft is not a scale model of this one** (1.55 kg / 2.15 m vs
0.325 kg / 1.0 m), so most of its Table 1 does not transfer. Relative density
μ = m/(ρAb) goes 0.507 → 1.355 and Froude 1.691 → 4.516, both 2.67×; Reynolds
falls to 0.41×; flat AR rises 3.99 → 5.10. Identical coefficients therefore do
**not** give identical dynamics: at the same C_lp the non-dimensional roll
damping is 3.5× weaker here. Seven coefficients were carried over, nine must be
identified on the vehicle. Each entry in `paramotor_params.py` is tagged
`[PAPER]`, `[AR]`, `[ROBOT]` or `[PROVISIONAL]`.

**Never copy the paper's inertia tensor.** Its I_xz is 18% of I_xx (a
pilot-carrying trike); this robot's is 0.5% — it is nearly symmetric. MuJoCo
accumulates the real tensor from the BOM and that is better information.

Current behaviour: unpowered glide 5.47 m/s, sink 1.69 m/s, **L/D 3.07** (the
spec assumes 2–3). The pod now has drag at all — 0.340 N at 6 m/s, 11% of
weight, previously zero.

### Strip theory over the panels (default)

`mode="strip"` applies lift and drag to **each of the fourteen panels
separately** instead of as one lumped force at the canopy CoM. Because the
canopy is arched, every panel sees a different local incidence, and three
effects fall out of the geometry with no coefficient to identify:

| | lumped | strip | note |
|---|---:|---:|---|
| C_lp | −0.127 | **−0.221** | 1.74× the damping |
| C_lβ | 0 | **−0.292 /rad** | was produced by *nothing* |
| dM/dβ | 0 | **−1.26 N·m/rad** | 0.71° sideslip balances prop torque |
| total lift | 1.729 N | 1.727 N | preserved, see arch recovery |

**This fixed the powered spiral.** At 0.5 N the lumped model rolls to 63° and
dives; strip holds 0.1° and glides. At 1.0 N it now *climbs* with the propeller
drag reaction left intact — the lumped model needed the torque deleted to
manage that.

The mechanism was a roll *restoring* moment, not more damping. Damping against a
steady torque gives a steady roll **rate**, so bank grows without bound; the
arc's dihedral effect bounds it instead.

**Arch recovery.** An arched canopy lifts on its *projected* area, and strip
theory reproduces that from geometry exactly — measured 0.7979 against the
generator's `PROJ_FRACTION = 0.797`. But the paper's C_L0 is referenced to
**flat** area on a wing that was already arched, so the loss is inside the
coefficient; applying the arch again costs 20% of the lift. `arch_recovery`
(1.2533, computed from the geometry, not hard-coded) removes it from the
coefficient so the geometry can supply it.

`mode="lumped"` keeps the paper's single-force form and is what the rigid
replication of the paper's aircraft must use, since that model has no panels.

### Open: departure above ~1.0 N thrust

Usable thrust range is **0 to about 1.0 N** (T/W 0.31). Above it the vehicle
pitches up, the tension-only suspension goes fully slack (20/20 lines) and the
canopy tumbles, with α reaching ±180° — far outside the ±8/+18° envelope where a
linear no-stall C_L means anything. Established *not* to be a thrust-line offset
(moving the prop from z=0.010 to the CG at z=0.126 does not help) and *not* a
step-input artifact (ramping with the §3.6 motor lag τ=0.45 s does not help).
Needs a stall model and a rigging re-trim. Note the lumped model was already
failing above 0.5 N as a dive, so strip theory did not introduce this — it
widened the usable range. `test_departure_above_one_newton_is_known` pins it.

### Still not modelled

Brake-specific aerodynamics (eqs. 20–22). No additional brake force is applied: the paper's Table 1 gives C_lδa = +0.0021 while its own §3.2
identifies −0.2959 — 140× and a sign flip — and those belong to a d/b = 0.186
brake cascade, where this robot pulls one tendon at one corner of one tip panel.
The brake chain is still purely geometric. Stall is not modelled by the paper
either; α is clamped to [−8°, +18°] and excursions are counted
(`aero.envelope_report()`).

## 7. Closer-to-reality canopy models

The canopy is **rigid**: one free body, seven welded panels. Cheap, stable, gets
the gross pendulum dynamics right. It cannot do camber change under brake, tip
twist, spanwise load redistribution, collapse or cravat — which for a single-skin
PEEK wing is precisely the unknown the spec flags.

**A. Rigid (current).** Brake input rotates the whole wing. Fine for developing
the CAN stack, actuator loop, logging and a first attitude controller. Do not use
it to size the canopy.

**B. Spanwise twist DOFs.** Hinge each panel about its own spanwise axis with
torsional stiffness and damping; keep the arc rigid. Six extra DOFs. Captures
what brakes actually do — local incidence and camber change near the tips — and
gives a real asymmetric-brake yaw mechanism instead of whole-wing rotation. Needs
a torsional stiffness per panel, obtainable from a bench twist test of a PEEK
strip. **Best value for the effort; recommended next step once aero is real.**

**C. Hinged arc chain.** Hinge panels to each other about chordwise axes with
stiffness, so the arch deforms and tips can fold. Captures roll response,
asymmetric loading, the onset of tip collapse. Combine with B. The arc's
equilibrium shape becomes load-dependent, so line lengths must be re-trimmed.

**D. Deformable shell via `<flexcomp>`.** MuJoCo 3.x can simulate a 2-D shell
with the elasticity plugin: real membrane and bending stiffness, wrinkling, lines
attached to vertices. Closest thing to a 125 µm single skin, and unusually here
**the parameters are identifiable** — PEEK's modulus (~3.6 GPa) and the thickness
are both known, so bending stiffness follows from `Et³/12(1−ν²)` rather than
being invented. Cost: far slower, self-collision tuning, and the suspension
attachment scheme must be rebuilt against vertices. This is the right model for
"does this canopy hold its shape at all", which is the scored risk.

**E. Co-simulation.** MuJoCo for pod, lines, actuators, contact; an external
aerodynamic solver over the deformed shape returning forces via `xfrc_applied`.
Highest fidelity, most work, only worth it once a real wing exists to validate
against.

Recommended order: **real aero (§6) → B → D if canopy shape stability becomes the
binding question.**

---

## 8. XML hygiene

Two rules are enforced on the whole emitted document by `_sanitize_comments()`,
so they cannot regress when comments are edited:

1. **No `--` inside a comment**, and no comment body ending in `-`. The XML spec
   forbids both; MuJoCo tolerates them, strict XML viewers and editors reject the
   file outright.
2. **Every comment is collapsed onto one line.**

`main()` then gates on `xml.dom.minidom.parse()` plus an explicit `--` scan, so
an invalid model cannot ship.

---

## 9. Seeing the tendons

The real UHMWPE is under 1 mm and invisible on screen. Render widths are
deliberately fat and are **purely visual — physics is unaffected**:

```python
W_LINE  = 0.0025   # suspension, drawn cyan
W_BRAKE = 0.0035   # brake lines, drawn red
```

The canopy is drawn at 0.55 alpha so lines behind it stay readable.

```
../.venv/bin/python view_paramotor.py           # hanging at 6 m/s trim
../.venv/bin/python view_paramotor.py --sweep   # brakes cycle 0 -> 0.5 rev
```

The viewer forces `mjVIS_TENDON` and `mjVIS_ACTUATOR` on. `--sweep` is the
quickest way to confirm the brake lines actually move.

---

## 10. Known limitations

* The merged `airframe_block` still lumps PCB and structure together, so its
  internal distribution is approximate. The battery — the mass that actually
  mattered — is split back out. See §1.
* Inertia tensors otherwise come from budgeted mass in plausible envelopes, not
  measured parts. Motor and rotor are axisymmetric about the shaft; the rest are
  boxes.
* PCB area is invented; the canopy planform is a constant-chord approximation of
  the spec's "mean chord, not necessarily centre chord".
* Suspension is eight lines to four pod hardpoints; a real cascade has more lines
  and A/B/C rows. Brakes attach at one trailing-edge point per side; a real brake
  cascade spreads over several. Add parallel tendons from the same origin site.
* The brake coupling is bilateral, so brake commands are pull-only. A released
  brake line does not go slack the way a real one does.
* Collision is disabled throughout (`contype=0 conaffinity=0`). No ground, no
  launch, no landing.
* Aerodynamic load on the lines is not modelled. The pod now has drag (§6);
  the lines do not.
* The spec's own caveats stand: battery mass unverified, ESC mass an allowance,
  canopy CL and inflation unestablished, thrust at speed unmeasured.
* Propeller torque is uncompensated in powered flight — see §6.
