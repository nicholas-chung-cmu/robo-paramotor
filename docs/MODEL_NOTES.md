# 1 m PEEK paramotor — MuJoCo model notes

Working notes for whoever develops this model and the flight code alongside it.
Source of truth for hardware is `../itemized_spec.md`.

```
build_paramotor.py    generator -> paramotor.xml + scene.xml   (edit THIS, not the XML)
paramotor.xml         the model: geometry, mass, tendons, actuators
paramotor_params.py   aero COEFFICIENTS                        (edit for tuning)
paramotor_aero.py     aero MODEL, eqs. (8)-(18)                (edit for physics)
test_aero.py          implementation checks and flight acceptance
view_paramotor.py     interactive viewer, tendons forced visible
```


```bash
# look at it (macOS needs mjpython: the viewer must own the main thread)
viewer/view.sh                      # live, strip aero, default 0.8 N, tracks the pod
viewer/view.sh --thrust 0           # unpowered glide baseline
viewer/view.sh --aero lumped        # the paper's single-force model, for comparison
viewer/view.sh --aero off           # no aero: confirms it falls ballistically
viewer/view.sh --sweep              # brakes cycling
viewer/view.sh --build              # regenerate the XML first
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

Aero reaches MuJoCo through `qfrc_passive` (vertex forces straight onto their
slide dofs, pod drag via `mj_applyFT`), *not* `xfrc_applied`. `xfrc_applied` belongs to the mouse: a callback that clears it
each step eats Ctrl+drag on exactly the bodies you want to grab.

Every BOM item is a **block** with its budgeted mass set explicitly: the model is
a 3-D block diagram of the bill of materials. Compiled total **325.40 g** against
the itemised BOM's 325.40 g (the suite asserts it), 74.6 g under the ceiling.

---

## 5. Propulsion and actuator drive modes

The default model has three actuators: `thrust`, `servo_pos_L`, `servo_pos_R`.
Thrust is in newtons (0–2 N); brakes are arm angles (0–3 rad), with a 0.275 N·m
torque cap. Torque/both servo modes remain generator options.

`paramotor_control.sync_prop()` sets thrust and matching rotor speed. The viewer
and MJX resynchronize speed when thrust changes. `prop_spin` is a free hinge,
not a velocity actuator; its disk inertia supplies native gyroscopic coupling.
`T = K_T omega²` assumes 10,000 rpm at 4.02 N and needs identification. This is
direct simplified thrust rather than a motor/ESC or advance-ratio model.

The `propeller` actuator site belongs to the **pod**, at the propeller location,
with `gear="0 0 1 0 0 -0.0105"`. Thrust and drag reaction act on the airframe.
Previously the site belonged to the free rotor: its drag torque drove the rotor
backwards under constant positive thrust. The corrected site preserves geometry
and inertia, without forcing rotor speed each timestep to hide that torque error.

## 6. Aerodynamics — explicit coefficient model

Reduced aerodynamics follow Umenberger & Göktoğan 2012 (`pap151.pdf`, equations
8–18), with coefficients provisionally adapted to this 0.325 kg, 1 m vehicle.
The paper's aircraft is a 1.55 kg, 2.15 m model paramotor, not a pilot-carrying
trike or a geometrically similar version of this robot.

`paramotor_params.py` supplies coefficients, `paramotor_aero.py` is the native
callback, and `paramotor_mjx.py` supplies the same strip model for training.
Native forces use `qfrc_passive` via `mj_applyFT`; MJX supplies equivalent
`xfrc_applied` wrenches. Built-in fluid density and viscosity remain zero.

Default strips apply linear lift and quadratic drag to every cell of the
deformable skin (§7), in each cell's current frame and at its corners' mean
velocity, so camber, twist and brake deflection change local incidence.
Cell forces produce roll moments, so an additional lumped roll-damping term is
suppressed. Explicit pitch/yaw moments remain provisional because cell forces
also contribute those moments. Pod drag is separate. Brakes pull the trailing
edge through servo arms and tendons; there is no brake-specific aerodynamic
coefficient. `mode="lumped"` remains available for comparison.

### References and assumptions corrected in this review

The massless `canopy` root's `xipos` is not the assembly mass center: initial
z = 1.260 m versus the actual panel-assembly z = 0.54136 m. Canopy reference
velocity, lumped force point, and total-moment probes now use
`subtree_com[canopy]`; strip forces remain at panel mass centers. This corrects
force/moment bookkeeping and does not introduce a measured center of pressure.

The existing drag adjustment now uses the updated robot lift slope:
`CDa_robot = CDa_paper - CLa_paper²/(pi e AR_paper) + CLa_robot²/(pi e AR_robot)`.
For assumed e = 0.9 and CLa_robot = 2.08, CDa is approximately 0.945, previously
0.92. This decomposition is provisional, not an identified drag polar.

The geometry-derived arch normalization (1.2533) is retained to preserve the
existing flat-reference lift. Level-flow strip/lumped lift agreement does not
validate transferring the paper's coefficients to this canopy. The paper does
not establish this normalization for the robot. Alpha clipping to −8°..+18° is
an envelope guard, **not a stall model**.

### October 2026 verification and remaining flight failures

MuJoCo/MJX 3.13.0 checks used CPU execution. Mass remains 325.40 g with three
actuators, eight suspension lines, and two brake tendons. Corrected 6 m/s probes
give C_lp = −0.2337 and C_lbeta = +0.3034/rad. Positive beta is rightward FRD
velocity; earlier negative reports used leftward FLU velocity. A static sideslip
derivative does not establish bank stability in the coupled vehicle.

Twelve-second neutral-brake runs start at 6 m/s with matching rotor speed.
Maximum bank and alpha excursions are measured after the initial four seconds;
an excursion means any skin cell exceeds the coefficient envelope.

| Thrust | Maximum bank | Final rotor speed | Steps outside envelope |
|---:|---:|---:|---:|
| 0 N | <0.001° | approximately 0 rad/s | 0% |
| 0.8 N | 26.6° | 467.04 rad/s | 0% |
| 1.0 N | 39.6° | 522.13 rad/s | 0% |
| 1.7 N | 152.9° | 681.77 rad/s | 28.6% |
| 1.8 N | 152.1° | 701.02 rad/s | 18.0% |

Unpowered glide remains approximately 5.76 m/s, sink 1.86 m/s, L/D 2.93.
At 0.8 N the original rotor went from +467 to −3439 rad/s in twelve seconds;
corrected routing preserves positive spin. Powered neutral flight develops a
sustained turn. Left, right, and symmetric 1 rad pulls during seconds 4–8 at
0.8 N remain finite, but neutral turning dominates and some pulls exceed the
alpha envelope. These runs do not establish bidirectional steering or tracking.

All ten existing RL regression tests pass, including native/MJX force,
acceleration, sensor, and short-rollout parity. Aero checks now query panel
velocities independently, test torque routing, and compare rho=0 with disabled
aero using computed energy; the original energy check passed on zeros.

**The old claim of a validated 0–1 N powered flight envelope is withdrawn.**
Powered flight with neutral brakes turns under the propeller reaction torque.
That is accepted, not a defect to trim out: holding a line is the controller's
job. `test_aero.py` therefore REPORTS powered bank and sink as diagnostics
(and only asserts that the run stays finite). `test_aero_reporting.py` verifies
pytest and standalone exit codes using the unpowered-glide check as its probe.

Thrust is clamped at 2.0 N in `build_paramotor.py` (`THRUST_MAX`), so the XML
`ctrlrange` and the RL `thrust_max` agree. It was 1.0 N; the departure above
about 1.7 N in the table above (earlier rigid canopy) was not investigated and
has not been re-checked on the current canopy.

### Run locally on Windows

From PowerShell in the repository, the installed environment can run:

```powershell
.\.venv-rl\Scripts\python.exe -m tests.test_aero
.\.venv-rl\Scripts\python.exe -m pytest tests/test_aero.py tests/test_aero_reporting.py -q
.\.venv-rl\Scripts\python.exe -m pytest tests/test_rl.py -q
.\.venv-rl\Scripts\python.exe -m viewer.view_paramotor --thrust 0
.\.venv-rl\Scripts\python.exe -m viewer.view_paramotor --thrust 0.8
```

The viewer opens a window: Up/Down changes thrust, Left/Right pulls brakes, and
release commands zero brake. Start with the unpowered baseline; powered runs
demonstrate the unresolved turn. No GPU is required for these checks.

For a fresh viewer-only installation with an installed Python 3.12:

```powershell
python -m venv .venv-viewer
.\.venv-viewer\Scripts\python.exe -m pip install mujoco==3.13.0 numpy==2.5.3
.\.venv-viewer\Scripts\python.exe -m viewer.view_paramotor --thrust 0
```

See `docs/RL.md` for the larger training environment. On this Windows host, installing
all requirements hit a long-path error in an Orbax test fixture; installing the
remaining MJX/Flax/Optax wheels supplied the imports needed for the ten checks.
Full training dependencies and GPU execution were not verified in this review.

## 7. Canopy model: deformable shell (option D)

The canopy is a **deformable single skin** (option D, adopted 2026-10-03). The
earlier rigid canopy (fourteen welded panels on one free body) is gone: it could
not bend, twist or camber, so brake response and wing behaviour were wrong in
kind.

**Structure.** A MuJoCo `<flex>` shell (dim 2) over a 16 × 5 grid of point-mass
vertices: spanwise rows at both tips and every strip centre, chordwise stations
at the leading edge, A row, mid chord, B row and trailing edge. Each vertex is a
world-child body with three world-axis slide joints (240 DOFs in all) and an
attachment site; the 68 g skin mass is split over them by tributary area.
Suspension lines attach to the A-row vertex of each line station (C row: the
trailing-edge vertex, inboard), brakes to the trailing-edge tip vertices.

**Material.** Real 125 µm PEEK: bending from plate stiffness
`D = Et³/12(1−ν²)` ≈ 7×10⁻⁴ N·m (E = 3.6 GPa, ν = 0.4) via flex elasticity
(`elastic2d="bend"`). In-plane the film is treated as inextensible: every edge
length is a flex equality constraint (`solref 0.004 1`). At these loads the real
strain is ~1e-4, and a true 4.5×10⁵ N/m membrane spring on sub-gram vertices
would need a ~0.05 ms timestep. No self-collision (MuJoCo Warp has none for
flex), so cravats and folds can pass through the skin.

**Integrator.** Euler (semi-implicit). MuJoCo 3.13 rejects flex elasticity
under `implicit`/`implicitfast` and asks for `discrete`, which MuJoCo Warp does
not support. Only the soft bending is explicit; stretch is a constraint.

**Aerodynamics.** Every grid cell is a strip element: frame from the current
vertex positions (chord and surface normal), velocity from its corners, force
split over its corners (§6). Pitch/yaw pure moments act on the skin as a
zero-net-force couple about its mass centre.

**Cost.** About 11× slower than the rigid canopy on MuJoCo Warp: 4,119 vs 47,081
control steps/s at 4096 environments (RTX 5080).

**Finding: the specified skin does not hold its shape.** In free flight from
6 m/s the skin folds chordwise within 0.2 s (chord 0.196 → ~0.1 m, cell
incidence −75° to +110°) and the vehicle spirals down at ~5 m/s sink. Adding A,
B and C lines at every station does not help: the unsupported leading edge folds
back. The numbers agree: at q ≈ 22 Pa the 38 mm of leading edge ahead of the A
row carries ~0.016 N·m/m against D ≈ 7×10⁻⁴ N·m, a ~4 cm curl radius. A real
single-skin wing holds its nose with battens/rods and its section with many line
attachments. This is exactly the risk this model exists to expose. Two caveats:
the cell aerodynamics are flat-plate strips, not a membrane pressure solution,
so the details of the collapse are approximate, but the order-of-magnitude
argument above does not depend on them.

**Other options considered.** B (spanwise twist hinges on rigid panels) and C
(hinged arc chain) were partial steps toward D and are superseded. E
(co-simulation with an external aerodynamic solver over the deformed shape) is
the higher-fidelity path for the aerodynamics, once a real wing exists to
validate against.

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
../.venv/bin/python -m viewer.view_paramotor           # hanging at 6 m/s trim
../.venv/bin/python -m viewer.view_paramotor --sweep   # brakes cycle 0 -> 0.5 rev
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
* Brake tendons are tension-only length limits and can go slack on release.
  They pull the deformable trailing edge (§7).
* Collision is disabled throughout (`contype=0 conaffinity=0`). No ground, no
  launch, no landing.
* Aerodynamic load on the lines is not modelled. The pod now has drag (§6);
  the lines do not.
* The spec's own caveats stand: battery mass unverified, ESC mass an allowance,
  canopy CL and inflation unestablished, thrust at speed unmeasured.
* Propeller torque is uncompensated in powered flight — see §6.
