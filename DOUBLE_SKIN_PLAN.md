# Double-skin (ram-air) canopy: modeling plan

Status: **plan only, nothing implemented.** Checkpoint before this work:
commit `21ce851`.

## Why

The current canopy is a single 125 µm PEEK skin, modeled as a deformable flex
shell (docs/MODEL_NOTES.md §7). In free flight it folds chordwise within 0.2 s:
the unsupported leading edge curls back under ~22–27 Pa of dynamic pressure,
because the film's bending stiffness (D ≈ 7×10⁻⁴ N·m) is far too low.

Full-size paragliders solve this with **internal pressure**. The wing has an
upper skin and a lower skin joined by ribs. Air enters openings at the leading
edge and pressurizes each cell to roughly the dynamic pressure. The pressure
tensions the fabric, and tension, not bending stiffness, holds the shape.
Internal pressure needs an inside, so modeling it means modeling a
double-surface wing. This is a **design change** from the single-skin spec, not
just a model refinement.

## Geometry

Keep today's planform: 1.0 m flat span, 0.196 m chord, the same arch
(R ≈ 0.44 m, 79.7% projected span).

- **Airfoil.** Give the section a thickness profile. Paraglider sections run
  about 12–17% thick, so 24–33 mm at this chord. The upper and lower skins
  follow the airfoil's upper and lower surfaces about the arch.
- **Mesh.** Two vertex grids on the same 16 spanwise rows (both tips plus the
  14 strip centres):
  - upper skin: about 7 chord stations, leading edge to trailing edge;
  - lower skin: about 5 stations, from the end of the inlet (~10% chord) to the
    trailing edge;
  - the trailing-edge vertices are shared by both skins.
  That is about 16 × 11 ≈ 176 vertices, ~530 degrees of freedom (240 today).
- **Inlets.** Open the lower leading edge from 0 to ~10% chord: no lower-skin
  cells there. The inlet is where pressure comes from (see below); its geometry
  only matters for the pressure coefficient.
- **Ribs.** One rib at each of the 16 rows, joining the upper and lower vertices
  at the same chord station. Model each rib as **tension-only tendons** (one
  vertical per station plus diagonals), with rest lengths from the design
  section. Fabric ribs only carry tension, which is exactly what a limited
  spatial tendon does, and tendons avoid a non-manifold flex mesh (three faces
  meeting on one edge), which MuJoCo's per-edge bending model is not built for.

## Structure and material

- Each skin is its own `<flex dim="2">` over its vertex grid. Both keep the
  current treatment: bending from plate stiffness, stretch as a flex edge
  equality (inextensible film), no self-collision, Euler integrator.
- **Mass.** The BOM allots 65 g to the skin. One 125 µm PEEK skin of 0.196 m²
  weighs only ~32 g (1320 kg/m³), so two skins fit the existing allowance; ribs
  add ~10 g (15 ribs × ~4×10⁻³ m²). Pressure-tensioned skins can also be
  thinner, which would cut this.
- **Lines.** Attach to the lower skin at rib stations (A, B, C rows), as on a
  real wing; the brake lines to the trailing-edge tip vertices as today.

## Internal pressure

MuJoCo has no gas pressure for flex shells, so the pressure goes in the aero
code next to the strip forces (model/paramotor_aero.py, and the same lines in
mjx/paramotor_mjx.py). Both already compute every cell's area and normal.

- **Pressure.** `p_int = Cp_int · ½ρ|V|²`, with V the canopy's mean airspeed
  and `Cp_int ≈ 0.7–1.0` (stagnation at the inlet; real wings measure below 1).
  Clip at zero, so a tail-first or collapsed wing deflates.
- **Force.** Each cell gets `p_int · area · n_out`, outward from the cell
  volume: up on upper-skin cells, down on lower-skin cells, split over the four
  corners like the strip forces.
- **Net effect.** Uniform pressure on a closed volume sums to zero force. It
  adds **no lift**, only tension and shape. The open inlet leaves a small net
  force (`p_int` × inlet area), which is part of what `CD0` already lumps in.
  A test should assert both.
- **Fill dynamics (optional).** Real cells take ~0.1–0.3 s to fill or empty.
  Start instantaneous. Add a first-order lag later if the dynamics need it;
  that needs one extra state value (in `State` for training, an attribute in
  the native callback).

## Aerodynamics

Lift and drag act on the wing as a whole, not once per skin, or lift is counted
twice.

- Compute each strip cell from the **mean surface** (midpoint of the upper and
  lower vertices at each station, the camber line): its frame, area and
  velocity.
- Split each cell's force half to the upper and half to the lower corners.
- Keep `CL0`, `CLa`, `CD0`, `CDa` and the arch recovery as they are; recheck
  the arch factor against the new rest shape.

## Code changes

| File | Change |
| --- | --- |
| `model/build_paramotor.py` | Airfoil profile; upper and lower vertex grids (`cvu_*`, `cvl_*`) sharing trailing-edge vertices; two flex shells; rib tendons; lines to the lower skin; rest lengths from the design shape |
| `model/paramotor_aero.py` | `CanopyMesh` reads both grids; mean-surface strip cells; pressure term; canopy state over all vertices |
| `mjx/paramotor_mjx.py` | Same as the native aero, through the shared geometry functions |
| `rl/rl_env.py` | Nothing structural (it already uses the vertex set via `translate` and `canopy_com`); check the angle-of-attack envelope uses the mean surface |
| `viewer/view_paramotor.py` | Canopy attitude from the mean surface |
| `tests/` | See below |

## Validation, in order

1. **Rest shape.** Built shape is stress-free in bending (interior rows; the
   current single skin shows ~0.06 mN residual). Use uniform spanwise spacing
   at the tips: today's half-width tip cells carry ~10 mN of built-in stress.
2. **Pressure bookkeeping.** Closed-volume pressure force sums to zero;
   inlet force matches `p_int` × inlet area.
3. **Static inflation.** Hold the pod fixed in a 6 m/s flow: the skins
   separate, ribs go taut, chord stays within ~2% of design and thickness near
   the design profile.
4. **Parity.** Native vs Warp, as in `tests/test_rl.py`.
5. **Glide.** `test_unpowered_glide` passes as written (L/D 2–5), without
   loosening it.

## Cost

About 2.2× the vertices plus ~80 rib tendons: expect training roughly 2–3×
slower than the single flex skin, which is already ~11× slower than the old
rigid canopy (4.1k vs 47k control steps/s at 4096 envs). Coarsen the mesh first
if training time matters: fewer chord stations on the lower skin, or ribs
every other row.

## Open decisions

- Single skin or double skin: the double skin changes the vehicle.
  double skin
- Airfoil thickness, and where the inlet sits and how large it is.
  aproximate teh best thickness numerically, same for below
- Skin material and thickness; whether ribs are separate parts.
- `Cp_int`, and whether to model fill lag.
- Number of cells (ribs every row or every other row).
