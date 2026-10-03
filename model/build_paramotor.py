#!/usr/bin/env python3
"""
Generate a MuJoCo MJCF model of the 1 m PEEK paramotor described in itemized_spec.md.

Every BOM item is drawn as a BLOCK (box geom) with its budgeted mass set explicitly,
so the model is literally a 3-D block diagram of the bill of materials.

Consolidations applied (per request):
  * All avionics are merged into ONE custom PCB block with the same total mass
    (ESC + controller + IMU + baro + GNSS + RX + regulators + INA260 + logging).
  * Fasteners / retention / harness are folded into the FRAME block.
  * PCB area is a GUESS (see PCB_L x PCB_W below) -- nothing in the spec fixes it.

Usage:
    python -m model.build_paramotor        # write paramotor.xml
    python -m model.build_paramotor --calibrate  # compile, measure tendon lengths, rewrite
"""
import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np

from model.paramotor_aero import canopy_topology, triangle_areas

HERE = Path(__file__).parent

# ----------------------------------------------------------------------------
# MASS BUDGET  (grams) -- traceable to the BOM table in itemized_spec.md
# ----------------------------------------------------------------------------
G = 1e-3  # gram -> kg

# Spanwise vertex rows of the double-skin canopy, evenly spaced along the arch,
# both tips included. Every row carries a rib; the line stations and brake
# anchors are rows.
N_SPAN = 16

M = {
    # --- one merged avionics PCB: 10+30+1.7+3+7.3+2.2+7+4+3 = 68.2 g ---
    # --- merged airframe block: airframe 45 + harness 12 + fasteners 8
    #     + PCB 68.2 + pod riser hardware 3 + steering brackets/rigging 4 ---
    "airframe":  140.2,
    # Battery stays SEPARATE: it is the single largest discrete mass and it sits
    # 30 mm below the rest, so merging it visibly falsified the pod CG.
    "battery":    65.0,
    "motor":      19.6,
    "prop":        5.0,
    "servo":      11.8,   # each, x2  = 23.6
    "arm":         2.0,   # servo arm/horn, each, x2
    # Canopy: the 65 g skin allowance (both skins and the ribs, see
    # FABRIC_AREAL below) PLUS the 3 g canopy half of the 6 g line allowance,
    # spread over the vertices by tributary fabric area (vertex_masses()).
    # Keeping the line mass here keeps the 325.40 g BOM total, which the suite
    # asserts.
    "fabric":     65.0,
    "line_half":   3.0,
}

BOM_TOTAL_G = (
    M["airframe"] + M["battery"] + M["motor"] + M["prop"]
    + 2 * M["servo"] + 2 * M["arm"] + M["fabric"] + M["line_half"]
)

# ----------------------------------------------------------------------------
# CANOPY GEOMETRY  (from itemized_spec.md)
# ----------------------------------------------------------------------------
SPAN_FLAT = 1.000        # m, flat laid-out span
# Flat aspect ratio. The spec's original 4.0 is roughly a real wing's PROJECTED
# AR, not its flat AR, so the wing was much stubbier than a real paramotor.
# Source: Ozone Spyder 3 paramotor wing, flat AR 5.1 across all six sizes
# (https://flyozone.com/paramotor/products/gliders/spyder-3). Cross-checks:
# Ozone Roadster 4 flat AR 5.1; BGD Dual Motor (tandem) flat AR 5.3.
AR_FLAT = 5.1
S_FLAT = SPAN_FLAT ** 2 / AR_FLAT        # 0.196 m^2
CHORD = S_FLAT / SPAN_FLAT               # 0.196 m constant chord
# Projected span / flat span, which is what the arc solve below actually uses.
# Source: Spyder 3, 7.98 / 10.01 = 0.797 (identical ratio at every size).
# NOTE: with a constant chord, projected AREA / flat area equals this same
# number. A real tapered wing keeps more area (Spyder 3: 17.2/20 = 0.86)
# because the tips are less arched and narrower than a constant-chord panel.
PROJ_FRACTION = 0.797

# Solve the arch: arc length = SPAN_FLAT, chord-of-arc = PROJ_FRACTION * SPAN_FLAT
#   2R sin(phi) = 0.8 * b   and   2R phi = b     =>   sin(phi)/phi = 0.8
def _solve_half_angle(ratio: float) -> float:
    lo, hi = 1e-6, math.pi - 1e-6
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if math.sin(mid) / mid > ratio:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

PHI_HALF = _solve_half_angle(PROJ_FRACTION)   # ~1.1311 rad
THETA = 2 * PHI_HALF                          # total arc angle
R_ARC = SPAN_FLAT / THETA                     # ~0.4421 m radius of curvature

TE_X = -CHORD / 2                             # trailing edge, where the brakes pull

# ----------------------------------------------------------------------------
# POD / HARDWARE GEOMETRY  -- box half-sizes (m)
# ----------------------------------------------------------------------------
# *** PCB AREA IS A GUESS: 80 mm x 60 mm, 12 mm installed envelope ***
PCB_L, PCB_W, PCB_H = 0.080, 0.060, 0.012
BAT_L, BAT_W, BAT_H = 0.056, 0.030, 0.016     # Gens ace 850 mAh 2S, EC2
SERVO_L, SERVO_W, SERVO_H = 0.029, 0.012, 0.030   # Hitec BD10BL-CAN envelope
# Servos lie FLAT: the 29 mm length and the 30 mm case height are both horizontal,
SERVO_BOX = (SERVO_L / 2, SERVO_H / 2, SERVO_W / 2)
SERVO_X = -0.020
SERVO_Y = 0.040
SERVO_Z = 0.052

# CANOPY_FLUIDCOEF is deleted, not translated. MuJoCo's ellipsoid fluid model is a
# different FUNCTION from eqs. (12)-(18), not a different parameterisation of the
# same one: it has no C_L0, no C_Lalpha and no rate derivatives. Aero now lives in
# paramotor_aero.py; the coefficients are in paramotor_params.py.

# "FAKE HOLLOW" CANOPY. A single 125 um PEEK skin folds chordwise in free
# flight (docs/MODEL_NOTES.md section 7): its bending stiffness is far too low.
# A real paraglider is a double skin held in shape by internal pressure. This
# model does not simulate the air. It meshes the SHAPE the inflated wing takes
# as a solid, gives it the stiffness the double skin would have, and gives its
# vertices the mass of the hollow skins and ribs (vertex_masses()).
#
# SECTION. Symmetric NACA 4-digit thickness about the old single skin, which
# becomes the mean (camber) surface: the aerodynamic reference, its area and its
# arch are unchanged, so every coefficient carries over. Upper and lower
# surfaces share the leading- and trailing-edge vertices.
THICKNESS = 0.15              # t/c; paraglider sections run about 12-17%
# Chord stations, fractions of chord from the leading edge: the old skin's, so
# the A and B line rows stay where they were.
A_FRAC = 0.5 - 0.060 / CHORD  # 0.194
B_FRAC = 0.5 + 0.060 / CHORD  # 0.806


def chord_fractions():
    return (0.0, A_FRAC, 0.5, B_FRAC, 1.0)


N_CHORDWISE = 5
TE_INDEX = N_CHORDWISE - 1
ROW_INDEX = {"A": 1, "B": 3, "C": TE_INDEX}


def half_thickness(f: float) -> float:
    """NACA 4-digit half-thickness at chord fraction f (closed trailing edge)."""
    return 5 * THICKNESS * CHORD * (0.2969 * math.sqrt(f) - 0.1260 * f
                                    - 0.3516 * f**2 + 0.2843 * f**3 - 0.1036 * f**4)


# MASS. PEEK film, 1320 kg/m^3. A real double skin's upper and lower skins and
# one rib per span row share the 65 g fabric allowance, so the film is whatever
# thickness fits it (about 120 um). Each vertex carries the mass of the fabric
# around it, so mass, mass centre and inertia are those of the HOLLOW wing.
PEEK_E, PEEK_RHO = 3.6e9, 1320.0
#
# STIFFNESS. In a pressurized double skin the two skins are the faces of a
# sandwich: tension, not the film's own plate stiffness, carries bending, and
# EI per unit width = E t h^2 / 2 for faces of thickness t a depth h apart
# (sandwich_ei()). For this film that is ~10^2 N.m at the thickest section:
# effectively rigid against air loads of a few newtons. The model gets the same
# mechanism from geometry. The solid between the leading edge and the last
# lower-surface station is a tetrahedral flex whose every edge length is an
# equality constraint. Edges along the upper and lower surfaces act as the
# inextensible faces, so bending stiffness scales with depth squared, as in the
# sandwich. Constraints are solved implicitly, so this stays stable at the
# 0.5 ms timestep, where a 3-D elastic flex at the equivalent modulus
# (E_eff = 6 E t / h ~ 1e8 Pa) would need about 0.03 ms. FLEX_SOLREF sets how
# closely "rigid" is met; the residual compliance is reported by main().
FLEX_SOLREF = "0.004 1"       # edge-length constraint: stiff, critically damped
#
# BRAKE FLAP. The bay from the last lower-surface station to the trailing edge
# is NOT solid. Its upper surface is an inextensible membrane (a 2-D flex
# without bending), and its lower surface a tension-only tendon per row. A real
# double skin behaves the same way there. Pulling the trailing edge down
# wrinkles the lower skin (the tendon goes slack) and the flap deflects, while
# lift cannot push it up past the design section (the tendon goes taut). A solid
# trailing edge would leave the brakes nothing to deflect.


def fabric_thickness() -> float:
    """PEEK film thickness that puts both skins and every rib in 65 g."""
    return M["fabric"] * G / (PEEK_RHO * fabric_area())


def sandwich_ei() -> float:
    """Bending stiffness per unit width of the double skin at its thickest
    section, E t h^2 / 2 (N.m)."""
    h = 2 * max(half_thickness(f / 100) for f in range(101))
    return PEEK_E * fabric_thickness() * h * h / 2


# Scenery. Purely visual: contype/conaffinity are 0, so there is no ground
# contact and the dynamics are identical with or without it.
GROUND_Z = -18.0
TERRAIN_HALF = 90.0           # m, half-extent of the heightfield
TERRAIN_H = 12.0              # m, height of the tallest peak (tops out at -6 m)

# Tendon render widths. Purely visual -- physics is unaffected. The real UHMWPE
# is well under 1 mm and invisible on screen, so these are deliberately fat.
W_LINE = 0.0025      # suspension lines, drawn cyan
W_BRAKE = 0.0035     # brake lines, drawn red
MOTOR_D, MOTOR_H = 0.028, 0.026               # SunnySky X2204
PROP_D = 0.2032                               # GWS EP8043, 8 in
FRAME_L, FRAME_W, FRAME_H = 0.140, 0.090, 0.110
MOTOR_X = -0.098                              # pusher: motor aft
PROP_X = MOTOR_X - 0.018

# Height of the canopy above the pod, i.e. the riser/line height. A real paraglider's suspension
# lines are "typically 4-5 m long, with the end attached to 2-4 further lines of
# around 2 m", plus ~0.4 m risers (https://en.wikipedia.org/wiki/Paragliding),
# so ~6.5-7 m of line height under a wing of ~10-11 m flat span: about 0.63 of
# the flat span. Scaled to this 1 m wing that is 0.63 m.
LINE_HEIGHT_FRAC = 0.63
Z_CANOPY = LINE_HEIGHT_FRAC * SPAN_FLAT       # canopy mid-surface above pod origin
V_DESIGN = 6.0                                # m/s, spec illustrative airspeed 5.74-6.33
# Brake actuation is a SERVO ARM (horn), not a capstan. The tendon is anchored
# at the arm tip; swinging the arm moves that anchor and pays the line in or out
# purely geometrically. No constraint, and the line stays tension-only.
ARM_LEN = 0.040                               # m, horn length, pivot to tip
# The arm's zero points AT the trailing-edge anchor, so line length is
#   sqrt(D^2 + r^2 - 2*D*r*cos(theta)),
# which is strictly increasing on [0, pi]: no over-centre reversal, and the full
# 2*ARM_LEN of travel is usable. ARM_EULER is computed from geometry below.
ARM_RANGE = (0.0, 3.00)                       # rad, mechanical travel
BRAKE_FREE = 0.005                            # m of rigged free travel before the brake bites

# Servo model. The bare drum's own inertia is ~6e-8 kg.m^2, which is numerically
# explosive against any realistic torque cap. SERVO_ARMATURE stands in for the
# gearbox-reflected rotor inertia of the BD10BL and is what makes the joint
# integrable; it is an identification parameter, not a datasheet value.
SERVO_ARMATURE = 5e-5        # kg.m^2, reflected
SERVO_KP = 2.0               # N.m/rad, arm must hold line tension x arm length
SERVO_KV = 0.020             # N.m.s/rad
# SERVO TORQUE CAP, in N.m on the arm hinge.
#
SERVO_TAU_CAP = 0.275          # N.m on the arm hinge
SERVO_TAU_STALL = 0.275      # N.m, BD10BL-CAN datasheet stall at 7.4 V

# Propeller drag-torque / thrust ratio, the skydio_x2 "gear" trick.
# GWS EP8043 at ~10 krpm: P_mech ~44 W -> Q ~0.042 N.m against T = 4.02 N.
KM_KT = 0.0105               # m

# Propeller spin inertia: solid disk, I = m*R^2/2.  NOT AN INPUT -- nothing reads
# I_SPIN.  No <inertial> element is emitted anywhere in this model; the compiler
# derives every tensor from geom shape + explicit mass, and it reproduces the
# line below exactly (prop_disk diaginertia[2] = 2.580640e-05).  Kept as the
# cross-check, and so the number is written down somewhere.
#
# The propeller carries a REAL spin DOF with zero armature, so the inertia is
# entirely the disk's own and H = I*omega is real angular momentum.  MuJoCo's
# own Coriolis terms then produce M = -(omega_body x H) natively; no callback
# computes it. The integrator is Euler (required by the flex skin), which
# applies that Coriolis term explicitly, in full.
I_SPIN = M["prop"] * G * (PROP_D / 2)**2 / 2.0    # kg.m^2, ~2.58e-5
# T = K_T * omega^2, anchored on the spec's static bench point (410 gf) at an
# assumed ~10 krpm.
# 
#  BOTH ends of that anchor are estimates -- fit to real thruster
T_BENCH = 4.02                                # N, spec 
OMEGA_BENCH = 10_000 * 2 * math.pi / 60.0     # rad/s at that figure
K_T = T_BENCH / OMEGA_BENCH**2
# Commanded thrust ceiling, i.e. the ctrlrange on the thrust actuator.
#
# 1.0 N = T/W 0.31 on the 325.4 g all-up mass: the hardware thrust clamp the
# RL policy trains against. Far BELOW the 4.02 N (410 gf) static bench figure,
# so this is a deliberate limit, not what the motor/prop can pull.
THRUST_MAX = 1.0                              # N
OMEGA_MAX = math.sqrt(THRUST_MAX / K_T)       # rad/s needed for THRUST_MAX
ROTOR_KV = 0.005              # N.m.s/rad, velocity-servo gain
ROTOR_TAU_CAP = 0.2           # N.m, spin-up torque limit

# suspension: 4 pod hardpoints x (A row, B row)
# Arm pivots on the servo output face, outboard, about y.
ARM_POS = (SERVO_X, SERVO_Y + SERVO_H / 2 + 0.004, SERVO_Z)
# One merged airframe box
AIRFRAME_BOX = (0.070, 0.045, 0.042)
AIRFRAME_POS = (0.010, 0.0, 0.026)

POD_HP = {
    "fl": (0.050,  0.045, 0.070),     # A row
    "fr": (0.050, -0.045, 0.070),
    "rl": (-0.030,  0.045, 0.070),    # B row
    "rr": (-0.030, -0.045, 0.070),
    "cl": (-0.058,  0.040, 0.070),    # C row, anti-flap
    "cr": (-0.058, -0.040, 0.070),
}
# A (front) and B (rear) rows are CHORD_STATIONS[1] and [3]. The C row is an
# anti-flap line at the trailing edge, on the INBOARD stations.
# Spanwise stations that carry suspension lines, given as span fractions so they
# stay put when N_SPAN changes. 0.5 is the centre; these bracket it symmetrically.
#
# EVENLY SPACED, chosen against the measured spanwise load distribution rather
# than by eye.
#
LINE_FRACTIONS = (0.18, 0.32, 0.46, 0.54, 0.68, 0.82)
# The brake anchors sit on the tip rows, so the rows next to them MUST carry
# lines of their own: otherwise the tip is unsupported exactly where the brake
# load goes in and it simply gets yanked. Real wings run tip ("stabilo") lines
# for this.
LINE_ROWS = sorted({round(f * (N_SPAN - 1)) for f in LINE_FRACTIONS} | {1, N_SPAN - 2})
# The C row goes only on the inboard stations: the brakes pull the tips, so the
# centre trailing edge is the part left unsupported there.
C_ROWS = [r for r in LINE_ROWS if abs(r / (N_SPAN - 1) - 0.5) < 0.25]


def station_rows(r):
    return ("C",) if r in C_ROWS else ("A",)


def row_phi(r: int) -> float:
    """Arc angle of span row r (negative = right wing)."""
    return THETA * (r / (N_SPAN - 1) - 0.5)


def arc_pos(phi: float):
    """Position on the arch, relative to the canopy body origin."""
    return 0.0, R_ARC * math.sin(phi), R_ARC * math.cos(phi) - R_ARC


def vertex_pos(skin: str, s: int, c: int):
    """World position of a vertex at qpos0: surface "u" or "l", span row s,
    chord station c. Thickness is laid off along the arch's outward normal."""
    phi = row_phi(s)
    _, y, z = arc_pos(phi)
    f = chord_fractions()[c]
    h = half_thickness(f) * (1.0 if skin == "u" else -1.0)
    return (CHORD / 2 - f * CHORD, y + h * math.sin(phi), Z_CANOPY + z + h * math.cos(phi))


def vname(skin: str, s: int, c: int) -> str:
    """Body (and site) of a vertex. The lower surface's leading and trailing
    edges ARE the upper surface's."""
    if skin == "l" and c in (0, TE_INDEX):
        skin = "u"
    return f"cv{skin}_{s:02d}_{c}"


def vertex_list():
    """(skin, s, c) of every vertex body, in paramotor_aero.canopy_topology order."""
    topo = canopy_topology(N_SPAN, N_CHORDWISE)
    flat = [None] * (N_SPAN * (2 * N_CHORDWISE - 2))
    for s in range(N_SPAN):
        for c in range(N_CHORDWISE):
            flat[topo["upper"][s, c]] = ("u", s, c)
        for c in range(1, TE_INDEX):
            flat[topo["lower"][s, c]] = ("l", s, c)
    return flat


def vertex_positions():
    return np.array([vertex_pos(*v) for v in vertex_list()])


def station_vertex(r: int, row: str):
    """Vertex carrying the line of span row r, line row A/B/C: the lower
    surface, as on a real wing."""
    return "l", r, ROW_INDEX[row]


# Brakes pull the trailing edge at the OUTBOARD tip, one per side.
BRAKE_VERTEX = {"L": ("u", N_SPAN - 1, TE_INDEX), "R": ("u", 0, TE_INDEX)}


def fabric_area() -> float:
    """Area of both skins and every rib, design shape (m^2)."""
    P, topo = vertex_positions(), canopy_topology(N_SPAN, N_CHORDWISE)
    tris = np.concatenate([topo[k] for k in ("upper_tris", "lower_tris", "rib_tris")])
    return float(np.linalg.norm(triangle_areas(P, tris), axis=1).sum())


def vertex_masses():
    """Canopy mass per vertex (kg), in vertex_list() order: each fabric
    triangle's area split over its corners, scaled to the canopy total."""
    P, topo = vertex_positions(), canopy_topology(N_SPAN, N_CHORDWISE)
    tris = np.concatenate([topo[k] for k in ("upper_tris", "lower_tris", "rib_tris")])
    a = np.linalg.norm(triangle_areas(P, tris), axis=1)
    w = np.zeros(len(P))
    np.add.at(w, tris.ravel(), np.repeat(a / 3, 3))
    return w * (M["fabric"] + M["line_half"]) * G / w.sum()


def flap_tendons():
    """(name, vertex a, vertex b): the brake flap's lower surface, one per row."""
    return [(f"flap{s:02d}", ("l", s, TE_INDEX - 1), ("u", s, TE_INDEX))
            for s in range(N_SPAN)]


def arm_euler(side: str, sgn: float) -> float:
    """Rotation about y that aims the arm's zero at its trailing-edge anchor.

    Rotation about +y maps local +x to (cos a, 0, -sin a), so the aiming angle
    is atan2(-dz, dx) in the pivot-to-anchor direction.
    """
    tx, _, tz = vertex_pos(*BRAKE_VERTEX[side])
    return math.atan2(-(tz - ARM_POS[2]), tx - ARM_POS[0])


def arm_tip(sgn: float):
    """World position of the brake-line anchor at the arm tip, design pose."""
    psi = arm_euler("L", 1.0)                  # both arms share the aiming angle
    return (ARM_POS[0] + ARM_LEN * math.cos(psi),
            sgn * ARM_POS[1],
            ARM_POS[2] - ARM_LEN * math.sin(psi))


def rest_lengths() -> dict:
    """Rigged length of every line and flap tendon, from the design pose.

    A rigged length is a property of the airframe as built, so it is plain
    3-D distance between the two ends.
    """
    out = {}
    for r in LINE_ROWS:
        side = "l" if row_phi(r) > 0 else "r"
        for row in station_rows(r):
            hp = {"A": "f", "B": "r", "C": "c"}[row] + side
            out[f"line_{row}{r}"] = math.dist(POD_HP[hp], vertex_pos(*station_vertex(r, row)))
    for side, sgn in (("L", 1.0), ("R", -1.0)):
        out[f"brake_{side}"] = math.dist(arm_tip(sgn), vertex_pos(*BRAKE_VERTEX[side])) + BRAKE_FREE
    for name, u, v in flap_tendons():
        out[name] = math.dist(vertex_pos(*u), vertex_pos(*v))
    return out


def build(tendon_ranges: dict | None = None, servo_mode: str = "position",
          compat: bool = False) -> str:
    tr = tendon_ranges or {}
    assert servo_mode in ("position", "torque", "both"), servo_mode
    L = []
    A = L.append

    A('<mujoco model="paramotor_1m_peek">')
    A('  <!-- Generated by build_paramotor.py -- do not hand-edit; edit the generator. -->')
    A('  <compiler angle="radian" autolimits="true" balanceinertia="true"/>')
    A('  <!-- density/viscosity are 0 ON PURPOSE. Aerodynamics come from the explicit coefficient model in paramotor_aero.py (Umenberger and Goktogan 2012, eqs. 8-18), applied through qfrc_passive via mj_applyFT. Any non-zero value here re-enables MuJoCo\'s own fluid model on top of it and every force is counted twice. rho lives in paramotor_params.py. -->')
    A('  <!-- Euler (semi-implicit), not implicitfast: MuJoCo 3.13 refuses flex elasticity under the implicit integrators (it asks for "discrete", which MuJoCo Warp does not support). Only the soft skin bending is integrated explicitly; stretch is a constraint. -->')
    A('  <option timestep="0.0005" integrator="Euler" density="0" viscosity="0">')
    A('    <flag multiccd="disable"/>')
    A('  </option>')
    A('  <size njmax="1500" nconmax="200"/>')
    A('')
    A('  <default>')
    A('    <geom contype="0" conaffinity="0" friction="0.6 0.005 0.0001"/>')
    A('    <default class="block">')
    A('      <geom type="box" rgba="0.55 0.58 0.62 1"/>')
    A('    </default>')
    A('    <!-- No fluidshape/fluidcoef: the ellipsoid primitive has no C_L0, no C_Lalpha and no stability derivatives, so it cannot represent the reference model. Lift and drag are applied externally. -->')
    A('    <default class="skin">')
    A('      <geom type="box" rgba="0.85 0.72 0.25 0.55"/>')
    A('    </default>')
    A('    <!-- One class per actuator type: motor/velocity/position share one default slot and would overwrite each other. -->')
    A('    <default class="prop">')
    A(f'      <motor ctrlrange="0 {THRUST_MAX:.1f}"/>')
    A('    </default>')
    A('    <default class="servo_pos">')
    if compat:
        A(f'      <position kp="{SERVO_KP}"')
    else:
        A(f'      <position kp="{SERVO_KP}" kv="{SERVO_KV}"')
    A(f'                ctrlrange="{ARM_RANGE[0]} {ARM_RANGE[1]}"')
    A(f'                forcerange="-{SERVO_TAU_CAP:.4f} {SERVO_TAU_CAP:.4f}"/>')
    A('    </default>')
    A('    <default class="servo_tau">')
    A(f'      <motor gear="1" ctrlrange="-{SERVO_TAU_CAP:.4f} {SERVO_TAU_CAP:.4f}"/>')

    A('    </default>')
    A('    <default class="line">')
    A(f'      <tendon width="{W_LINE}" rgba="0.10 0.95 1.00 1" limited="true"')
    A('              solreflimit="0.006 1" solimplimit="0.95 0.99 0.001"/>')
    A('    </default>')
    A('    <default class="flap">')
    A('      <tendon width="0.0008" rgba="0.75 0.55 0.20 0.8" limited="true"')
    A('              solreflimit="0.004 1" solimplimit="0.95 0.99 0.001"/>')
    A('    </default>')
    A('    <default class="brake">')
    A(f'      <tendon width="{W_BRAKE}" rgba="1.00 0.15 0.10 1" limited="true"')
    A('              solreflimit="0.004 1" solimplimit="0.95 0.99 0.001"/>')
    A('    </default>')
    A('  </default>')
    A('')
    A('  <visual>')
    A('    <headlight ambient="0.45 0.45 0.45" diffuse="0.6 0.6 0.6"/>')
    A('    <scale forcewidth="0.005" contactwidth="0.02" jointlength="0.04" jointwidth="0.004"/>')
    A('  </visual>')
    A('')
    A('  <asset>')
    A('    <material name="mat_block" rgba="0.55 0.58 0.62 1"/>')
    A('  </asset>')
    A('')
    A('  <worldbody>')


    # ---------------- POD ----------------
    A('    <!-- ============================== POD ============================== -->')
    A('    <body name="pod" pos="0 0 0">')
    A('      <freejoint name="pod_free"/>')
    A('      <site name="pod_com" size="0.004" rgba="1 0 1 1"/>')
    A('')
    A('      <!-- Merged airframe block 205.2 g: frame 65 + PCB 68.2 + battery 65 + risers 3 + guides 4. One uniform box cannot reproduce the CG or inertia of the parts it replaces; mass is exact, distribution is not. -->')
    A(f'      <geom name="airframe_block" class="block" type="box"')
    A(f'            pos="{AIRFRAME_POS[0]:.4f} {AIRFRAME_POS[1]:.4f} {AIRFRAME_POS[2]:.4f}"')
    A(f'            size="{AIRFRAME_BOX[0]:.5f} {AIRFRAME_BOX[1]:.5f} {AIRFRAME_BOX[2]:.5f}"')
    A(f'            mass="{M["airframe"]*G:.6f}" rgba="0.38 0.42 0.48 0.85"/>')
    A('      <site name="imu" pos="0.030 0 0.048" size="0.003" rgba="0 1 1 1"/>')
    A('')
    A('      <!-- Battery, kept as its own block: slide it fore/aft to trim CG. -->')
    A(f'      <geom name="battery_block" class="block" pos="0.010 0 -0.020"')
    A(f'            size="{BAT_L/2:.5f} {BAT_W/2:.5f} {BAT_H/2:.5f}"')
    A(f'            mass="{M["battery"]*G:.6f}" rgba="0.75 0.20 0.15 1"/>')
    A('')
    A('      <!-- Motor: SunnySky X2204-1800KV, pusher. Outrunner can is a cylinder about the shaft, not a box. -->')
    A(f'      <geom name="motor_can" type="cylinder" pos="{MOTOR_X:.4f} 0 0.010"')
    A(f'            size="{MOTOR_D/2:.5f} {MOTOR_H/2:.5f}" euler="0 1.5707963 0"')
    A(f'            mass="{M["motor"]*G:.6f}" rgba="0.20 0.20 0.24 1"/>')

    A('')
    A('      <!-- Thrust and propeller reaction torque act on the airframe at the propeller location. Attaching this actuator site to the free rotor applies drag torque to prop_spin and reverses its commanded speed. -->')
    A(f'      <site name="propeller" pos="{PROP_X:.4f} 0 0.010" size="0.004" rgba="1 0.5 0 1" zaxis="1 0 0"/>')
    A('      <!-- Propeller: actuator disk after mujoco_menagerie/skydio_x2. Spins on prop_spin so MuJoCo generates the gyroscopic moment natively. Inertia is the solid-disk value m*R^2/2 taken straight from the geom. -->')
    A(f'      <body name="propeller" pos="{PROP_X:.4f} 0 0.010">')
    A('        <joint name="prop_spin" type="hinge" axis="1 0 0" limited="false"')
    A('               damping="0" frictionloss="0"/>')
    A('        <!-- The fluidcoef=0 opt-out that used to sit here is gone: with option density=0 there is no legacy inertia-box force to exclude the propeller from. -->')
    A(f'        <geom name="prop_disk" type="cylinder" size="{PROP_D/2:.5f} 0.0015"')
    A(f'              euler="0 1.5707963 0" mass="{M["prop"]*G:.6f}" rgba="0.25 0.28 0.34 0.45"/>')
    A('      </body>')
    A('')
    A('      <!-- RISER HARDPOINTS -->')
    for name, (x, y, z) in POD_HP.items():
        A(f'      <site name="hp_{name}" pos="{x:.4f} {y:.4f} {z:.4f}" size="0.0025" rgba="0 0.9 0.2 1"/>')

    A('')

    # ---- steering: servo block, capstan drum, take-up slider, guide pulley ----
    for side, sgn in (("L", 1.0), ("R", -1.0)):
        sy = sgn * SERVO_Y
        A(f'      <!-- ---------- STEERING CHANNEL {side} ---------- -->')
        A(f'      <geom name="servo_{side}" class="block" pos="{SERVO_X:.4f} {sy:.4f} {SERVO_Z:.4f}"')
        A(f'            size="{SERVO_BOX[0]:.5f} {SERVO_BOX[1]:.5f} {SERVO_BOX[2]:.5f}"')
        A(f'            mass="{M["servo"]*G:.6f}" rgba="0.15 0.18 0.55 1"/>')
        A('')
        A('      <!-- Servo arm on the output face, swinging about y: the shaft axis follows the case when the servo is laid flat. The brake line is anchored at the arm tip. -->')
        A(f'      <body name="arm_{side}" pos="{ARM_POS[0]:.4f} {sgn*ARM_POS[1]:.4f} {ARM_POS[2]:.4f}"')
        A(f'            euler="0 {arm_euler(side, sgn):.6f} 0">')
        dmp = SERVO_KV if compat else 0.0015   # compat has no position kv; use joint damping
        # Both arms swing about +y, NOT mirrored axes. Mirroring the axis makes
        # the two arms sweep opposite ways in x-z, which reads as one servo
        # installed backwards even though the brake pull stays symmetric (the
        # line length depends only on the angle away from its own anchor).
        # Same axis = the pair looks like a true mirror image on screen, and
        # corresponds to both servos mounted the same way up.
        A(f'        <joint name="arm_{side}" type="hinge" axis="0 1 0" damping="{dmp}"')
        A(f'               armature="{SERVO_ARMATURE:.2e}" frictionloss="0.0008"')
        A(f'               range="{ARM_RANGE[0]} {ARM_RANGE[1]}"/>')
        A(f'        <geom name="arm_geom_{side}" class="block" pos="{ARM_LEN/2:.4f} 0 0"')
        A(f'              size="{ARM_LEN/2:.4f} 0.0025 0.0015"')
        A(f'              mass="{M["arm"]*G:.6f}" rgba="0.85 0.55 0.10 1"/>')
        A(f'        <site name="bl_start_{side}" pos="{ARM_LEN:.4f} 0 0" size="0.0020"')
        A('              rgba="0.9 0.1 0.1 1"/>')
        A('      </body>')
        A('')

    A('    </body>  <!-- /pod -->')
    A('')

    # ---------------- CANOPY ----------------
    topo = canopy_topology(N_SPAN, N_CHORDWISE)
    verts = vertex_list()
    A('    <!-- ============================ CANOPY ============================= -->')
    A(f'    <!-- "Fake hollow" deformable canopy: the inflated double-skin shape ({THICKNESS:.0%} thick) as a solid, {N_SPAN} span rows; upper surface {N_CHORDWISE} chord stations leading edge to trailing edge, lower surface {N_CHORDWISE - 2}, sharing both edges. Every vertex is a point mass free to translate on three world-axis slides; the flexes below span them. The {M["fabric"] + M["line_half"]:.0f} g canopy (both skins, ribs, line allowance) is split by tributary fabric area, so mass and inertia are the hollow wing\'s. -->')
    masses = vertex_masses()
    for v, mass in zip(verts, masses):
        x, y, z = vertex_pos(*v)
        n = vname(*v)
        A(f'    <body name="{n}" pos="{x:.6f} {y:.6f} {z:.6f}">')
        for ax, vec in (("x", "1 0 0"), ("y", "0 1 0"), ("z", "0 0 1")):
            A(f'      <joint name="{n}_{ax}" type="slide" axis="{vec}"/>')
        A(f'      <inertial pos="0 0 0" mass="{mass:.8f}" diaginertia="1e-10 1e-10 1e-10"/>')
        A(f'      <site name="{n}" size="0.002" rgba="1 1 0 0.6"/>')
        A('    </body>')
    A('  </worldbody>')
    A('')
    A('  <deformable>')
    A('    <!-- Stiffness of the double skin from edge-length equalities (see build_paramotor.py): canopy_solid is the leading edge to the last lower-surface station, canopy_flap the upper surface of the brake flap. No bending, contact or self-collision. -->')
    tets = topo["tets"].copy()
    P0 = vertex_positions()
    a_, b_, c_, d_ = (P0[tets[:, k]] for k in range(4))
    neg = np.einsum("ij,ij->i", np.cross(b_ - a_, c_ - a_), d_ - a_) < 0
    tets[neg, 1], tets[neg, 2] = tets[neg, 2], tets[neg, 1].copy()
    flap = topo["upper_tris"].reshape(N_SPAN - 1, N_CHORDWISE - 1, 2, 3)[:, -1].reshape(-1, 3)
    for name, dim, elems, rgba in (("canopy_solid", 3, tets, "0.85 0.72 0.25 0.75"),
                                   ("canopy_flap", 2, flap, "0.80 0.62 0.22 0.75")):
        used = sorted(set(elems.ravel()))
        local = {g: k for k, g in enumerate(used)}
        A(f'    <flex name="{name}" dim="{dim}" radius="0.0005" rgba="{rgba}"')
        A(f'          body="{" ".join(vname(*verts[g]) for g in used)}"')
        A(f'          vertex="{" ".join(["0 0 0"] * len(used))}"')
        A(f'          element="{" ".join(str(local[g]) for g in elems.ravel())}">')
        A('      <contact contype="0" conaffinity="0" selfcollide="none"/>')
        A('    </flex>')
    A('  </deformable>')
    A('')
    A('  <equality>')
    for name in ("canopy_solid", "canopy_flap"):
        A(f'    <flex flex="{name}" solref="{FLEX_SOLREF}"/>')
    A('  </equality>')
    A('')

    # ---------------- TENDONS ----------------
    A('  <tendon>')
    A('    <!-- ---- suspension lines: UHMWPE, tension-only (upper limit = taut), on the lower surface ---- -->')
    for r in LINE_ROWS:
        side = "l" if row_phi(r) > 0 else "r"
        rows = [(row, {"A": "f", "B": "r", "C": "c"}[row] + side) for row in station_rows(r)]
        for row, hp in rows:
            name = f"line_{row}{r}"
            rng = tr.get(name, 0.40)
            A(f'    <spatial name="{name}" class="line" range="0 {rng:.6f}">')
            A(f'      <site site="hp_{hp}"/>')
            A(f'      <site site="{vname(*station_vertex(r, row))}"/>')
            A('    </spatial>')
    A('')
    A('    <!-- Brake lines: servo arm tip straight to the trailing-edge tip vertex, two sites. Tension-only, like the suspension: slack below L0, carrying load at L0. Pulling it deflects the deformable trailing edge. -->')
    for side in ("L", "R"):
        name = f"brake_{side}"
        rng = tr.get(name, 0.40)
        A(f'    <spatial name="{name}" class="brake" range="0 {rng:.6f}">')
        A(f'      <site site="bl_start_{side}"/>')
        A(f'      <site site="{vname(*BRAKE_VERTEX[side])}"/>')
        A('    </spatial>')
    A('')
    A('    <!-- Brake flap lower surface: tension-only, so the trailing edge deflects down under the brake but lift cannot push it up past the design section. -->')
    for name, u, v in flap_tendons():
        A(f'    <spatial name="{name}" class="flap" range="0 {tr.get(name, 0.05):.6f}">')
        A(f'      <site site="{vname(*u)}"/>')
        A(f'      <site site="{vname(*v)}"/>')
        A('    </spatial>')
    A('  </tendon>')
    A('')

    # ---------------- EQUALITY: the capstan ----------------

    # ---------------- ACTUATORS ----------------
    A('  <actuator>')
    A(f'    <!-- Propeller, skydio_x2 pattern, ONE control. Ceiling {THRUST_MAX:.1f} N = T/W {THRUST_MAX/(BOM_TOTAL_G*G*9.81):.2f}, the hardware thrust clamp, well BELOW the {T_BENCH} N (410 gf) static bench figure. The prop_spin DOF exists only to carry angular momentum so MuJoCo generates the gyroscopic moment natively; it has no actuator and is driven kinematically from thrust, omega = sqrt(T/K_T). Nothing is drawn spinning: the disk is axisymmetric. -->')
    A(f'    <motor class="prop" name="thrust" site="propeller" gear="0 0 1 0 0 {-KM_KT:.4f}"/>')
    A('')
    A(f'    <!-- Two physical servos. servo_mode = "{servo_mode}" (position | torque |')
    A('         both). With "both", drive ONE channel and leave the other ctrl at 0:')
    A(f'         MuJoCo sums actuators on the same joint. forcerange is a SOFTWARE cap of {SERVO_TAU_CAP:.1f} N.m')
    A(f'         on the arm hinge. The real BD10BL-CAN stalls at {SERVO_TAU_STALL} N.m (7.4 V), so the')
    A('         hardware is the binding limit, not this number. -->')
    if servo_mode in ("position", "both"):
        for side in ("L", "R"):
            A(f'    <position class="servo_pos" name="servo_pos_{side}" joint="arm_{side}"/>')
    if servo_mode in ("torque", "both"):
        for side in ("L", "R"):
            A(f'    <motor class="servo_tau" name="servo_tau_{side}" joint="arm_{side}"/>')
    A('  </actuator>')
    A('')
    A('  <sensor>')
    A('    <!-- Sensors mirror the BOM: LSM6DSOX accel+gyro, BMP581 baro, GOKU GM10 Pro GNSS + QMC5883L compass. All read GROUND TRUTH: MuJoCo 3.13 dropped the sensor-noise feature, so the noise attribute parses but is never applied. Degrade these yourself, see sensor_demo.py. -->')
    A('    <framequat name="pod_quat" objtype="site" objname="imu"/>')
    A('    <gyro name="gyro" site="imu"/>')
    A('    <accelerometer name="accel" site="imu"/>')
    A('    <magnetometer name="mag" site="imu"/>')
    A('    <velocimeter name="vel_body" site="imu"/>')
    A('    <framepos name="pod_pos" objtype="site" objname="pod_com"/>')
    A('    <framelinvel name="pod_vel" objtype="site" objname="pod_com"/>')
    A('    <frameangvel name="pod_angvel" objtype="site" objname="pod_com"/>')
    A('    <jointvel name="prop_omega" joint="prop_spin"/>')
    for side in ("L", "R"):
        A(f'    <jointpos name="arm_pos_{side}" joint="arm_{side}"/>')
        A(f'    <tendonpos name="brake_len_{side}" tendon="brake_{side}"/>')

    A('  </sensor>')
    A('</mujoco>')
    return _sanitize_comments("\n".join(L) + "\n")


def _sanitize_comments(xml: str) -> str:
    """Make every comment body legal XML.

    Two rules, enforced on the whole emitted document so they cannot regress:
    (1) the XML spec forbids "--" inside a comment and a body ending in "-";
        MuJoCo tolerates both, strict XML viewers reject the file outright.
    (2) every comment is collapsed onto ONE line.
    """
    out, i = [], 0
    while True:
        a = xml.find("<!--", i)
        if a < 0:
            out.append(xml[i:])
            break
        b = xml.find("-->", a + 4)
        assert b > a, "unterminated XML comment"
        out.append(xml[i:a + 4])
        body = re.sub(r"-{2,}", "-", xml[a + 4:b])
        body = " " + re.sub(r"\s+", " ", body).strip() + " "   # single line, always
        if body.endswith("- "):
            body = body[:-2] + " "
        out.append(body)
        out.append("-->")
        i = b + 3
    return "".join(out)


def build_scene(model_file: str = "paramotor.xml") -> str:
    """The mujoco_menagerie split: scene.xml is the world, and it <include>s the
    aircraft. paramotor.xml stays the aircraft alone, so it can be dropped into
    another scene or loaded bare.

    Everything here is VISUAL ONLY: collision is off on the ground and terrain,
    so loading scene.xml gives dynamics identical to loading paramotor.xml.
    """
    L = []
    A = L.append
    A('<mujoco model="paramotor_scene">')
    A('  <!-- Generated by build_paramotor.py - do not hand-edit; edit the generator. -->')
    A(f'  <include file="{model_file}"/>')
    A('')
    A('    <!-- CRITICAL: znear and zfar are MULTIPLES of stat.extent, and the terrain')
    A('         drags the auto-computed extent from ~1 m to ~256 m. That pushes the near')
    A('         clip plane out to ~2.6 m and the whole aircraft disappears whenever the')
    A('         camera is closer than that. Pin the extent to the aircraft. -->')
    A('    <statistic extent="1.0" center="0 0 0.3"/>')
    A('')
    A('  <visual>')
    A('    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5" specular="0 0 0"/>')
    A('    <map znear="0.005" zfar="400"/>')
    A('    <quality shadowsize="4096"/>')
    A('  </visual>')
    A('')
    A('  <asset>')
    A('    <texture type="skybox" builtin="gradient" rgb1="0.45 0.62 0.86"')
    A('             rgb2="0.10 0.14 0.24" width="512" height="512"/>')
    A('    <texture name="grid" type="2d" builtin="checker" rgb1="0.22 0.29 0.21"')
    A('             rgb2="0.27 0.35 0.25" width="300" height="300"/>')
    A('    <material name="grid" texture="grid" texrepeat="60 60" texuniform="true"')
    A('             reflectance="0.02"/>')
    A('    <material name="rock" rgba="0.40 0.42 0.38 1" reflectance="0.02"/>')
    A('    <!-- Ships flat: MuJoCo allocates zeros when no file is given. make_terrain() in view_paramotor.py fills it procedurally at load. -->')
    A(f'    <hfield name="terrain" nrow="96" ncol="96"')
    A(f'            size="{TERRAIN_HALF} {TERRAIN_HALF} {TERRAIN_H} 1.0"/>')
    A('  </asset>')
    A('')
    A('  <worldbody>')
    A('    <!-- Directional, not a local lamp: a positional light at the origin leaves the rig unlit once it flies away from it. -->')
    A('    <light directional="true" pos="0 0 10" dir="-0.3 -0.4 -1"')
    A('           diffuse="0.9 0.9 0.85" specular="0.2 0.2 0.2"/>')
    A('    <!-- Ground and mountains give the parallax needed to see motion. -->')
    A(f'    <geom name="ground" type="plane" size="0 0 0.05" pos="0 0 {GROUND_Z}"')
    A('          material="grid" contype="0" conaffinity="0"/>')
    A(f'    <geom name="mountains" type="hfield" hfield="terrain" pos="0 0 {GROUND_Z:.1f}"')
    A('          material="rock" contype="0" conaffinity="0"/>')
    A('  </worldbody>')
    A('</mujoco>')
    return _sanitize_comments("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "paramotor.xml"))
    ap.add_argument("--servo-mode", default="position",
                    choices=["position", "torque", "both"],
                    help="which drive channel(s) to emit per servo (default: position)")
    ap.add_argument("--compat", action="store_true",
                    help="older runtimes (mujoco-js / WASM): position servos "
                         "without kv; joint damping stands in")
    args = ap.parse_args()

    out = Path(args.out)
    ranges = rest_lengths()
    out.write_text(build(ranges, args.servo_mode, args.compat))

    print(f"geometry:  arc half-angle {PHI_HALF:.4f} rad, R = {R_ARC:.4f} m, "
          f"arc {THETA:.4f} rad")
    print(f"           flat area {S_FLAT:.3f} m^2, projected {S_FLAT*PROJ_FRACTION:.3f} m^2")
    print(f"section:   {THICKNESS:.0%} thick; hollow fabric {fabric_area():.3f} m^2 "
          f"-> PEEK {fabric_thickness()*1e6:.0f} um, double-skin EI {sandwich_ei():.0f} N.m per m")
    print(f"BOM total: {BOM_TOTAL_G:.1f} g  (spec itemised sum = 325.4 g)")

    try:
        import mujoco
    except ImportError:
        print("mujoco not installed -- wrote model, skipped verification",
              file=sys.stderr)
        return

    m = mujoco.MjModel.from_xml_path(str(out))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    total = m.body_mass.sum()

    # Verify the analytic lengths against the compiled geometry. This is a
    # correctness check on the arithmetic, not a calibration: nothing is
    # measured from it and nothing is written back.
    worst = 0.0
    for t in range(m.ntendon):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TENDON, t)
        want = ranges[name] - (BRAKE_FREE if name.startswith("brake") else 0.0)
        worst = max(worst, abs(float(d.ten_length[t]) - want))
    # Tolerance is set by XML print precision, not by the arithmetic: site and
    # body positions are emitted at %.5f, i.e. 10 um resolution, so a rest
    # length built from several of them can differ by a few um.
    assert worst < 5e-5, f"analytic rest length disagrees with geometry by {worst:.2e} m"

    print(f"compiled:  {m.nbody} bodies, {m.nq} nq, {m.ntendon} tendons, "
          f"{m.neq} equalities, {m.nu} actuators")
    print(f"model mass {total*1e3:.2f} g")
    print(f"rest lengths computed analytically, match geometry to {worst*1e9:.1f} nm")
    scene = out.parent / "scene.xml"
    scene.write_text(build_scene(out.name))
    ms = mujoco.MjModel.from_xml_path(str(scene))
    assert abs(ms.body_mass.sum() - total) < 1e-9, "scene.xml changed the mass"
    assert ms.nbody == m.nbody and ms.nu == m.nu, "scene.xml changed the model"
    print(f"scene:     {scene.name} includes {out.name}, "
          f"+{ms.ngeom - m.ngeom} visual geoms, mass unchanged")

    import xml.dom.minidom
    try:
        xml.dom.minidom.parse(str(out))
    except Exception as e:                       # noqa: BLE001
        raise SystemExit(f"emitted XML is not strictly valid: {e}")
    assert "--" not in re.sub(r"-->", "", "".join(
        re.findall(r"<!--(.*?)-->", out.read_text(), re.S))), "'--' inside a comment"
    print("strict XML parse: ok (no '--' inside any comment)")
    print(f"wrote {out} and scene.xml")


if __name__ == "__main__":
    main()
