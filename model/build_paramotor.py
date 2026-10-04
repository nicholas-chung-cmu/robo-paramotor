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

HERE = Path(__file__).parent

# ----------------------------------------------------------------------------
# MASS BUDGET  (grams) -- traceable to the BOM table in itemized_spec.md
# ----------------------------------------------------------------------------
G = 1e-3  # gram -> kg

# Number of spanwise canopy strips. The deformable skin has a spanwise vertex
# row at both tips and at each strip centre (N_PANEL + 2 rows), so the line
# stations, which sit at strip centres, land on vertices. The mass table, the
# arc geometry, the line stations and the brake anchors all scale from it.
N_PANEL = 8

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
    # Canopy: 65 g of 125 um PEEK skin PLUS the 3 g canopy half of the 6 g line
    # allowance, split N_PANEL ways (then spread over the skin's vertices by
    # tributary area, see vertex_masses()).
    #
    # The 3 g used to be eight separate "line-termination tab" geoms, one per
    # suspension station. They were deleted: NO TENDON EVER ATTACHED TO THEM.
    # The suspension lines terminate on the att_A_* / att_C_* sites, which live
    # on the panel bodies, so the tabs were pure mass sitting near - but not on
    # - the attachment points, and not on the load path at all.
    #
    # The mass is real hardware and is kept, just carried by the panels. Cost
    # of the move, measured: canopy Ixx -0.4%, Iyy -0.3%, Izz -0.4%. The tabs
    # sat at roughly the canopy's own radius of gyration, so spreading them
    # over fourteen panels instead of eight stations barely moves the tensor.
    # (Deleting the mass outright would have cost 4.5% of Ixx and broken the
    # 325.40 g BOM total, which the suite asserts.)
    "panel":      (65.0 + 3.0) / N_PANEL,
}

BOM_TOTAL_G = (
    M["airframe"] + M["battery"] + M["motor"] + M["prop"]
    + 2 * M["servo"] + 2 * M["arm"] + N_PANEL * M["panel"]
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
SKIN_T = 125e-6                          # m, PEEK thickness (real)
SKIN_T_DRAW = 1.0e-3                     # m, drawn thickness (renderable); mass is set explicitly

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

# DEFORMABLE SKIN. The canopy is a MuJoCo flex shell (dim=2) over a grid of
# point-mass vertices, not a rigid body. Material is the real skin:
#   PEEK E = 3.6 GPa, t = 125 um, nu = 0.4,  D = E t^3 / 12(1-nu^2) ~ 7e-4 N.m.
# Bending uses that plate stiffness (elastic2d="bend"); a 125 um film is nearly
# limp, so the wing holds its shape through line tension and air load, as a
# real single-skin wing does. In-plane the film is effectively inextensible at
# these loads (Et = 4.5e5 N/m, strain ~1e-4), so stretch is a flex equality
# constraint on every edge rather than a spring: a 4.5e5 N/m spring on sub-gram
# vertices would need a timestep around 0.05 ms to stay stable.
PEEK_E, PEEK_T, PEEK_NU = 3.6e9, 125e-6, 0.4
PLATE_D = PEEK_E * PEEK_T**3 / (12 * (1 - PEEK_NU**2))
FLEX_SOLREF = "0.004 1"       # edge-length constraint: stiff, critically damped
# SAIL PROFILE. The skin is a closed airfoil section: a NACA 4-digit section's
# surface, laid off along the arch's outward normal, running from the trailing
# edge forward along the underside, round the leading edge, then aft over the
# top and back to the trailing edge, where it closes on itself.
#
# RIBS. Every span row carries a flat internal RIB across the section: a
# separate flex with edge-length constraints and NO bending, joining upper and
# lower stations as a ladder of triangles. A rib-less closed skin cannot hold
# an airfoil (no internal pressure either): with correct section aerodynamics
# it collapses in flight. The ribs make each section rigid in its own plane;
# the wing still flexes spanwise, between rows.
#
# TIPS. Toward each tip the section thins elliptically, over the outer
# TIP_ROUND of the half-span, to TIP_MIN of its thickness, so the wing rounds
# off; the tip rows' ribs close the wing ends. The thickness deliberately
# stops short of zero. Folding the upper surface onto the lower
# makes a 180-degree crease in the bending skin, where MuJoCo's flex bending
# (not stress-free on curved rest shapes) put ~0.7 N on 24 mg tip vertices and
# the simulation went to NaN.
SAIL_CAMBER = 0.04            # max camber, fraction of chord (NACA 4415)
SAIL_CAMBER_POS = 0.4         # its position, fraction of chord from the LE
SAIL_THICKNESS = 0.15         # section thickness, fraction of chord
TIP_ROUND = 0.4               # outer fraction of each half-span that rounds off
TIP_MIN = 0.35                # thickness at the tips, fraction of the inboard section
# RIGGING. The whole canopy is pitched nose-down by RIG_NOSE_DOWN about the
# centre-row A-line point (the line lengths follow from the pitched shape).
# This sets the wing's trim angle of attack. Trim study (runs/tools/trim.py):
# 4-6 deg glides best (L/D 2.6-2.7) but deep-stalls under power; 10 deg glides
# at L/D 2.2 and cruises near level at 0.8 N (4.5 m/s, sink 0.55 m/s). Above
# ~1.5 N the propeller torque rolls every setting into a spiral.
RIG_NOSE_DOWN = math.radians(10.0)
A_FRAC = 0.5 - 0.060 / CHORD  # A line row, fraction of chord from the LE (0.194)
B_FRAC = 0.5 + 0.060 / CHORD  # B line row (0.806)
# Chordwise vertex stations as (surface, fraction of chord from the LE), round
# the section; the skin closes from the last station (the trailing edge) back
# to the first. The A and B line rows are stations on the UNDERSIDE, as on a
# real wing, so attachments land on vertices; the C (anti-flap) row attaches
# at the trailing-edge vertex, where the two surfaces meet.
SAIL_STATIONS = (("l", B_FRAC), ("l", 0.5), ("l", A_FRAC), ("l", 0.08), ("l", 0.025),
                 ("u", 0.0), ("u", 0.025), ("u", 0.08),
                 ("u", A_FRAC), ("u", 0.5), ("u", B_FRAC), ("u", 1.0))
# Lift and drag act on the upper surface, leading edge aft: one lifting surface
# for the wing as a whole. The lower surface faces backwards in the station
# order, where a strip element's angle of attack means nothing, so it is
# structure and mass only. The model records the leading-edge station for
# paramotor_aero.CanopyMesh (custom numeric canopy_le_station).
LE_INDEX = 5
TE_INDEX = len(SAIL_STATIONS) - 1
ROW_INDEX = {"A": 2, "B": 0, "C": TE_INDEX}


def tip_thickness(s: int) -> float:
    """Thickness factor of span row s: 1 inboard, elliptical to TIP_MIN at the tips."""
    eta = abs(span_phis()[s]) / (THETA / 2)
    if eta <= 1 - TIP_ROUND:
        return 1.0
    e = (eta - 1 + TIP_ROUND) / TIP_ROUND
    return TIP_MIN + (1 - TIP_MIN) * math.sqrt(max(0.0, 1 - e * e))


def rib_elements():
    """Triangles across every row's section (grid indices). Upper and lower
    stations at the same chord fraction are joined, so each rib is a ladder of
    quads from a nose triangle to a trailing-edge triangle."""
    u = {f: k for k, (side, f) in enumerate(SAIL_STATIONS) if side == "u"}
    l = {f: k for k, (side, f) in enumerate(SAIL_STATIONS) if side == "l"}
    fs = sorted(l)                                 # lower fractions, nose to tail
    tri = [(u[0.0], u[fs[0]], l[fs[0]])]           # nose
    for f0, f1 in zip(fs, fs[1:]):
        tri += [(u[f0], u[f1], l[f1]), (u[f0], l[f1], l[f0])]
    tri.append((u[fs[-1]], u[1.0], l[fs[-1]]))     # trailing edge
    out = []
    for s in range(N_SPAN):
        out += [s * N_CHORDWISE + c for t in tri for c in t]
    return out


def section_point(side: str, f: float, thick: float = 1.0):
    """(x, h) of a NACA 4-digit surface point at chord fraction f, unscaled (m):
    x forward of mid-chord, h above the chord line. side is "u" or "l"; thick
    scales the thickness (the camber line stays)."""
    m, q, t = SAIL_CAMBER, SAIL_CAMBER_POS, SAIL_THICKNESS * thick
    if f < q:
        yc, slope = m / q**2 * (2 * q * f - f * f), 2 * m / q**2 * (q - f)
    else:
        yc = m / (1 - q)**2 * (1 - 2 * q + 2 * q * f - f * f)
        slope = 2 * m / (1 - q)**2 * (q - f)
    yt = 5 * t * (0.2969 * math.sqrt(f) - 0.1260 * f - 0.3516 * f**2
                  + 0.2843 * f**3 - 0.1036 * f**4)
    th, sg = math.atan(slope), (1.0 if side == "u" else -1.0)
    xs = f - sg * yt * math.sin(th)
    return (0.5 - xs) * CHORD, (yc + sg * yt * math.cos(th)) * CHORD


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
Z_CANOPY = LINE_HEIGHT_FRAC * SPAN_FLAT       # canopy chord line above pod origin
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
# 2.0 N = T/W 0.63 on the 325.4 g all-up mass: the hardware thrust clamp the
# RL policy trains against. Below the 4.02 N (410 gf) static bench figure, so
# this is a deliberate limit, not what the motor/prop can pull. Raised from
# 1.0 N; on the earlier rigid canopy powered flight departed above ~1.7 N
# (docs/MODEL_NOTES.md), not re-checked on the current canopy.
THRUST_MAX = 2.0                              # N
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
# A (front) and B (rear) rows are SAIL_STATIONS[5] and [7]. The C row is an
# anti-flap line at the trailing edge, on the INBOARD stations.
# Spanwise stations that carry suspension lines, given as span fractions so they
# stay put when N_PANEL changes. 0.5 is the centre; these bracket it symmetrically.
#
# EVENLY SPACED, chosen against the measured spanwise load distribution rather
# than by eye.
#
LINE_FRACTIONS = (0.18, 0.32, 0.46, 0.54, 0.68, 0.82)
# The brake anchors sit on the tip panels, so those panels MUST carry lines of
# their own: otherwise the tip is unsupported exactly where the brake load goes
# in and it simply gets yanked. Real wings run tip ("stabilo") lines for this.
TIP_PANELS = {0, N_PANEL - 1}
LINE_PANELS = sorted({min(N_PANEL - 1, max(0, int(round(f * N_PANEL - 0.5))))
                      for f in LINE_FRACTIONS} | TIP_PANELS)
# The C row goes only on the inboard stations: the brakes pull the tips, so the
# centre trailing edge is the part left unsupported there.
C_PANELS = [i for i in LINE_PANELS
            if abs((i + 0.5) / N_PANEL - 0.5) < 0.25]


def station_rows(i):
    """Line rows at strip i: every station has an A line at the front; the
    inboard ones also have a C line to the trailing edge."""
    return ("A", "C") if i in C_PANELS else ("A",)


def panel_phi(i: int) -> float:
    """Arc angle of the centre of panel i (negative = right wing)."""
    return THETA * ((i + 0.5) / N_PANEL - 0.5)


def arc_pos(phi: float):
    """Position on the arch, relative to the canopy body origin."""
    return 0.0, R_ARC * math.sin(phi), R_ARC * math.cos(phi) - R_ARC


def span_phis():
    """Arc angle of each spanwise vertex row: both tips and every strip centre.

    Row 0 is the right tip (negative y); the last row is the left tip.
    """
    return [-THETA / 2] + [panel_phi(i) for i in range(N_PANEL)] + [THETA / 2]


N_SPAN = N_PANEL + 2
N_CHORDWISE = len(SAIL_STATIONS)


def vertex_pos(s: int, c: int):
    """World position of skin vertex (spanwise row s, chord station c) at qpos0."""
    return _vertex_pos(s, c, SECTION_SCALE)


def _vertex_pos(s: int, c: int, k: float):
    """Vertex position with the section (chord stations and sail profile)
    scaled by k. The sail profile is laid off along the arch's outward normal."""
    phi = span_phis()[s]
    _, y, z = arc_pos(phi)
    x, h = section_point(*SAIL_STATIONS[c], tip_thickness(s))
    x, h = k * x, k * h
    # Rigging pitch about the centre A-line point (positive = nose down).
    xa, _ = section_point("u", A_FRAC)
    dx, dz = x - k * xa, z + h * math.cos(phi)
    ca, sa = math.cos(RIG_NOSE_DOWN), math.sin(RIG_NOSE_DOWN)
    return (k * xa + dx * ca + dz * sa, y + h * math.sin(phi),
            Z_CANOPY - dx * sa + dz * ca)


def _skin_area(k: float) -> float:
    """Sum of the lifting cells' areas (leading edge aft), as
    paramotor_aero.cell_frames measures them."""
    P = [[_vertex_pos(s, c, k) for c in range(N_CHORDWISE)] for s in range(N_SPAN)]
    sub = lambda u, v: [u[i] - v[i] for i in range(3)]
    total = 0.0
    for s in range(N_SPAN - 1):
        for c in range(LE_INDEX, N_CHORDWISE - 1):
            a, b, cc, d = P[s][c], P[s + 1][c], P[s][c + 1], P[s + 1][c + 1]
            p, q = sub(d, a), sub(b, cc)
            x = (p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0])
            total += 0.5 * math.sqrt(sum(v * v for v in x))
    return total


def _solve_section_scale() -> float:
    """Scale that makes the lifting sail's surface area the flat area S_FLAT.

    Flat area is measured along the surface, as for a real wing. The sail's
    curve and its offset from the arch both add area, so the section is
    scaled down (chord and profile together, keeping the airfoil's shape).
    The nose wrap is extra fabric on top of it.
    """
    lo, hi = 0.5, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _skin_area(mid) < S_FLAT else (lo, mid)
    return 0.5 * (lo + hi)


SECTION_SCALE = _solve_section_scale()


def vname(s: int, c: int) -> str:
    return f"cv_{s:02d}_{c}"


def station_vertex(i: int, row: str):
    """Vertex carrying the line of strip i, row A/B/C."""
    return i + 1, ROW_INDEX[row]


# Brakes pull the trailing edge at the OUTBOARD tip, one per side.
BRAKE_VERTEX = {"L": (N_SPAN - 1, TE_INDEX), "R": (0, TE_INDEX)}


def _tributary(u):
    """Half the distance to each neighbour: the length a station represents."""
    u = list(u)
    return [((u[min(k + 1, len(u) - 1)] - u[max(k - 1, 0)]) / 2) for k in range(len(u))]


def vertex_masses():
    """Skin mass per vertex (kg), by tributary area; sums to the canopy total."""
    ws = _tributary([R_ARC * phi for phi in span_phis()])
    pts = [section_point(*st) for st in SAIL_STATIONS]
    n = len(pts)                            # closed loop: TE joins station 0
    wc = [(math.dist(pts[k - 1], pts[k]) + math.dist(pts[k], pts[(k + 1) % n])) / 2
          for k in range(n)]
    total = N_PANEL * M["panel"] * G
    norm = sum(ws) * sum(wc)
    return [[total * ws[s] * wc[c] / norm for c in range(N_CHORDWISE)]
            for s in range(N_SPAN)]


def skin_elements():
    """Two triangles per grid cell, consistently wound. The last chord column
    closes the section, from the trailing edge back to station 0."""
    tri = []
    for s in range(N_SPAN - 1):
        for c in range(N_CHORDWISE):
            c1 = (c + 1) % N_CHORDWISE
            a, b = s * N_CHORDWISE + c, (s + 1) * N_CHORDWISE + c
            a1, b1 = s * N_CHORDWISE + c1, (s + 1) * N_CHORDWISE + c1
            tri += [a, a1, b, b, a1, b1]
    return tri


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
    """Rigged length of every line, computed from geometry in the design pose.

    A rigged length is a property of the airframe as built, so it is plain
    3-D distance between the two ends. 
    """
    out = {}
    for i in LINE_PANELS:
        side = "l" if panel_phi(i) > 0 else "r"
        for row in station_rows(i):
            hp = {"A": "f", "B": "r", "C": "c"}[row] + side
            out[f"line_{row}{i}"] = math.dist(POD_HP[hp], vertex_pos(*station_vertex(i, row)))
    for side, sgn in (("L", 1.0), ("R", -1.0)):
        out[f"brake_{side}"] = math.dist(arm_tip(sgn), vertex_pos(*BRAKE_VERTEX[side])) + BRAKE_FREE
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
    A('  <!-- njmax: the closed skin alone has ~600 edge constraints; MuJoCo Warp drops rows beyond it. -->')
    A('  <size njmax="1000" nconmax="200"/>')
    A('  <custom>')
    A(f'    <numeric name="canopy_le_station" data="{LE_INDEX}"/>')
    A('  </custom>')
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
    A('    <!-- ============================ CANOPY ============================= -->')
    A(f'    <!-- Deformable single skin: {N_SPAN} x {N_CHORDWISE} point-mass vertices (spanwise rows at both tips and every strip centre; chord stations round a closed airfoil section: trailing edge forward along the underside, round the leading edge, aft over the top), each free to translate on three world-axis slides. The flex shell below spans them. Vertex masses are the 68 g skin split by tributary area. -->')
    masses = vertex_masses()
    for s_ in range(N_SPAN):
        for c in range(N_CHORDWISE):
            x, y, z = vertex_pos(s_, c)
            n = vname(s_, c)
            A(f'    <body name="{n}" pos="{x:.6f} {y:.6f} {z:.6f}">')
            for ax, vec in (("x", "1 0 0"), ("y", "0 1 0"), ("z", "0 0 1")):
                A(f'      <joint name="{n}_{ax}" type="slide" axis="{vec}"/>')
            A(f'      <inertial pos="0 0 0" mass="{masses[s_][c]:.8f}" diaginertia="1e-10 1e-10 1e-10"/>')
            A(f'      <site name="{n}" size="0.002" rgba="1 1 0 0.6"/>')
            A('    </body>')
    A('  </worldbody>')
    A('')
    A('  <deformable>')
    A(f'    <!-- PEEK skin: bending from plate stiffness D = E t^3/12(1-nu^2) = {PLATE_D:.2e} N.m; stretch is the flex equality below (inextensible film). No contact or self-collision (MuJoCo Warp has no flex self-collision). -->')
    bodies = " ".join(vname(s_, c) for s_ in range(N_SPAN) for c in range(N_CHORDWISE))
    A(f'    <flex name="canopy" dim="2" radius="0.0005" rgba="0.85 0.72 0.25 0.7"')
    A(f'          body="{bodies}"')
    A(f'          vertex="{" ".join(["0 0 0"] * (N_SPAN * N_CHORDWISE))}"')
    A(f'          element="{" ".join(map(str, skin_elements()))}">')
    A('      <contact contype="0" conaffinity="0" selfcollide="none"/>')
    A(f'      <elasticity young="{PEEK_E:.3e}" poisson="{PEEK_NU}" thickness="{PEEK_T:.3e}" elastic2d="bend"/>')
    A('    </flex>')
    A('    <!-- Internal ribs at every span row (the tip rows close the wing ends): edge-length constraints only, no bending (see build_paramotor.py, RIBS). -->')
    plate = rib_elements()
    used = sorted(set(plate))
    local = {g: k for k, g in enumerate(used)}
    A('    <flex name="canopy_ribs" dim="2" radius="0.0005" rgba="0.80 0.62 0.22 0.7"')
    A(f'          body="{" ".join(vname(divmod(g, N_CHORDWISE)[0], divmod(g, N_CHORDWISE)[1]) for g in used)}"')
    A(f'          vertex="{" ".join(["0 0 0"] * len(used))}"')
    A(f'          element="{" ".join(str(local[g]) for g in plate)}">')
    A('      <contact contype="0" conaffinity="0" selfcollide="none"/>')
    A('    </flex>')
    A('  </deformable>')
    A('')
    A('  <equality>')
    A(f'    <flex flex="canopy" solref="{FLEX_SOLREF}"/>')
    A(f'    <flex flex="canopy_ribs" solref="{FLEX_SOLREF}"/>')
    A('  </equality>')
    A('')

    # ---------------- TENDONS ----------------
    A('  <tendon>')
    A('    <!-- ---- suspension lines: UHMWPE, tension-only (upper limit = taut) ---- -->')
    for i in LINE_PANELS:
        side = "l" if panel_phi(i) > 0 else "r"
        rows = [(r, {"A": "f", "B": "r", "C": "c"}[r] + side) for r in station_rows(i)]
        for row, hp in rows:
            name = f"line_{row}{i}"
            rng = tr.get(name, 0.40)
            A(f'    <spatial name="{name}" class="line" range="0 {rng:.6f}">')
            A(f'      <site site="hp_{hp}"/>')
            A(f'      <site site="{vname(*station_vertex(i, row))}"/>')
            A('    </spatial>')
    A('')
    A('    <!-- Brake lines: servo arm tip straight to the trailing-edge tip vertex, two sites. Tension-only, like the suspension: slack below L0, carrying load at L0. Pulling it deflects the deformable trailing edge, as on a real single-skin wing. -->')
    for side in ("L", "R"):
        name = f"brake_{side}"
        rng = tr.get(name, 0.40)
        A(f'    <spatial name="{name}" class="brake" range="0 {rng:.6f}">')
        A(f'      <site site="bl_start_{side}"/>')
        A(f'      <site site="{vname(*BRAKE_VERTEX[side])}"/>')
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
    print(f"sail:      NACA upper surface, section scaled {SECTION_SCALE:.4f} so the surface "
          f"area is {_skin_area(SECTION_SCALE):.5f} m^2; projected chord {SECTION_SCALE*CHORD*1e3:.1f} mm")
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
