#!/usr/bin/env python3
"""
Explicit coefficient-based aerodynamics for the paramotor, after

    J. Umenberger and A. H. Goktogan, "Guidance, Navigation and Control of a
    Small-Scale Paramotor", ACRA 2012 (pap151.pdf), eqs. (8)-(18).

This REPLACES MuJoCo's ellipsoid fluid primitive, which cannot represent the
reference model: it has no C_L0, no C_Lalpha and no rate derivatives.  It is a
different function, not a different parameterisation of the same one.  The
generator therefore emits option density="0" viscosity="0" and no fluidshape
anywhere; if either is re-enabled every force below is counted twice.

FRAMES
------
The paper works in forward-right-down (FRD).  MuJoCo is forward-left-up (FLU).
Rather than transcribe a table of sign-flipped coefficients -- which is exactly
where C_ldelta_a and C_ndelta_a get inverted -- every equation below is written
VERBATIM in the paper's frame and the conversion happens once, at the boundary:

    T = diag(1, -1, -1),    v_frd = T v_flu,    f_flu = T f_frd

T is its own inverse and its own transpose, so one matrix serves both ways.

DEFORMABLE CANOPY
-----------------
The paper welds canopy and fuselage into one rigid body. Here the pod is a free
body and the canopy is a "FAKE HOLLOW" WING: the airfoil-shaped volume a
double-skin wing inflates to, meshed as a solid of point-mass vertices with the
mass distribution of the hollow skins and ribs (build_paramotor.py). No air
pressure is modeled; the stiffness the pressure would give the fabric comes
from edge-length constraints instead.

Lift and drag act on the wing as a whole, once. They are computed on the MEAN
SURFACE: the midpoint of the upper and lower vertices at each chord station (the
leading- and trailing-edge vertices are shared). Each mean-surface grid cell
(four neighbouring points) is one aerodynamic strip element. Its frame comes
from the CURRENT positions: x along the local chord (trailing edge to leading
edge), z along the local surface normal. Its velocity is the mean of its
corners'. Its force is split equally over its four corners, and each corner's
share half to the upper and half to the lower vertex. So twist and brake
deflection change the local incidence directly.

The canopy reference for the pure moments and the lumped mode is the vertices'
mass centre, mean velocity, mass-weighted angular velocity, and the mean
surface's frame (canopy_state()). Pure moments are applied as the force couple
that gives the canopy that moment with zero net force. These references do not
establish the physical canopy's centre of pressure.

LUMPED vs STRIP
---------------
mode="strip" (default) applies eqs. (12)/(15) to every cell, which generates the
roll moment -- damping AND the arc's dihedral effect -- from the geometry
instead of from a coefficient.  See strip_forces_frd().  mode="lumped" is the
paper's own single-force form, applied at the skin's mass centre.

BRAKES
------
Brake servos pull trailing-edge tip vertices through tendons. The deflected
trailing edge changes those cells' incidence; there are no additional virtual
brake-panel aerodynamic forces.

The mesh geometry below is written with elementwise array operations only, so
the same functions serve NumPy here and JAX in mjx/paramotor_mjx.py (GPU matmuls
default to TF32, which would cost ~1e-3 relative accuracy).
"""
import math
import numpy as np

# FLU <-> FRD.  Self-inverse, self-transpose.
T_FLIP = np.diag([1.0, -1.0, -1.0])
FLIP = np.array([1.0, -1.0, -1.0])

_EPS_V = 1e-6          # m/s below which no aerodynamic force is generated


# ============================================================================
# Pure functions.  numpy only, no MuJoCo.  All vectors are FRD body frame.
# ============================================================================

def parafoil_force_frd(v_frd, p):
    """Parafoil lift + drag, eqs. (12) and (15).

    v_frd : velocity of the parafoil mass centre, parafoil frame, FRD.
    Returns (force_frd, alpha_used, alpha_raw, V).

    alpha is CLAMPED to [alpha_min, alpha_max] before it reaches C_L and C_D.
    That clamp is NOT in the paper: eq. (15) is linear with no stall, so at
    C_Lalpha = 2 the model reaches C_L = 1.4 by 30 deg and keeps going.  Both
    the clamped and the raw angle are returned so the caller can flag
    excursions rather than silently flying outside the model's envelope.
    """
    V = float(np.linalg.norm(v_frd))
    if V < _EPS_V:
        return np.zeros(3), 0.0, 0.0, 0.0

    u, _, w = v_frd
    alpha_raw = math.atan2(w, u)
    alpha = min(max(alpha_raw, p["alpha_min"]), p["alpha_max"])

    CL = p["CL0"] + p["CLa"] * alpha
    CD = p["CD0"] + p["CDa"] * alpha * alpha

    # eq. (12): 1/2 rho A ||v|| ( C_L [w, 0, -u] - C_D [v] ).
    # In FRD with u > 0 and w = 0 the lift direction is [0, 0, -u], i.e. -z,
    # i.e. UP.  Getting this sign wrong flies the vehicle inverted.
    lift_dir = np.array([w, 0.0, -u])
    f = 0.5 * p["rho"] * p["AP"] * V * (CL * lift_dir - CD * np.asarray(v_frd))
    return f, alpha, alpha_raw, V


def strip_forces_frd(v_frd, areas, p):
    """Strip theory: eqs. (12)/(15) applied to each spanwise panel separately.

    v_frd : (n, 3) velocity of each panel's aerodynamic centre, in THAT PANEL'S
            OWN frame, FRD.  Because the canopy is arched, every panel sees a
            different local incidence, and that is the whole point.
    areas : (n,) panel planform areas, summing to A^P.

    Returns (forces_frd (n,3), alpha_used (n,), alpha_raw (n,), V (n,)).

    WHAT THIS BUYS over the lumped form.  Three effects that the single-force
    model cannot produce at all, none of which needs a coefficient:

      * roll damping from different panel velocities during rotation.
      * sideslip-induced roll moments from the curved surface. Report beta
        in FRD (positive to the right); a negative derivative measured against
        FLU leftward velocity has the opposite sign in FRD. A static derivative
        alone does not establish stability of the coupled wing/payload system.
      * changes in force distribution when the rigid canopy's attitude changes
        under brake. Local brake-induced deformation is not represented.

    LIMITATION. Each strip uses a whole-wing, AR-corrected C_Lalpha without
    spanwise induced-flow coupling or tip relief. Matching total lift does not
    validate the resulting rate derivatives; C_lp remains an identification
    target. The paper's derivative on a different aircraft is not ground truth
    for this geometry.

    p["strip_cl_scale"] multiplies C_L. ParamotorAero defaults it to the
    provisional arch normalization described in _arch_recovery().
    """
    V = np.linalg.norm(v_frd, axis=1)
    live = V > _EPS_V
    u, w = v_frd[:, 0], v_frd[:, 2]

    alpha_raw = np.arctan2(w, u)
    alpha = np.clip(alpha_raw, p["alpha_min"], p["alpha_max"])
    CL = (p["CL0"] + p["CLa"] * alpha) * p.get("strip_cl_scale", 1.0)
    CD = p["CD0"] + p["CDa"] * alpha * alpha

    lift_dir = np.stack([w, np.zeros_like(w), -u], axis=1)
    q = 0.5 * p["rho"] * areas * V
    f = q[:, None] * (CL[:, None] * lift_dir - CD[:, None] * v_frd)
    f[~live] = 0.0
    return f, alpha, alpha_raw, V


def fuselage_force_frd(v_frd, p):
    """Fuselage drag, eqs. (8) and (10).  The fuselage generates NO lift.

    v_frd : velocity of the fuselage mass centre, body frame, FRD.
    Returns (force_frd, alphaF, V).
    """
    V = float(np.linalg.norm(v_frd))
    if V < _EPS_V:
        return np.zeros(3), 0.0, 0.0
    u, _, w = v_frd
    alphaF = math.atan2(w, u)
    CDF = p["CD0F"] + p["CDaF"] * alphaF * alphaF
    f = -0.5 * p["rho"] * p["AF"] * V * CDF * np.asarray(v_frd)
    return f, alphaF, V


def pure_moments_frd(V_P, omega_frd, alpha, phi, p, roll_native=False):
    """Pure aerodynamic moments about the parafoil mass centre, eq. (18).

    roll_native=True ZEROES the roll row.  In strip mode the roll moment is
    generated by the panel force distribution itself, and applying C_lp and
    C_lphi on top of it would double-count the one effect strip theory exists
    to provide.

    Pitch and yaw rows stay explicit in both modes. The arch's vertical offsets
    also give drag a pitch lever arm and can overlap these derivatives. Their
    total values, including the panel contributions, remain provisional.
    """
    if V_P < _EPS_V:
        return np.zeros(3)
    pr, qr, rr = omega_frd
    b, c = p["b"], p["c"]
    qbar = 0.5 * p["rho"] * p["AP"] * V_P * V_P
    roll = 0.0 if roll_native else (
        p["Clp"] * b * b * pr / (2.0 * V_P) + p["Clphi"] * b * phi)
    return qbar * np.array([
        roll,
        p["Cmq"] * c * c * qr / (2.0 * V_P) + p["Cm0"] * c + p["Cma"] * c * alpha,
        p["Cnr"] * b * b * rr / (2.0 * V_P),
    ])


def euler_phi_from_R_frd(R_frd):
    """Roll angle of an FRD body->world-FRD rotation, 3-2-1 sequence."""
    return math.atan2(R_frd[2, 1], R_frd[2, 2])


# ============================================================================
# Deformable-canopy geometry. Elementwise only; xp is numpy or jax.numpy.
# Vertices are a flat (N, 3) array. Mean-surface grids are (span rows S, chord
# stations C, 3), chord station 0 = leading edge, span row 0 = right tip.
# ============================================================================

def canopy_topology(S, C):
    """Index layout of the fake-hollow canopy: S span rows, C chord stations.

    Flat vertex order, row by row: the upper surface's stations 0..C-1
    (leading edge to trailing edge), then the lower surface's stations 1..C-2.
    The leading- and trailing-edge vertices are shared by both surfaces.

    Returns a dict of int arrays:
      upper, lower  (S, C) flat index of each station. lower[:, 0] and
                    lower[:, C-1] are the shared edge vertices, so
                    0.5 * (X[upper] + X[lower]) is the mean surface.
      upper_tris, lower_tris
                    (T, 3) surface triangles, normals out of the volume.
      rib_tris      every row's section (the ribs a real double skin has).
      tets          (K, 4) solid elements from the leading edge to the last
                    lower-surface station. The trailing-edge bay is left out:
                    it is the brake flap (see build_paramotor.py).
    """
    n_row = 2 * C - 2
    rows = np.arange(S)[:, None] * n_row
    upper = rows + np.arange(C)[None]
    lower = upper.copy()
    lower[:, 1:C - 1] = rows + C + np.arange(C - 2)[None]

    def quads(g, c0, c1, outward_up):
        tris = []
        for s in range(S - 1):
            for c in range(c0, c1):
                a, b, cc, d = g[s, c], g[s + 1, c], g[s, c + 1], g[s + 1, c + 1]
                tris += ([(a, b, cc), (b, d, cc)] if outward_up
                         else [(a, cc, b), (b, cc, d)])
        return tris

    def section(s):
        return [upper[s, c] for c in range(C)] + [lower[s, c] for c in range(C - 2, 0, -1)]

    def fan(poly):
        return [(poly[0], poly[k], poly[k + 1]) for k in range(1, len(poly) - 1)]

    # Every bay of every row pair is cut into triangular prisms (section
    # triangle at row s, extruded to row s+1), and each prism into three tets.
    tets = []
    for s in range(S - 1):
        tri = [(0, 1, (1, "l"))]                       # nose bay
        for c in range(1, C - 2):                      # four-sided bays
            tri += [(c, c + 1, (c + 1, "l")), (c, (c + 1, "l"), (c, "l"))]
        for t in tri:
            ids = [lower[:, k[0]] if isinstance(k, tuple) else upper[:, k] for k in t]
            tets += _prism_tets([g[s] for g in ids] + [g[s + 1] for g in ids])
    out = dict(upper=upper, lower=lower,
               upper_tris=quads(upper, 0, C - 1, True),
               lower_tris=quads(lower, 1, C - 1, False),
               rib_tris=[t for s in range(S) for t in fan(section(s))],
               tets=tets)
    return {k: np.asarray(v, dtype=int) for k, v in out.items()}


# Dompierre et al., "How to subdivide pyramids, prisms and hexahedra into
# tetrahedra" (1999): rotate the prism so its lowest global index is vertex 0,
# then cut each quad face along the diagonal from its lowest index. Neighbouring
# prisms cut a shared face the same way, so the mesh is conforming.
_PRISM_ROT = ((0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3), (2, 0, 1, 5, 3, 4),
              (3, 5, 4, 0, 2, 1), (4, 3, 5, 1, 0, 2), (5, 4, 3, 2, 1, 0))


def _prism_tets(v):
    """Three tets of the prism with bottom (v0, v1, v2) and top (v3, v4, v5)."""
    v = [v[i] for i in _PRISM_ROT[int(np.argmin(v))]]
    if min(v[1], v[5]) < min(v[2], v[4]):
        return [(v[0], v[1], v[2], v[5]), (v[0], v[1], v[5], v[4]), (v[0], v[4], v[5], v[3])]
    return [(v[0], v[1], v[2], v[4]), (v[0], v[4], v[2], v[5]), (v[0], v[4], v[5], v[3])]


def scatter_add(n, idx, values, xp=np):
    """Sum values (len(idx), ...) into n rows by index; repeated indices add."""
    if xp is np:
        out = np.zeros((n,) + values.shape[1:])
        np.add.at(out, idx, values)
        return out
    return xp.zeros((n,) + values.shape[1:], values.dtype).at[idx].add(values)


def mean_surface(X, upper, lower):
    """Mean-surface grid (S, C, ...) from flat per-vertex values X."""
    return 0.5 * (X[upper] + X[lower])


def split_to_skins(F, upper, lower, n, xp=np):
    """Each mean-surface point's force half to its upper, half to its lower vertex."""
    idx = np.concatenate((upper.ravel(), lower.ravel()))
    half = 0.5 * F.reshape(-1, 3)
    return scatter_add(n, idx, xp.concatenate((half, half)), xp)


def triangle_areas(P, tris, xp=np):
    """Vector area of each triangle, along its outward normal."""
    a, b, c = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
    return 0.5 * _cross(b - a, c - a, xp)


def _cross(a, b, xp):
    return xp.stack((a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                     a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                     a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]), axis=-1)


def _dot(a, b, xp):
    return (a * b).sum(axis=-1)


def _unit(a, xp):
    return a / xp.maximum(xp.sqrt(_dot(a, a, xp)), 1e-12)[..., None]


def to_local(R, v, xp=np):
    """R^T v for frames R (..., 3, 3) whose columns are the local axes."""
    return (R * v[..., :, None]).sum(axis=-2)


def to_world(R, v, xp=np):
    """R v."""
    return (R * v[..., None, :]).sum(axis=-1)


def cell_frames(P, xp=np):
    """Frames and areas of the skin cells from vertex positions P (S, C, 3).

    Returns (R (S-1, C-1, 3, 3) with columns x forward, y left, z normal;
    area (S-1, C-1); corner-mean of P (S-1, C-1, 3)).
    """
    a, b, c, d = P[:-1, :-1], P[1:, :-1], P[:-1, 1:], P[1:, 1:]
    fwd = 0.5 * (a + b - c - d)          # trailing edge -> leading edge
    span = 0.5 * (b + d - a - c)         # right -> left
    z = _unit(_cross(fwd, span, xp), xp)
    x = _unit(fwd - _dot(fwd, z, xp)[..., None] * z, xp)
    y = _cross(z, x, xp)
    R = xp.stack((x, y, z), axis=-1)
    diag = _cross(d - a, b - c, xp)
    area = 0.5 * xp.sqrt(_dot(diag, diag, xp))
    return R, area, 0.25 * (a + b + c + d)


def cell_mean(V):
    """Mean of each cell's four corner values, (S-1, C-1, ...)."""
    return 0.25 * (V[:-1, :-1] + V[1:, :-1] + V[:-1, 1:] + V[1:, 1:])


def spread_to_corners(F, xp=np):
    """Split each cell's force equally over its four corners -> (S, C, 3)."""
    q = 0.25 * F
    pad = lambda x, s, c: xp.pad(x, ((s, 1 - s), (c, 1 - c), (0, 0)))
    return pad(q, 0, 0) + pad(q, 1, 0) + pad(q, 0, 1) + pad(q, 1, 1)


def solve3(A, b, xp=np):
    """Solve a 3x3 system by Cramer's rule (elementwise, no matmul)."""
    c0, c1, c2 = A[:, 0], A[:, 1], A[:, 2]
    det = _dot(c0, _cross(c1, c2, xp), xp)
    return xp.stack((_dot(b, _cross(c1, c2, xp), xp),
                     _dot(c0, _cross(b, c2, xp), xp),
                     _dot(c0, _cross(c1, b, xp), xp))) / det


def canopy_state(P, V, mass, R_cells, area, xp=np):
    """Canopy reference state: mass centre, mean velocity, angular velocity, frame.

    P, V, mass are the flat (N, ...) vertex arrays; R_cells and area are the
    mean-surface cells. The angular velocity is the one that gives the
    vertices' actual angular momentum about their mass centre (exact for rigid
    motion). The frame averages the cells' chord and normal directions by area.
    Also returns the inertia tensor about the mass centre and the offsets r of
    every vertex.
    """
    m = mass[..., None]
    total = mass.sum()
    com = (m * P).sum(axis=0) / total
    vbar = (m * V).sum(axis=0) / total
    r = P - com
    rr = _dot(r, r, xp)
    inertia = (mass[..., None, None] * (rr[..., None, None] * xp.eye(3)
               - r[..., :, None] * r[..., None, :])).sum(axis=0)
    momentum = (m * _cross(r, V - vbar, xp)).sum(axis=0)
    omega = solve3(inertia, momentum, xp)
    w = area[..., None]
    z = _unit((w * R_cells[..., :, 2]).sum(axis=(0, 1)), xp)
    x = (w * R_cells[..., :, 0]).sum(axis=(0, 1))
    x = _unit(x - _dot(x, z, xp) * z, xp)
    frame = xp.stack((x, _cross(z, x, xp), z), axis=-1)
    return com, vbar, omega, frame, inertia, r


def couple_forces(moment, inertia, r, mass, xp=np):
    """Vertex forces with zero sum and total moment `moment` about the centre."""
    alpha = solve3(inertia, moment, xp)
    return mass[..., None] * _cross(xp.broadcast_to(alpha, r.shape), r, xp)


def strip_cells(P, V, p, xp=np):
    """Strip forces on every cell, world frame.

    Returns (forces (S-1, C-1, 3), alpha_used, alpha_raw, R, area)."""
    R, area, _ = cell_frames(P, xp)
    v_local = to_local(R, cell_mean(V), xp) * FLIP
    speed = xp.sqrt(_dot(v_local, v_local, xp))
    u, w = v_local[..., 0], v_local[..., 2]
    alpha_raw = xp.arctan2(w, u)
    alpha = xp.clip(alpha_raw, p["alpha_min"], p["alpha_max"])
    cl = (p["CL0"] + p["CLa"] * alpha) * p.get("strip_cl_scale", 1.0)
    cd = p["CD0"] + p["CDa"] * alpha ** 2
    lift = xp.stack((w, xp.zeros_like(w), -u), axis=-1)
    f = (0.5 * p["rho"] * area * speed)[..., None] * (cl[..., None] * lift - cd[..., None] * v_local)
    f = xp.where((speed > _EPS_V)[..., None], f, 0.0)
    return to_world(R, f * FLIP, xp), alpha, alpha_raw, R, area


# ============================================================================
# MuJoCo binding
# ============================================================================

class CanopyMesh:
    """Where the deformable canopy's vertices live in a compiled model.

    Vertex bodies are named cvu_<row>_<station> (upper surface, stations
    0..C-1) and cvl_<row>_<station> (lower surface, stations 1..C-2; the
    leading and trailing edges are the upper surface's) by build_paramotor.py. Each has three world-axis
    slide joints, so its velocity is its three qvel entries and a force on it
    is added straight to those three dofs. Arrays are flat, in the order of
    canopy_topology().
    """

    def __init__(self, model):
        import mujoco
        found = {}
        for b in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if name.startswith(("cvu_", "cvl_")):
                skin, row, station = name.split("_")
                found[skin[2], int(row), int(station)] = b
        if not found:
            raise ValueError("model has no deformable canopy (cvu_*/cvl_* vertex bodies)")
        S = 1 + max(k[1] for k in found)
        C = 1 + max(k[2] for k in found if k[0] == "u")
        order = [("u", s, c) for s in range(S) for c in range(C)]
        order += [("l", s, c) for s in range(S) for c in range(1, C - 1)]
        if set(order) != set(found):
            raise ValueError("canopy vertex names do not form an S x C upper/lower grid")
        topo = canopy_topology(S, C)
        flat = np.empty(len(order), int)
        for k in order:
            skin, s, c = k
            flat[(topo["upper"] if skin == "u" else topo["lower"])[s, c]] = found[k]
        self.bodies = flat
        dof, qpos = np.zeros((len(flat), 3), int), np.zeros((len(flat), 3), int)
        for i, b in enumerate(flat):
            if model.body_dofnum[b] != 3 or model.body_parentid[b] != 0:
                raise ValueError(f"vertex body {b} must be a world child with 3 slides")
            for k in range(3):
                j = model.body_jntadr[b] + k
                if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_SLIDE or model.jnt_axis[j][k] != 1.0:
                    raise ValueError("vertex slides must be world x, y, z in order")
                dof[i, k] = model.jnt_dofadr[j]
                qpos[i, k] = model.jnt_qposadr[j]
        self.dof, self.qpos = dof, qpos
        self.rest = model.body_pos[flat].copy()      # world, at qpos = 0
        self.mass = model.body_mass[flat].copy()
        self.upper, self.lower = topo["upper"], topo["lower"]
        self.topology = topo
        self.shape = (S, C)            # mean-surface grid

    def positions(self, qpos):
        return self.rest + qpos[self.qpos]

    def velocities(self, qvel):
        return qvel[self.dof]

    def mean(self, X):
        return mean_surface(X, self.upper, self.lower)

    def split(self, F):
        return split_to_skins(F, self.upper, self.lower, len(self.bodies))


def set_linear_velocity(model, data, v_world):
    """Give the pod and every canopy vertex the same world linear velocity."""
    import mujoco
    a = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pod_free")]
    data.qvel[a:a + 3] = v_world
    data.qvel[CanopyMesh(model).dof] = v_world


class ParamotorAero:
    """Applies the model above to a compiled paramotor MJCF.

    Usage:
        aero = ParamotorAero(model, paramotor_params.PEEK_1M)   # strip
        mujoco.set_mjcb_passive(aero)

    mode : "strip"  per-cell forces; roll moment is native (default)
           "lumped" one force at the skin's mass centre, roll from C_lp / C_lphi
    """

    def __init__(self, model, params, pod="pod", mode="strip"):
        import mujoco
        self._mj = mujoco
        self.p = dict(params)
        self.mode = mode
        if mode not in ("strip", "lumped"):
            raise ValueError(f"mode must be 'strip' or 'lumped', got {mode!r}")
        self.pod = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, pod)
        if self.pod < 0:
            raise ValueError(f"body not found: {pod!r}")
        self.mesh = CanopyMesh(model)

        ch, sh = math.cos(self.p["chi"]), math.sin(self.p["chi"])
        self.T_BP = np.array([[ch, 0.0, sh], [0.0, 1.0, 0.0], [-sh, 0.0, ch]])

        R, area, _ = cell_frames(self.mesh.mean(self.mesh.rest))
        self.cell_area = area
        self.arch_recovery = self._arch_recovery(R, area)
        self.p.setdefault("strip_cl_scale", self.arch_recovery)
        tot = float(area.sum())
        if abs(tot - self.p["AP"]) / self.p["AP"] > 0.02:
            raise ValueError(
                "mean-surface area is %.5f m^2 but AP = %.5f m^2 (%.1f%% apart). "
                "The parameter set and the geometry disagree."
                % (tot, self.p["AP"], 100 * abs(tot - self.p["AP"]) / self.p["AP"]))

        # World wrenches per body (rows of the vertex bodies and the pod).
        self.wrench = np.zeros((model.nbody, 6))
        self._enabled = True
        self.reset_diagnostics()

        if model.opt.density != 0.0 or model.opt.viscosity != 0.0:
            raise ValueError(
                "model has option density=%g viscosity=%g; MuJoCo's own fluid "
                "model is active and every force would be counted twice. "
                "Rebuild with build_paramotor.py." % (model.opt.density, model.opt.viscosity))

    @staticmethod
    def _arch_recovery(R, area):
        """sum(A_i) / sum(A_i * n_i.zhat) for the mean surface at rest.

        Cell lift follows each cell's normal. This factor normalizes the
        straight-flow vertical lift to the lumped whole-wing coefficient model
        (~1.25 for the arch). The paper supplies whole-wing coefficients, not
        section polars or an arch correction, so this is a provisional
        normalization, fixed at the design shape. Pass strip_cl_scale
        explicitly in the parameter set to override.
        """
        proj = float((area * R[..., 2, 2]).sum() / area.sum())
        if proj < 0.3:
            raise ValueError(f"canopy projects only {proj:.3f} of its area onto "
                             "the horizontal; geometry looks wrong")
        return 1.0 / proj

    # -- diagnostics ---------------------------------------------------------
    def reset_diagnostics(self):
        self.n_calls = 0
        self.n_alpha_clamped = 0
        self.alpha_raw_min = math.inf
        self.alpha_raw_max = -math.inf
        self.last = {}

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, v):
        self._enabled = bool(v)

    def canopy_state(self, data):
        """(com, mean velocity, angular velocity, frame) of the canopy."""
        P = self.mesh.positions(data.qpos)
        V = self.mesh.velocities(data.qvel)
        R, area, _ = cell_frames(self.mesh.mean(P))
        return canopy_state(P, V, self.mesh.mass, R, area)[:4]

    # -- the callback --------------------------------------------------------
    def __call__(self, model, data):
        # Aero goes into qfrc_passive, NOT into xfrc_applied. xfrc_applied
        # belongs to the user: the viewer's Ctrl+drag writes there, and a
        # callback that zeroes it each step silently eats the mouse force.
        # self.wrench keeps the record for tests and telemetry.
        self.wrench[:] = 0.0
        if not self._enabled:
            return
        mesh = self.mesh
        P = mesh.positions(data.qpos)
        V = mesh.velocities(data.qvel)
        f_cells, alpha, alpha_raw, R, area = strip_cells(mesh.mean(P), mesh.mean(V), self.p)
        com, vbar, omega, R_c, inertia, r = canopy_state(P, V, mesh.mass, R, area)
        w_body = T_FLIP @ (R_c.T @ omega)              # canopy frame, FRD

        if self.mode == "strip":
            F = mesh.split(spread_to_corners(f_cells))
            V_P = float(np.linalg.norm(vbar))
            roll_native = True
            self.last["f_cells"] = f_cells
        else:
            v_P = self.T_BP @ (T_FLIP @ (R_c.T @ vbar))
            f_P, a_l, a_raw_l, V_P = parafoil_force_frd(v_P, self.p)
            F_total = R_c @ (T_FLIP @ (self.T_BP.T @ f_P))   # eq. (16)
            F = mesh.mass[..., None] * F_total / mesh.mass.sum()
            alpha, alpha_raw = np.array([a_l]), np.array([a_raw_l])
            area = None
            roll_native = False
            self.last["f_P"] = f_P.copy()

        # Pure moments, as a zero-net-force couple on the skin. In strip mode
        # the roll row is suppressed: the cell distribution already produced it.
        phi = euler_phi_from_R_frd(T_FLIP @ R_c @ T_FLIP)
        a_mean = float(np.average(alpha, weights=area))
        M_P = pure_moments_frd(V_P, w_body, a_mean, phi, self.p, roll_native)
        M_w = R_c @ (T_FLIP @ M_P)
        F = F + couple_forces(M_w, inertia, r, mesh.mass)
        self.wrench[mesh.bodies, :3] = F
        data.qfrc_passive[mesh.dof] += F

        # ---- fuselage ------------------------------------------------------
        vel = np.zeros(6)
        self._mj.mj_objectVelocity(model, data, self._mj.mjtObj.mjOBJ_BODY,
                                   self.pod, vel, 0)
        R_f = data.xmat[self.pod].reshape(3, 3)
        v_F = T_FLIP @ (R_f.T @ vel[3:])
        f_F, alphaF, V_F = fuselage_force_frd(v_F, self.p)
        self.wrench[self.pod, :3] = R_f @ (T_FLIP @ f_F)
        self._mj.mj_applyFT(model, data, self.wrench[self.pod, :3], np.zeros(3),
                            data.xipos[self.pod], self.pod, data.qfrc_passive)

        # ---- diagnostics ---------------------------------------------------
        self.n_calls += 1
        if not np.allclose(alpha, alpha_raw):
            self.n_alpha_clamped += 1
        if V_P >= _EPS_V:
            self.alpha_raw_min = min(self.alpha_raw_min, float(np.min(alpha_raw)))
            self.alpha_raw_max = max(self.alpha_raw_max, float(np.max(alpha_raw)))
        self.last.update(V_P=V_P, alpha=a_mean, alpha_raw=float(np.mean(alpha_raw)),
                         phi=phi, M_P=M_P.copy(), M_world=M_w, f_F=f_F.copy(),
                         V_F=V_F, alphaF=alphaF, com=com, vbar=vbar, omega=omega)

    def envelope_report(self):
        if not self.n_calls:
            return "aero: never called"
        pct = 100.0 * self.n_alpha_clamped / self.n_calls
        return ("aero[%s]: %d steps, alpha_raw in [%+.2f, %+.2f] deg, "
                "clamped on %.1f%% of steps (envelope %+.1f to %+.1f deg)"
                % (self.mode, self.n_calls, math.degrees(self.alpha_raw_min),
                   math.degrees(self.alpha_raw_max), pct,
                   math.degrees(self.p["alpha_min"]), math.degrees(self.p["alpha_max"])))
