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
body and the canopy is a DEFORMABLE SKIN: a MuJoCo flex shell over a grid of
point-mass vertices (build_paramotor.py). The skin bends, twists and cambers
under load; nothing about its shape is fixed.

Each grid cell (four neighbouring vertices) from the leading edge aft is one
aerodynamic strip element. Stations ahead of the leading-edge station (the sail
wrapped under the nose) carry no aerodynamic force: they face backwards, where
a strip element's angle of attack means nothing.
Its frame comes from the CURRENT vertex positions: x along the local chord
(trailing edge to leading edge), z along the local surface normal. Its velocity
is the mean of its corners' velocities. Its force is split equally over its
four corners. So camber, twist and brake deflection all change the local
incidence directly.

The canopy reference for the pure moments and the lumped mode is the skin's
mass centre, mean velocity, mass-weighted angular velocity, and mean frame
(canopy_state()). Pure moments are applied as the force couple that gives the
skin that moment with zero net force. These references do not establish the
physical canopy's centre of pressure.

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
# Grids are (span rows S, chord stations C, 3), chord station 0 = leading edge,
# span row 0 = right tip.
# ============================================================================

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
    """Skin reference state: mass centre, mean velocity, angular velocity, frame.

    The angular velocity is the one that gives the skin's actual angular
    momentum about its mass centre (exact for rigid motion). The frame averages
    the cells' chord and normal directions by area. Also returns the inertia
    tensor about the mass centre and the offsets r of every vertex.
    """
    m = mass[..., None]
    total = mass.sum()
    com = (m * P).sum(axis=(0, 1)) / total
    vbar = (m * V).sum(axis=(0, 1)) / total
    r = P - com
    rr = _dot(r, r, xp)
    inertia = (mass[..., None, None] * (rr[..., None, None] * xp.eye(3)
               - r[..., :, None] * r[..., None, :])).sum(axis=(0, 1))
    momentum = (m * _cross(r, V - vbar, xp)).sum(axis=(0, 1))
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


def stall_blend(alpha_raw, p, xp=np):
    """Weight of the post-stall (flat-plate) model, 0..1.

    Exactly 0 inside [alpha_min, alpha_max], so the linear model there is
    unchanged; it rises smoothly (smoothstep) to 1 over stall_width beyond
    either limit.
    """
    width = p["stall_width"]
    t = xp.maximum(alpha_raw - p["alpha_max"], p["alpha_min"] - alpha_raw) / width
    t = xp.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def strip_cells(P, V, p, xp=np):
    """Strip forces on every cell, world frame.

    P, V are the LIFTING grid (leading edge first, trailing edge last). Each
    spanwise strip (between neighbouring rows) is one airfoil section: its
    angle of attack is measured from the SECTION CHORD LINE (leading edge to
    trailing edge) with the strip's area-weighted mean velocity, as for a real
    airfoil, not from each curved surface panel's own tilt. The strip's lift
    and drag are shared over its cells by area. Deflecting the trailing edge
    (brakes) or collapsing the section rotates or shortens the chord line and
    so changes the strip's incidence directly.

    Inside [alpha_min, alpha_max] the linear coefficients of eq. (15) apply.
    Beyond it they blend into a flat plate (C_L = sin 2a, C_D = C_D0 +
    2 sin^2 a) over stall_width: lift falls and drag rises past stall. The
    linear part is evaluated at the clipped angle so it never extrapolates.

    Returns (forces (S-1, C-1, 3), alpha_used, alpha_raw, R, area), with the
    per-strip angles and section frames broadcast to every cell; alpha_used is
    the clipped angle (used by the pure moments and diagnostics)."""
    _, area, _ = cell_frames(P, xp)
    chord = xp.stack((P[:, 0], P[:, -1]), axis=1)          # (S, 2, 3) LE, TE
    R_sec = cell_frames(chord, xp)[0][:, 0]                # (S-1, 3, 3)
    weight = area / xp.maximum(area.sum(axis=1, keepdims=True), 1e-12)
    v_strip = (weight[..., None] * cell_mean(V)).sum(axis=1)
    v_local = to_local(R_sec, v_strip, xp) * FLIP
    speed = xp.sqrt(_dot(v_local, v_local, xp))
    u, w = v_local[..., 0], v_local[..., 2]
    alpha_raw = xp.arctan2(w, u)
    alpha = xp.clip(alpha_raw, p["alpha_min"], p["alpha_max"])
    s = stall_blend(alpha_raw, p, xp)
    sin_a, cos_a = xp.sin(alpha_raw), xp.cos(alpha_raw)
    cl = ((1 - s) * (p["CL0"] + p["CLa"] * alpha) * p.get("strip_cl_scale", 1.0)
          + s * 2.0 * sin_a * cos_a)
    cd = (1 - s) * (p["CD0"] + p["CDa"] * alpha ** 2) + s * (p["CD0"] + 2.0 * sin_a ** 2)
    lift = xp.stack((w, xp.zeros_like(w), -u), axis=-1)
    f = (0.5 * p["rho"] * area.sum(axis=1) * speed)[..., None] * (
        cl[..., None] * lift - cd[..., None] * v_local)
    f = xp.where((speed > _EPS_V)[..., None], f, 0.0)
    forces = weight[..., None] * to_world(R_sec, f * FLIP, xp)[:, None, :]
    shape = area.shape
    return (forces, xp.broadcast_to(alpha[:, None], shape),
            xp.broadcast_to(alpha_raw[:, None], shape),
            xp.broadcast_to(R_sec[:, None], shape + (3, 3)), area)


# ============================================================================
# MuJoCo binding
# ============================================================================

class CanopyMesh:
    """Where the deformable canopy's vertices live in a compiled model.

    Vertex bodies are named cv_<row>_<station> by build_paramotor.py; each has
    three world-axis slide joints, so its velocity is its three qvel entries
    and a force on it is added straight to those three dofs.
    """

    def __init__(self, model):
        import mujoco
        # Grid from the vertex SITES: where the skin closes (wingtips), one
        # body carries the sites of two grid points.
        found = {}
        for site in range(model.nsite):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site) or ""
            if name.startswith("cv_"):
                _, row, station = name.split("_")
                found[int(row), int(station)] = model.site_bodyid[site]
        if not found:
            raise ValueError("model has no deformable canopy (cv_* vertex bodies)")
        S = 1 + max(k[0] for k in found)
        C = 1 + max(k[1] for k in found)
        self.bodies = np.array([[found[s, c] for c in range(C)] for s in range(S)])
        dof, qpos = np.zeros((S, C, 3), int), np.zeros((S, C, 3), int)
        for (s, c), b in found.items():
            if model.body_dofnum[b] != 3 or model.body_parentid[b] != 0:
                raise ValueError(f"vertex body {b} must be a world child with 3 slides")
            for k in range(3):
                j = model.body_jntadr[b] + k
                if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_SLIDE or model.jnt_axis[j][k] != 1.0:
                    raise ValueError("vertex slides must be world x, y, z in order")
                dof[s, c, k] = model.jnt_dofadr[j]
                qpos[s, c, k] = model.jnt_qposadr[j]
        self.dof, self.qpos = dof, qpos
        self.rest = model.body_pos[self.bodies].copy()      # world, at qpos = 0
        # Each body's mass counts once; repeated grid points carry none.
        self.mass = model.body_mass[self.bodies].copy()
        first = {}
        for idx in np.ndindex(S, C):
            if first.setdefault(self.bodies[idx], idx) != idx:
                self.mass[idx] = 0.0
        self.shape = (S, C)
        # First lifting chord station: the leading edge (0 without a nose wrap).
        k = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, "canopy_le_station")
        self.le = int(model.numeric_data[model.numeric_adr[k]]) if k >= 0 else 0

    def positions(self, qpos):
        return self.rest + qpos[self.qpos]

    def velocities(self, qvel):
        return qvel[self.dof]

    def lifting(self, X):
        """The leading-edge-aft part of a (S, C, ...) grid."""
        return X[:, self.le:]


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

        R, area, _ = cell_frames(self.mesh.lifting(self.mesh.rest))
        self.cell_area = area
        self.arch_recovery = self._arch_recovery(R, area)
        self.p.setdefault("strip_cl_scale", self.arch_recovery)
        tot = float(area.sum())
        if abs(tot - self.p["AP"]) / self.p["AP"] > 0.02:
            raise ValueError(
                "skin area is %.5f m^2 but AP = %.5f m^2 (%.1f%% apart). "
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
        """sum(A_i) / sum(A_i * n_i.zhat) for the skin at its rest shape.

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
        """(com, mean velocity, angular velocity, frame) of the skin."""
        P = self.mesh.positions(data.qpos)
        V = self.mesh.velocities(data.qvel)
        R, area, _ = cell_frames(self.mesh.lifting(P))
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
        f_cells, alpha, alpha_raw, R, area = strip_cells(mesh.lifting(P), mesh.lifting(V), self.p)
        com, vbar, omega, R_c, inertia, r = canopy_state(P, V, mesh.mass, R, area)
        w_body = T_FLIP @ (R_c.T @ omega)              # canopy frame, FRD

        if self.mode == "strip":
            F = np.pad(spread_to_corners(f_cells), ((0, 0), (mesh.le, 0), (0, 0)))
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
        # Add, not assign: a merged tip vertex appears twice in the grid.
        np.add.at(self.wrench, (mesh.bodies, slice(0, 3)), F)
        np.add.at(data.qfrc_passive, mesh.dof, F)

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
