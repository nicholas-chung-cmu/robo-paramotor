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

TWO-BODY vs RIGID
-----------------
The paper welds canopy and fuselage into one rigid body and therefore needs the
leverage moments of eq. (17), [S_PB][f_P] and [S_FB][f_F], written out
explicitly.  This model does not: the canopy and the pod are separate free
bodies. The canopy reference is the MASS CENTRE OF ITS WHOLE RIGID SUBTREE:
the massless canopy root's xipos is not the fourteen-panel assembly mass centre.
Strip forces act at panel mass centres; lumped force acts at the assembly mass
centre, following the paper's simplifying force-reference assumption. Fuselage
drag acts at the pod body's mass centre. mj_applyFT supplies the force leverage
through the actual application points; only the pure moments are added explicitly.
These references do not establish the physical canopy's centre of pressure.

LUMPED vs STRIP
---------------
mode="strip" (default) applies eqs. (12)/(15) to each of the fourteen spanwise
panels separately, which generates the roll moment -- damping AND the arc's
dihedral effect -- from the geometry instead of from a coefficient.  See
strip_forces_frd().  mode="lumped" is the paper's own single-force form and is
what the rigid replication of the paper's aircraft must use, since that model
has no panels.

BRAKES
------
Brake servos act mechanically through the suspension and trailing-edge tendons.
There are no additional virtual brake-panel aerodynamic forces.
"""
import math
import numpy as np

# FLU <-> FRD.  Self-inverse, self-transpose.
T_FLIP = np.diag([1.0, -1.0, -1.0])

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
# MuJoCo binding
# ============================================================================

class ParamotorAero:
    """Applies the model above to a compiled paramotor MJCF.

    Usage:
        aero = ParamotorAero(model, paramotor_params.PEEK_1M)   # strip
        mujoco.set_mjcb_passive(aero)

    mode : "strip"  per-panel forces; roll moment is native (default)
           "lumped" one force at the canopy CoM, roll from C_lp / C_lphi
    """

    def __init__(self, model, params, canopy="canopy", pod="pod", mode="strip"):
        import mujoco
        self._mj = mujoco
        self.p = dict(params)
        self.mode = mode
        if mode not in ("strip", "lumped"):
            raise ValueError(f"mode must be 'strip' or 'lumped', got {mode!r}")
        self.canopy = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, canopy)
        self.pod = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, pod)
        if self.canopy < 0 or self.pod < 0:
            raise ValueError(f"body not found: {canopy!r} / {pod!r}")

        ch, sh = math.cos(self.p["chi"]), math.sin(self.p["chi"])
        self.T_BP = np.array([[ch, 0.0, sh], [0.0, 1.0, 0.0], [-sh, 0.0, ch]])

        # ---- panel inventory -------------------------------------------
        ids, areas = [], []
        for b in range(model.nbody):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if not nm.startswith("panel_"):
                continue
            g = [gg for gg in range(model.ngeom) if model.geom_bodyid[gg] == b]
            if not g:
                continue
            ids.append(b)
            areas.append(4.0 * model.geom_size[g[0], 0] * model.geom_size[g[0], 1])
        self.panels = np.array(ids, dtype=int)
        self.panel_area = np.array(areas)

        if mode == "strip":
            if self.panels.size == 0:
                raise ValueError("mode='strip' but the model has no panel_* bodies")
            # The fast path computes panel velocities from the canopy's rigid
            # body state.  That is EXACT only while the panels carry no joints
            # of their own -- which N_CHORD > 1 would change.
            free = [int(b) for b in self.panels if model.body_dofnum[b] != 0]
            if free:
                raise ValueError(
                    "panels %s have their own DOFs; the rigid fast path is no "
                    "longer exact. Use mj_objectVelocity per panel." % free)
            self.arch_recovery = self._arch_recovery(model)
            self.p.setdefault("strip_cl_scale", self.arch_recovery)
            tot = self.panel_area.sum()
            if abs(tot - self.p["AP"]) / self.p["AP"] > 0.02:
                raise ValueError(
                    "panel areas sum to %.5f m^2 but AP = %.5f m^2 (%.1f%% apart). "
                    "The parameter set and the geometry disagree."
                    % (tot, self.p["AP"], 100 * abs(tot - self.p["AP"]) / self.p["AP"]))

        # World wrenches at panel/pod CoMs or the canopy assembly CoM.
        self.wrench = np.zeros((model.nbody, 6))
        self._enabled = True
        self.reset_diagnostics()

        if model.opt.density != 0.0 or model.opt.viscosity != 0.0:
            raise ValueError(
                "model has option density=%g viscosity=%g; MuJoCo's own fluid "
                "model is active and every force would be counted twice. "
                "Rebuild with build_paramotor.py." % (model.opt.density, model.opt.viscosity))

    def _arch_recovery(self, model):
        """sum(A_i) / sum(A_i * n_i.zhat) for the canopy at its rest pose.

        Panel lift follows each panel's normal. This factor normalizes the
        straight-flow vertical lift to the lumped whole-wing coefficient model.
        It preserves the existing ~1.253 lift multiplier. The paper supplies
        whole-wing simulation coefficients, not measured section polars or an
        arch correction, so this is a provisional normalization convention.

        Computed at the rest pose, which is valid because the canopy is rigid.
        Pass strip_cl_scale explicitly in the parameter set to override.
        """
        import mujoco
        d = mujoco.MjData(model)
        mujoco.mj_forward(model, d)
        n_z = np.array([d.xmat[b].reshape(3, 3)[2, 2] for b in self.panels])
        proj = float((self.panel_area * n_z).sum() / self.panel_area.sum())
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

    # -- state extraction ----------------------------------------------------
    def _canopy_state(self, model, data):
        """World angular velocity and velocity at the rigid assembly CoM."""
        vel = np.zeros(6)
        self._mj.mj_objectVelocity(model, data, self._mj.mjtObj.mjOBJ_BODY,
                                   self.canopy, vel, 0)
        # mj_objectVelocity reports at xipos, which is only a body reference
        # for this massless root. Translate to the actual panel-assembly CoM.
        offset = data.subtree_com[self.canopy] - data.xipos[self.canopy]
        v_com = vel[3:] + np.cross(vel[:3], offset)
        return vel[:3].copy(), v_com, data.xmat[self.canopy].reshape(3, 3)

    # -- the callback --------------------------------------------------------
    def __call__(self, model, data):
        # Aero goes into qfrc_passive via mj_applyFT, NOT into xfrc_applied.
        # xfrc_applied belongs to the user: the viewer's Ctrl+drag writes there,
        # and a callback that zeroes it each step silently eats the mouse force
        # on exactly the bodies you want to drag.  mj_applyFT is documented as
        # "outside the xfrc_applied mechanism" and is the right channel for a
        # passive force.  self.wrench keeps the record for tests and telemetry.
        self.wrench[:] = 0.0
        if not self._enabled:
            return

        omega_w, v_com_w, R_c = self._canopy_state(model, data)
        w_body = T_FLIP @ (R_c.T @ omega_w)          # canopy frame, FRD

        if self.mode == "strip":
            V_P, alpha, alpha_raw = self._apply_strip(data, omega_w, v_com_w)
            roll_native = True
        else:
            V_P, alpha, alpha_raw = self._apply_lumped(data, v_com_w, R_c)
            roll_native = False

        # Pure moments go on the canopy in both modes.  In strip mode the roll
        # row is suppressed: the panel distribution already produced it.
        phi = euler_phi_from_R_frd(T_FLIP @ R_c @ T_FLIP)
        # C_ma needs ONE incidence.  In strip mode take the area-weighted mean
        # of the panel incidences: that is the lumped equivalent of the
        # distribution, and it reduces to the single value in lumped mode.
        a_mean = float(np.average(alpha, weights=(self.panel_area
                                                  if self.mode == "strip" else None)))
        M_P = pure_moments_frd(V_P, w_body, a_mean, phi, self.p, roll_native)
        self.wrench[self.canopy, 3:] += R_c @ (T_FLIP @ M_P)

        # ---- fuselage ------------------------------------------------------
        vel = np.zeros(6)
        self._mj.mj_objectVelocity(model, data, self._mj.mjtObj.mjOBJ_BODY,
                                   self.pod, vel, 0)
        R_f = data.xmat[self.pod].reshape(3, 3)
        v_F = T_FLIP @ (R_f.T @ vel[3:])
        f_F, alphaF, V_F = fuselage_force_frd(v_F, self.p)
        self.wrench[self.pod, :3] = R_f @ (T_FLIP @ f_F)

        # ---- hand the accumulated wrenches to MuJoCo -----------------------
        for bid in np.flatnonzero(np.abs(self.wrench).sum(axis=1) > 0.0):
            point = (data.subtree_com[self.canopy] if bid == self.canopy
                     else data.xipos[bid])
            self._mj.mj_applyFT(model, data,
                                self.wrench[bid, :3], self.wrench[bid, 3:],
                                point, int(bid), data.qfrc_passive)

        # ---- diagnostics ---------------------------------------------------
        self.n_calls += 1
        raw_hi = float(np.max(alpha_raw)) if np.size(alpha_raw) else 0.0
        raw_lo = float(np.min(alpha_raw)) if np.size(alpha_raw) else 0.0
        if not np.allclose(alpha, alpha_raw):
            self.n_alpha_clamped += 1
        if V_P >= _EPS_V:
            self.alpha_raw_min = min(self.alpha_raw_min, raw_lo)
            self.alpha_raw_max = max(self.alpha_raw_max, raw_hi)
        self.last.update(V_P=V_P, alpha=a_mean,
                         alpha_raw=float(np.mean(alpha_raw)), phi=phi,
                         M_P=M_P.copy(), f_F=f_F.copy(), V_F=V_F, alphaF=alphaF)

    # -- per-panel strip theory ---------------------------------------------
    def _apply_strip(self, data, omega_w, v_com_w):
        ids = self.panels
        R_p = data.xmat[ids].reshape(-1, 3, 3)               # panel -> world
        r_w = data.xipos[ids] - data.subtree_com[self.canopy]

        # Rigid body: every panel's CoM velocity follows from the canopy's.
        v_w = v_com_w[None, :] + np.cross(omega_w[None, :], r_w)

        # world FLU -> panel FLU -> panel FRD
        v_panel = np.einsum('nji,nj->ni', R_p, v_w) * np.array([1.0, -1.0, -1.0])

        f_frd, alpha, alpha_raw, V = strip_forces_frd(v_panel, self.panel_area, self.p)

        # panel FRD -> panel FLU -> world
        f_w = np.einsum('nij,nj->ni', R_p, f_frd * np.array([1.0, -1.0, -1.0]))
        self.wrench[ids, :3] = f_w

        self.last["f_panel"] = f_w
        self.last["V_panel"] = V
        # A single reference airspeed for the pure-moment terms: the canopy CoM.
        V_ref = float(np.linalg.norm(v_com_w))
        return V_ref, alpha, alpha_raw

    # -- the paper's single lumped force ------------------------------------
    def _apply_lumped(self, data, v_com_w, R_c):
        v_body = T_FLIP @ (R_c.T @ v_com_w)
        v_P = self.T_BP @ v_body
        f_P, alpha, alpha_raw, V_P = parafoil_force_frd(v_P, self.p)
        f_body = self.T_BP.T @ f_P               # eq. (16)
        self.wrench[self.canopy, :3] = R_c @ (T_FLIP @ f_body)
        self.last["f_P"] = f_P.copy()
        return V_P, np.array([alpha]), np.array([alpha_raw])

    def envelope_report(self):
        if not self.n_calls:
            return "aero: never called"
        pct = 100.0 * self.n_alpha_clamped / self.n_calls
        return ("aero[%s]: %d steps, alpha_raw in [%+.2f, %+.2f] deg, "
                "clamped on %.1f%% of steps (envelope %+.1f to %+.1f deg)"
                % (self.mode, self.n_calls, math.degrees(self.alpha_raw_min),
                   math.degrees(self.alpha_raw_max), pct,
                   math.degrees(self.p["alpha_min"]), math.degrees(self.p["alpha_max"])))
