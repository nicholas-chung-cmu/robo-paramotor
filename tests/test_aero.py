#!/usr/bin/env python3
"""Tests for the explicit aero layer (paramotor_aero.py).

Covers the canopy and pod aerodynamic layer.
Actuator lags are not implemented.
"""
import math
from functools import lru_cache
from pathlib import Path
import numpy as np
import mujoco

from model import paramotor_aero as A
from model import paramotor_params as PP
from model.paramotor_control import sync_prop

XML = str(Path(__file__).resolve().parents[1] / "model" / "paramotor.xml")
GREEN, RED, DIM, OFF = "\033[32m", "\033[31m", "\033[2m", "\033[0m"
_results = []


class CheckFailure(AssertionError):
    """A failed aero check, visible to pytest and the standalone runner."""


def check(name, ok, detail=""):
    _results.append(ok)
    print(f"  {GREEN+'PASS'+OFF if ok else RED+'FAIL'+OFF}  {name}"
          + (f"\n        {DIM}{detail}{OFF}" if detail else ""))
    if not ok:
        raise CheckFailure(f"{name}: {detail}" if detail else name)


def diagnostic(name, detail=""):
    """Report a measurement without pass/fail. Used where no target is wanted."""
    print(f"  {DIM}INFO{OFF}  {name}" + (f"\n        {DIM}{detail}{OFF}" if detail else ""))


def _run_tests(*tests):
    """Continue after reported check failures so the CLI shows the full suite."""
    for test in tests:
        try:
            test()
        except CheckFailure:
            # check() already recorded and printed this failure.
            pass


def load():
    m = mujoco.MjModel.from_xml_path(XML)
    return m, mujoco.MjData(m)


def set_airspeed(m, d, v_world):
    """Give the pod and every canopy vertex the same world linear velocity."""
    A.set_linear_velocity(m, d, v_world)


def skin_force(aero):
    """Total aerodynamic force on the canopy skin (sum over its vertices)."""
    return aero.wrench[aero.mesh.bodies, :3].sum(axis=(0, 1))


# ---------------------------------------------------------------------------
def test_fluid_model_is_off():
    """3.1 prerequisite: MuJoCo's own fluid model must be disabled or every
    force is counted twice."""
    m, _ = load()
    check("option density == 0", m.opt.density == 0.0, f"density={m.opt.density}")
    check("option viscosity == 0", m.opt.viscosity == 0.0, f"viscosity={m.opt.viscosity}")
    n = sum(1 for g in range(m.ngeom) if m.geom_fluid[g].any())
    check("no geom has fluidshape/fluidcoef", n == 0, f"{n} geoms still carry fluid params")


def test_xfrc_acts_at_com():
    """The binding assumes xfrc_applied acts at the body CoM, not the body
    frame origin.  Verify rather than assume -- it decides whether an offset
    moment is needed."""
    xml = """<mujoco><worldbody><body name="b" pos="1 2 3"><freejoint/>
    <geom type="box" size=".1 .1 .1" mass="2" pos="0.5 0 0"/></body></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    off = np.linalg.norm(d.xipos[1] - d.xpos[1])
    d.xfrc_applied[1, :3] = [0, 0, 1.0]
    mujoco.mj_forward(m, d)
    ang = np.linalg.norm(d.qacc[3:])
    check("xfrc_applied acts at body CoM", ang < 1e-12 and off > 0.1,
          f"CoM offset {off:.3f} m from origin, resulting angular accel {ang:.2e}")


def test_ballistic_without_aero():
    """With the fluid model off and the callback disabled, the vehicle must
    fall at exactly g -- proof that nothing else was quietly making lift."""
    m, d = load()
    mujoco.mj_forward(m, d)
    z0 = d.subtree_com[0][2].copy()
    for _ in range(2000):
        mujoco.mj_step(m, d)
    n, dt = 2000, m.opt.timestep
    # Semi-implicit Euler lags velocity by one step, so the EXACT discrete
    # answer is g dt^2 n(n-1)/2, not g t^2/2.  The 2.45 mm difference over 1 s
    # is the integrator, not a stray force, and asserting the continuous
    # formula just bakes in a tolerance that hides real leaks.
    drop = z0 - d.subtree_com[0][2]
    exact = 9.81 * dt * dt * n * (n - 1) / 2.0
    check("falls ballistically with aero off", abs(drop - exact) < 1e-6,
          f"dropped {drop*1000:.4f} mm, discrete free-fall {exact*1000:.4f} mm "
          f"(continuous g t^2/2 = {0.5*9.81*(n*dt)**2*1000:.4f} mm)")


def test_static_lift():
    """A1: at alpha = 0 and V_P = 6 m/s, L = 1/2 rho A V^2 C_L0, and the force
    is perpendicular to the flow."""
    p = PP.PEEK_1M
    v = np.array([6.0, 0.0, 0.0])
    f, alpha, raw, V = A.parafoil_force_frd(v, p)
    L_exp = 0.5 * p["rho"] * p["AP"] * 36.0 * p["CL0"]
    D_exp = 0.5 * p["rho"] * p["AP"] * 36.0 * p["CD0"]
    check("eq.(12) lift magnitude at alpha=0",
          abs(-f[2] - L_exp) < 1e-9, f"got {-f[2]:.4f} N, expected {L_exp:.4f} N")
    check("eq.(12) drag magnitude at alpha=0",
          abs(-f[0] - D_exp) < 1e-9, f"got {-f[0]:.4f} N, expected {D_exp:.4f} N")
    check("lift points UP in FRD (-z)", f[2] < 0,
          f"f_z = {f[2]:+.4f} N; positive would fly it inverted")


def test_paper_static_lift():
    """The same check on the paper's own aircraft, as a replication anchor."""
    p = PP.PAPER_ACRA2012
    f, *_ = A.parafoil_force_frd(np.array([6.0, 0.0, 0.0]), p)
    L_exp = 0.5 * 1.225 * 1.16 * 36.0 * 0.4
    check("paper aircraft: L = 10.2 N at alpha=0, V=6",
          abs(-f[2] - L_exp) < 1e-9, f"got {-f[2]:.3f} N, expected {L_exp:.3f} N")


def test_alpha_clamp():
    """The stall guard is ours, not the paper's.  It must bite, and it must
    report the raw angle so excursions are visible."""
    p = PP.PEEK_1M
    V = 6.0
    a_big = math.radians(45.0)
    v = np.array([V * math.cos(a_big), 0.0, V * math.sin(a_big)])
    f, alpha, raw, _ = A.parafoil_force_frd(v, p)
    check("alpha clamped at envelope", abs(alpha - p["alpha_max"]) < 1e-12,
          f"raw {math.degrees(raw):.1f} deg -> used {math.degrees(alpha):.1f} deg")
    check("raw alpha reported for flagging", abs(raw - a_big) < 1e-9)
    CL = p["CL0"] + p["CLa"] * p["alpha_max"]
    check("C_L bounded by the clamp", CL < 1.1, f"C_L_max = {CL:.3f}")


def test_moment_signs():
    """eq.(18) is pure damping in p, q, r: each rate must produce an opposing
    moment on its own axis and nothing on the others."""
    p = PP.PEEK_1M
    V = 6.0
    for i, (axis, coef) in enumerate((("roll p", "Clp"), ("pitch q", "Cmq"), ("yaw r", "Cnr"))):
        w = np.zeros(3); w[i] = 1.0
        M = A.pure_moments_frd(V, w, 0.0, 0.0, p)
        M0 = A.pure_moments_frd(V, np.zeros(3), 0.0, 0.0, p)
        dM = M - M0
        opposes = dM[i] < 0
        clean = np.allclose(np.delete(dM, i), 0.0, atol=1e-15)
        check(f"eq.(18) {axis} damping opposes motion ({coef}<0)", opposes and clean,
              f"dM = [{dM[0]:+.3e} {dM[1]:+.3e} {dM[2]:+.3e}] N.m per rad/s")


def test_pitch_stiffness():
    """C_ma < 0 is what makes the wing weathercock in pitch: nose-up alpha must
    give a nose-DOWN moment."""
    p = PP.PEEK_1M
    M_up = A.pure_moments_frd(6.0, np.zeros(3), math.radians(5.0), 0.0, p)
    M_dn = A.pure_moments_frd(6.0, np.zeros(3), math.radians(-5.0), 0.0, p)
    check("eq.(18) C_ma gives static pitch stability", M_up[1] < M_dn[1],
          f"M_y: {M_up[1]:+.5f} at +5 deg vs {M_dn[1]:+.5f} at -5 deg N.m")


def test_frame_roundtrip():
    """T must be its own inverse and its own transpose -- the whole reason the
    equations are kept in FRD and converted once."""
    T = A.T_FLIP
    check("T is self-inverse and self-transpose",
          np.allclose(T @ T, np.eye(3)) and np.allclose(T, T.T))
    v_flu = np.array([1.0, 2.0, 3.0])
    check("FLU->FRD->FLU round trip", np.allclose(T @ (T @ v_flu), v_flu))


def test_double_count_guard():
    """Re-enabling MuJoCo's fluid model must be refused, loudly."""
    m, _ = load()
    m.opt.density = 1.225
    try:
        A.ParamotorAero(m, PP.PEEK_1M)
        check("refuses to run on top of MuJoCo's fluid model", False, "no error raised")
    except ValueError as e:
        check("refuses to run on top of MuJoCo's fluid model", "twice" in str(e))


def test_energy_conservation():
    """Zero density must reproduce the no-aero trajectory and computed energy.

    Native constraints/integration can dissipate energy. Comparing identical
    dynamics isolates aero; asserting conservation of uncomputed zeros did not.
    """
    m, d = load()
    m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    reference = mujoco.MjData(m)
    p = dict(PP.PEEK_1M); p["rho"] = 0.0
    aero = A.ParamotorAero(m, p)
    mujoco.set_mjcb_passive(aero)
    try:
        set_airspeed(m, d, [6.0, 0.3, -0.5])
        reference.qvel[:] = d.qvel
        mujoco.mj_forward(m, d)
        E0 = d.energy.copy()
        check("energy computation is enabled and nonzero", abs(E0.sum()) > 1.0,
              f"initial energy {E0.sum():.6f} J")
        for _ in range(20000):
            aero.enabled = True
            mujoco.mj_step(m, d)
            aero.enabled = False
            mujoco.mj_step(m, reference)
        error = max(np.max(np.abs(d.qpos - reference.qpos)),
                    np.max(np.abs(d.qvel - reference.qvel)),
                    np.max(np.abs(d.energy - reference.energy)))
        check("rho=0 matches disabled aero over 10 s", error < 1e-12,
              f"max state/energy difference {error:.2e}; native energy change "
              f"{d.energy.sum()-E0.sum():+.6f} J")
    finally:
        mujoco.set_mjcb_passive(None)


def test_lift_actually_supports_it():
    """3.4 end to end: held at the design airspeed, the parafoil force must
    come out comparable to weight.  Not a trim test -- a scale test."""
    m, d = load()
    aero = A.ParamotorAero(m, PP.PEEK_1M)
    set_airspeed(m, d, [6.0, 0.0, 0.0])
    mujoco.mj_forward(m, d)
    aero(m, d)
    W = m.body_subtreemass[0] * 9.81
    # The parafoil force lives on the skin's vertices.
    F = skin_force(aero)
    Lz = F[2]
    Dx = -F[0] - aero.wrench[aero.pod, 0]
    check("parafoil lift is the same order as weight at 6 m/s",
          0.5 * W < Lz < 2.0 * W,
          f"lift {Lz:.3f} N vs weight {W:.3f} N ({Lz/W:.2f}x), total drag {Dx:.3f} N, L/D {Lz/Dx:.2f}")
    check("pod now has a drag force at all (was zero before)",
          abs(aero.wrench[aero.pod, 0]) > 1e-4,
          f"fuselage drag {-aero.wrench[aero.pod,0]:.4f} N")


def test_lift_drag_decomposition():
    """Lift is perpendicular to the RELATIVE WIND, not to the body z axis.  At
    a non-zero alpha the FRD z-component of the resultant is not the lift --
    conflating the two is a ~2% error at 5.5 deg and much worse near stall."""
    p = PP.PEEK_1M
    alpha = math.radians(8.0)
    V = 6.0
    v = np.array([V * math.cos(alpha), 0.0, V * math.sin(alpha)])
    f, a_used, _, _ = A.parafoil_force_frd(v, p)
    vhat = v / V
    lhat = np.array([v[2], 0.0, -v[0]]); lhat /= np.linalg.norm(lhat)
    L = float(f @ lhat)
    D = float(-f @ vhat)
    q = 0.5 * p["rho"] * p["AP"] * V * V
    check("lift is perpendicular to the relative wind",
          abs(f @ vhat + D) < 1e-12 and abs(lhat @ vhat) < 1e-15)
    check("L = q A C_L(alpha) resolved perpendicular to flow",
          abs(L - q * (p["CL0"] + p["CLa"] * a_used)) < 1e-9,
          f"L = {L:.4f} N, C_L = {p['CL0']+p['CLa']*a_used:.4f} at {math.degrees(a_used):.1f} deg")
    check("D = q A C_D(alpha) resolved along flow",
          abs(D - q * (p["CD0"] + p["CDa"] * a_used ** 2)) < 1e-9,
          f"D = {D:.4f} N, L/D = {L/D:.2f}")


def test_trim_airspeed():
    """B2: the similarity audit predicts V_trim = 6.656 m/s at C_L = 0.6.
    Check the aero layer reproduces it from L = W, with L resolved correctly."""
    p = PP.PEEK_1M
    m, _ = load()
    W = m.body_subtreemass[0] * 9.81
    CL = 0.6
    V_pred = math.sqrt(2 * W / (p["rho"] * p["AP"] * CL))
    alpha = (CL - p["CL0"]) / p["CLa"]
    v = np.array([V_pred * math.cos(alpha), 0.0, V_pred * math.sin(alpha)])
    f, *_ = A.parafoil_force_frd(v, p)
    lhat = np.array([v[2], 0.0, -v[0]]); lhat /= np.linalg.norm(lhat)
    L = float(f @ lhat)
    check("V_trim at C_L=0.6 matches the similarity audit",
          abs(V_pred - 6.656) < 0.02 and abs(L - W) / W < 1e-6,
          f"V_trim {V_pred:.3f} m/s (audit: 6.656), lift {L:.4f} N vs weight {W:.4f} N")


@lru_cache(maxsize=None)
def _free_flight(thrust, km_kt=None, T=12.0, mode="strip"):
    """Free flight from 6 m/s level.  Returns (speed, climb rate, max |phi|).
    Speeds come from the CoM trajectory -- cvel's linear part is the velocity
    of the point at subtree_com, not the body's own velocity, and misreading
    it makes a healthy glide look like a dive."""
    xml = open(XML).read()
    if km_kt is not None:
        xml = xml.replace('gear="0 0 1 0 0 -0.0105"', f'gear="0 0 1 0 0 {km_kt}"')
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
    aero = A.ParamotorAero(m, PP.PEEK_1M, mode=mode)
    mujoco.set_mjcb_passive(aero)
    try:
        set_airspeed(m, d, [6.0, 0.0, 0.0])
        thr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust")
        sync_prop(m, d, thrust)
        hist = []
        for k in range(int(T / m.opt.timestep)):
            d.ctrl[thr] = thrust
            mujoco.mj_step(m, d)
            if k % 200 == 0:
                Rf = A.T_FLIP @ aero.canopy_state(d)[3] @ A.T_FLIP
                hist.append((d.time, d.subtree_com[0].copy(),
                             abs(math.degrees(A.euler_phi_from_R_frd(Rf)))))
        i = int(len(hist) * 0.6)
        vel = (hist[-1][1] - hist[i][1]) / (hist[-1][0] - hist[i][0])
        return np.linalg.norm(vel), vel[2], max(h[2] for h in hist[i:])
    finally:
        mujoco.set_mjcb_passive(None)


def _probe(omega=(0., 0, 0), vlat=0.0, mode="strip", V=6.0):
    """Hold the skin in a prescribed rigid motion about its mass centre and
    read the total aerodynamic force and moment about that centre."""
    m, d = load()
    aero = A.ParamotorAero(m, PP.PEEK_1M, mode=mode)
    mesh = aero.mesh
    w = np.array(omega, float)
    vcom = np.array([V, vlat, 0.0])
    P = mesh.rest
    c = (mesh.mass[..., None] * P).sum(axis=(0, 1)) / mesh.mass.sum()
    set_airspeed(m, d, vcom)
    d.qvel[mesh.dof] = vcom + np.cross(w, P - c)
    mujoco.mj_forward(m, d)
    aero(m, d)
    f = aero.wrench[mesh.bodies, :3]
    F = f.sum(axis=(0, 1))
    M = np.cross(P - c, f).sum(axis=(0, 1))
    return M, F, aero, m, d


def test_strip_matches_first_principles():
    """The implementation check: recompute every cell force independently from
    the vertex states and compare.  Validates the cell frames, the velocity
    field, both frame conversions and the arch recovery in one shot."""
    p = PP.PEEK_1M
    _, _, aero, m, d = _probe(omega=(0.7, 0.0, 0.0))
    mesh = aero.mesh
    P, Vv = d.xpos[mesh.bodies], d.qvel[mesh.dof]
    S, C = mesh.shape
    worst = 0.0
    for s in range(S - 1):
        for c in range(C - 1):
            k = [(s, c), (s + 1, c), (s, c + 1), (s + 1, c + 1)]
            a, b, cc, dd = (P[i] for i in k)
            fwd = (a + b - cc - dd) / 2
            span = (b + dd - a - cc) / 2
            z = np.cross(fwd, span); z /= np.linalg.norm(z)
            x = fwd - (fwd @ z) * z; x /= np.linalg.norm(x)
            R = np.column_stack((x, np.cross(z, x), z))
            area = 0.5 * np.linalg.norm(np.cross(dd - a, b - cc))
            vp = (R.T @ np.mean([Vv[i] for i in k], axis=0)) * np.array([1.0, -1.0, -1.0])
            V = np.linalg.norm(vp)
            al = np.clip(math.atan2(vp[2], vp[0]), p["alpha_min"], p["alpha_max"])
            CL = (p["CL0"] + p["CLa"] * al) * aero.arch_recovery
            CD = p["CD0"] + p["CDa"] * al * al
            f = 0.5 * p["rho"] * area * V * (CL * np.array([vp[2], 0.0, -vp[0]]) - CD * vp)
            worst = max(worst, np.linalg.norm(
                R @ (f * np.array([1.0, -1.0, -1.0])) - aero.last["f_cells"][s, c]))
    check("strip forces match first principles exactly", worst < 1e-12,
          f"worst cell error {worst:.2e} N over {(S - 1) * (C - 1)} cells")


def test_canopy_reference_state():
    """For rigid motion the skin's reference angular velocity is exact, and its
    velocity is the mass-weighted vertex velocity."""
    w = np.array([0.7, -0.2, 0.3])
    _, _, aero, m, d = _probe(omega=w, vlat=0.4)
    com, vbar, omega, _ = aero.canopy_state(d)
    check("canopy reference velocity and rate match the prescribed motion",
          np.allclose(vbar, [6.0, 0.4, 0.0], atol=1e-12) and np.allclose(omega, w, atol=1e-12),
          f"velocity {vbar}, rate {omega}")


def test_moment_couple_has_no_net_force():
    """Pure moments go on the skin as a couple: zero net force, exact moment."""
    m, d = load()
    aero = A.ParamotorAero(m, PP.PEEK_1M)
    mesh = aero.mesh
    P = mesh.rest
    R, area, _ = A.cell_frames(P)
    _, _, _, _, inertia, r = A.canopy_state(P, np.zeros_like(P), mesh.mass, R, area)
    M = np.array([0.01, -0.02, 0.03])
    f = A.couple_forces(M, inertia, r, mesh.mass)
    check("couple: zero net force, requested moment",
          np.allclose(f.sum(axis=(0, 1)), 0, atol=1e-15)
          and np.allclose(np.cross(r, f).sum(axis=(0, 1)), M, atol=1e-12))


def test_thrust_wrench_reaches_airframe():
    m, d = load()
    sync_prop(m, d, 0.8)
    mujoco.mj_forward(m, d)
    spin = m.joint("prop_spin").dofadr[0]
    pod = m.body("pod").id
    site = m.site("propeller").id
    check("thrust site is attached to the airframe", m.site_bodyid[site] == pod)
    check("thrust reaction does not brake the free rotor",
          abs(d.qfrc_actuator[spin]) < 1e-12,
          f"rotor actuator torque {d.qfrc_actuator[spin]:+.3e} N.m")
    address = m.joint("pod_free").dofadr[0]
    check("airframe receives the commanded thrust and roll reaction",
          abs(d.qfrc_actuator[address] - 0.8) < 1e-12
          and abs(d.qfrc_actuator[address + 3] + 0.8 * 0.0105) < 1e-12)


def test_arch_recovery():
    """Preserve the provisional flat-reference lift normalization."""
    _, _, aero, _, _ = _probe()
    proj = 1.0 / aero.arch_recovery
    check("arch recovery matches PROJ_FRACTION from the generator",
          abs(proj - 0.797) < 0.01,
          f"canopy projects {proj:.4f} of its area (generator: 0.7970), "
          f"recovery factor {aero.arch_recovery:.4f}")
    _, F_s, _, _, _ = _probe(mode="strip")
    _, F_l, _, _, _ = _probe(mode="lumped")
    check("strip and lumped agree on total lift", abs(F_s[2] - F_l[2]) / abs(F_l[2]) < 0.01,
          f"strip {-F_s[2]:.4f} N vs lumped {-F_l[2]:.4f} N")


def test_strip_generates_roll_physics():
    """The point of the exercise: roll damping AND the dihedral effect, both
    from geometry, neither from a coefficient."""
    p = PP.PEEK_1M
    V, b, S, rho = 6.0, p["b"], p["AP"], p["rho"]
    q0 = 0.5 * rho * V * V * S * b
    res = {}
    for mode in ("strip", "lumped"):
        M0, _, _, _, _ = _probe(mode=mode)
        Mp, _, _, _, _ = _probe(omega=(1.0, 0, 0), mode=mode)
        # Positive FRD beta points right; world FLU lateral velocity is negative.
        Mb, _, _, _, _ = _probe(vlat=-V * math.tan(math.radians(1.0)), mode=mode)
        res[mode] = ((Mp - M0)[0] / (q0 * b / (2 * V)),
                     (Mb - M0)[0] / (q0 * math.radians(1.0)),
                     (Mb - M0)[0] / math.radians(1.0))
    Clp_s, Clb_s, dMdb_s = res["strip"]
    Clp_l, Clb_l, _ = res["lumped"]
    check("strip roll damping exceeds the lumped coefficient",
          Clp_s < Clp_l < 0, f"C_lp: strip {Clp_s:+.4f} vs lumped {Clp_l:+.4f} "
          f"({Clp_s/Clp_l:.2f}x)")
    check("arched strips generate FRD sideslip-roll coupling",
          Clb_s > 0.15 and abs(Clb_l) < 1e-9,
          f"C_lbeta: strip {Clb_s:+.4f} vs lumped {Clb_l:+.4f}; "
          f"dM/dbeta = {dMdb_s:+.3f} N.m/rad")
    Q = 1.0 * 0.0105  # prop reaction torque at the 1.0 N thrust clamp
    check("sideslip moment scale exceeds the prop torque scale",
          math.degrees(Q / abs(dMdb_s)) < 3.0,
          f"{math.degrees(Q/abs(dMdb_s)):.2f} deg sideslip balances {Q:.4f} N.m "
          f"(static scale comparison, not a bank-stability proof)")


def test_unpowered_glide():
    V, w, _ = _free_flight(0.0)
    ld = math.sqrt(max(V * V - w * w, 0.0)) / abs(w)
    check("unpowered glide is sane", 4.0 < V < 7.0 and 2.0 < ld < 5.0,
          f"V = {V:.2f} m/s, sink {-w:.2f} m/s, L/D = {ld:.2f} "
          f"(itemized_spec assumes 2-3)")


def test_powered_flight_diagnostic():
    """Powered flight with neutral brakes: REPORTED, not asserted.

    Correctly routed propeller torque turns the vehicle under power, as on a
    real paramotor. Holding a straight line is the controller's job, so these
    runs only record bank and sink; the one hard requirement is that the
    simulation stays finite.
    """
    for thrust, T in ((0.5, 16.0), (1.0, 20.0)):
        V, w, phi = _free_flight(thrust, T=T)
        check(f"{thrust:.1f} N powered flight stays finite",
              all(math.isfinite(x) for x in (V, w, phi)))
        diagnostic(f"{thrust:.1f} N, neutral brakes, {T:.0f} s",
                   f"V {V:.2f} m/s, vertical {w:+.2f} m/s, |phi|max {phi:.1f} deg")


def main():
    _results.clear()
    print("\n\033[1maero layer -- plan sections 3.3 and 3.4\033[0m\n")
    print(" prerequisites (3.1)")
    _run_tests(test_fluid_model_is_off, test_xfrc_acts_at_com, test_ballistic_without_aero)
    print("\n frame handling (3.3)")
    _run_tests(test_frame_roundtrip, test_double_count_guard)
    print("\n parafoil lift and drag (3.4)")
    _run_tests(test_static_lift, test_paper_static_lift, test_alpha_clamp)
    print("\n pure moments (3.4)")
    _run_tests(test_moment_signs, test_pitch_stiffness)
    print("\n end to end")
    _run_tests(test_energy_conservation, test_lift_actually_supports_it,
               test_lift_drag_decomposition, test_trim_airspeed)
    print("\n strip theory (option 3)")
    _run_tests(test_strip_matches_first_principles, test_canopy_reference_state,
               test_moment_couple_has_no_net_force,
               test_thrust_wrench_reaches_airframe, test_arch_recovery,
               test_strip_generates_roll_physics)
    print("\n free flight")
    _run_tests(test_unpowered_glide, test_powered_flight_diagnostic)
    n, tot = sum(_results), len(_results)
    print(f"\n{(GREEN if n==tot else RED)}{n}/{tot} passed{OFF}\n")
    return 0 if n == tot else 1


if __name__ == "__main__":
    raise SystemExit(main())
