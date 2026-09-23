#!/usr/bin/env python3
"""Tests for the explicit aero layer (paramotor_aero.py).

Covers plan sections 3.3 and 3.4 only.  Brakes (3.5) and actuator lags (3.6)
are deliberately not implemented and are not tested here.
"""
import math
import numpy as np
import mujoco

import paramotor_aero as A
import paramotor_params as PP

XML = "paramotor.xml"
GREEN, RED, DIM, OFF = "\033[32m", "\033[31m", "\033[2m", "\033[0m"
_results = []


def check(name, ok, detail=""):
    _results.append(ok)
    print(f"  {GREEN+'PASS'+OFF if ok else RED+'FAIL'+OFF}  {name}"
          + (f"\n        {DIM}{detail}{OFF}" if detail else ""))


def load():
    m = mujoco.MjModel.from_xml_path(XML)
    return m, mujoco.MjData(m)


def set_airspeed(m, d, v_world):
    """Give every free body the same world linear velocity.

    Addresses are looked up, never assumed: the canopy freejoint starts at
    dof 9, not 6, because prop_spin and the two servo arms sit between the two
    freejoints.  Writing qvel[6:9] silently spins the propeller instead.
    """
    for name in ("pod_free", "canopy_free"):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        a = m.jnt_dofadr[j]
        d.qvel[a:a + 3] = v_world


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
    """B1: with rho = 0 the callback must do no work at all.  Catches any
    spurious moment or a force applied at the wrong point."""
    m, d = load()
    p = dict(PP.PEEK_1M); p["rho"] = 0.0
    aero = A.ParamotorAero(m, p)
    mujoco.set_mjcb_passive(aero)
    try:
        set_airspeed(m, d, [6.0, 0.3, -0.5])
        mujoco.mj_forward(m, d)
        E0 = d.energy.copy()
        for _ in range(20000):           # 10 s
            mujoco.mj_step(m, d)
        drift = abs((d.energy.sum() - E0.sum()) / max(abs(E0.sum()), 1e-9))
        check("no spurious work at rho=0 (10 s)", drift < 1e-6,
              f"relative energy drift {drift:.2e}")
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
    # In strip mode the parafoil force lives on the PANEL bodies, not on the
    # canopy body, so it has to be summed over the panels.
    F = aero.wrench[aero.canopy, :3] + aero.wrench[aero.panels, :3].sum(axis=0)
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
        for nm in ("pod_free", "canopy_free"):
            a = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)]
            d.qvel[a:a + 3] = [6.0, 0.0, 0.0]
        thr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust")
        hist = []
        for k in range(int(T / m.opt.timestep)):
            d.ctrl[thr] = thrust
            mujoco.mj_step(m, d)
            if k % 200 == 0:
                Rf = A.T_FLIP @ d.xmat[aero.canopy].reshape(3, 3) @ A.T_FLIP
                hist.append((d.time, d.subtree_com[0].copy(),
                             abs(math.degrees(A.euler_phi_from_R_frd(Rf)))))
        i = int(len(hist) * 0.6)
        vel = (hist[-1][1] - hist[i][1]) / (hist[-1][0] - hist[i][0])
        return np.linalg.norm(vel), vel[2], max(h[2] for h in hist[i:])
    finally:
        mujoco.set_mjcb_passive(None)


def _probe(omega=(0., 0, 0), vlat=0.0, mode="strip", V=6.0):
    """Hold the canopy in a prescribed motion and read the total aerodynamic
    wrench about its CoM.  MuJoCo's freejoint linear qvel is the velocity of
    the body FRAME ORIGIN, not the CoM, so a pure rotation about the CoM needs
    the linear term corrected -- otherwise the 0.089 m offset between the two
    inflates C_lp by 30%."""
    m, d = load()
    aero = A.ParamotorAero(m, PP.PEEK_1M, mode=mode)
    mujoco.mj_forward(m, d)
    w = np.array(omega, float)
    vcom = np.array([V, vlat, 0.0])
    jc = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "canopy_free")]
    jp = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "pod_free")]
    d.qvel[jc:jc + 3] = vcom - np.cross(w, d.xipos[aero.canopy] - d.xpos[aero.canopy])
    d.qvel[jc + 3:jc + 6] = w
    d.qvel[jp:jp + 3] = vcom
    mujoco.mj_forward(m, d)
    aero(m, d)
    M = aero.wrench[aero.canopy, 3:].copy()
    F = aero.wrench[aero.canopy, :3].copy()
    c = d.xipos[aero.canopy]
    for i in aero.panels:
        f = aero.wrench[i, :3]
        F += f
        M += np.cross(d.xipos[i] - c, f) + aero.wrench[i, 3:]
    return M, F, aero, m, d


def test_strip_matches_first_principles():
    """The implementation check: recompute every panel force independently from
    the model state and compare.  Validates the velocity field, both frame
    conversions and the arch recovery in one shot."""
    p = PP.PEEK_1M
    _, _, aero, m, d = _probe(omega=(0.7, 0.0, 0.0))
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, aero.canopy, vel, 0)
    worst = 0.0
    for k, i in enumerate(aero.panels):
        Rp = d.xmat[i].reshape(3, 3)
        v_w = vel[3:] + np.cross(vel[:3], d.xipos[i] - d.xipos[aero.canopy])
        vp = (Rp.T @ v_w) * np.array([1.0, -1.0, -1.0])
        V = np.linalg.norm(vp)
        al = np.clip(math.atan2(vp[2], vp[0]), p["alpha_min"], p["alpha_max"])
        CL = (p["CL0"] + p["CLa"] * al) * aero.arch_recovery
        CD = p["CD0"] + p["CDa"] * al * al
        f = 0.5 * p["rho"] * aero.panel_area[k] * V * (
            CL * np.array([vp[2], 0.0, -vp[0]]) - CD * vp)
        worst = max(worst, np.linalg.norm(
            Rp @ (f * np.array([1.0, -1.0, -1.0])) - aero.wrench[i, :3]))
    check("strip forces match first principles exactly", worst < 1e-12,
          f"worst panel error {worst:.2e} N over {len(aero.panels)} panels")


def test_arch_recovery():
    """An arched canopy lifts on its PROJECTED area.  Strip theory reproduces
    that from geometry -- but the paper's C_L0 is referenced to FLAT area on a
    wing that was already arched, so without compensation the loss is applied
    twice and 20% of the lift disappears."""
    _, _, aero, _, _ = _probe()
    proj = 1.0 / aero.arch_recovery
    check("arch recovery matches PROJ_FRACTION from the generator",
          abs(proj - 0.797) < 0.005,
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
        Mb, _, _, _, _ = _probe(vlat=V * math.tan(math.radians(1.0)), mode=mode)
        res[mode] = ((Mp - M0)[0] / (q0 * b / (2 * V)),
                     (Mb - M0)[0] / (q0 * math.radians(1.0)),
                     (Mb - M0)[0] / math.radians(1.0))
    Clp_s, Clb_s, dMdb_s = res["strip"]
    Clp_l, Clb_l, _ = res["lumped"]
    check("strip roll damping exceeds the lumped coefficient",
          Clp_s < Clp_l < 0, f"C_lp: strip {Clp_s:+.4f} vs lumped {Clp_l:+.4f} "
          f"({Clp_s/Clp_l:.2f}x); analytic strip prediction -0.213")
    check("strip generates the dihedral effect; lumped generates none",
          Clb_s < -0.15 and abs(Clb_l) < 1e-9,
          f"C_lbeta: strip {Clb_s:+.4f} vs lumped {Clb_l:+.4f}; "
          f"dM/dbeta = {dMdb_s:+.3f} N.m/rad")
    Q = 1.5 * 0.0105
    check("dihedral bounds the bank against propeller torque",
          math.degrees(Q / abs(dMdb_s)) < 3.0,
          f"{math.degrees(Q/abs(dMdb_s)):.2f} deg sideslip balances {Q:.4f} N.m "
          f"(lumped: unbounded, no restoring term at all)")


def test_unpowered_glide():
    V, w, _ = _free_flight(0.0)
    ld = math.sqrt(max(V * V - w * w, 0.0)) / abs(w)
    check("unpowered glide is sane", 4.0 < V < 7.0 and 2.0 < ld < 5.0,
          f"V = {V:.2f} m/s, sink {-w:.2f} m/s, L/D = {ld:.2f} "
          f"(itemized_spec assumes 2-3)")


def test_strip_fixes_the_spiral():
    """The headline result.  At 0.5 N the lumped model rolls off to 63 deg and
    dives; strip theory holds it level, with the propeller torque balanced by
    sideslip instead of integrating into bank."""
    V_l, w_l, phi_l = _free_flight(0.5, T=16.0, mode="lumped")
    V_s, w_s, phi_s = _free_flight(0.5, T=16.0, mode="strip")
    check("strip mode removes the powered spiral", phi_l > 25.0 and phi_s < 5.0,
          f"0.5 N over 16 s: lumped |phi|max {phi_l:.0f} deg ({w_l:+.2f} m/s) -> "
          f"strip |phi|max {phi_s:.1f} deg ({w_s:+.2f} m/s)")


def test_powered_climb():
    """1.0 N buys climb with the propeller torque left ON -- the lumped model
    needed the torque deleted to achieve this."""
    V, w, phi = _free_flight(1.0, T=20.0)
    check("1.0 N thrust climbs with prop torque intact", w > -0.2 and phi < 10.0,
          f"{w:+.2f} m/s at {V:.2f} m/s, |phi|max {phi:.1f} deg")


def test_validated_thrust_envelope():
    """Pins the UPPER EDGE of the validated envelope, so it cannot drift
    unnoticed.  Not a defect: the spiral is gone across the thrust range the
    vehicle is meant to fly, and departure beyond it is accepted.

    Above roughly 1.0 N (T/W = 0.31) the vehicle pitches up, the tension-only
    suspension goes fully slack and the canopy tumbles; alpha reaches +-180 deg,
    far outside the +-8/+18 deg envelope where a linear no-stall C_L means
    anything.  Established NOT to be:
      * a thrust-line offset -- moving the prop from z=0.010 to the CG at
        z=0.126 does not help;
      * a step-input artifact -- ramping with the section 3.6 motor lag
        (tau = 0.45 s) does not help.
    It needs a stall model and a rigging re-trim, both existing plan items.
    Note the lumped model was ALREADY failing above 0.5 N, as a dive rather
    than a tumble, so strip theory did not introduce this -- it widened the
    usable thrust range from about 0 to about 1.0 N.
    """
    _, _, phi_lo = _free_flight(1.0, T=20.0)
    _, _, phi_hi = _free_flight(1.8, T=16.0)
    check("validated envelope is 0 to ~1.0 N thrust", phi_lo < 10.0 and phi_hi > 30.0,
          f"1.0 N -> |phi|max {phi_lo:.1f} deg (good); 1.8 N -> {phi_hi:.0f} deg "
          f"(departs; needs stall + rigging trim)")


def test_brakes_are_stubbed():
    """3.5 is explicitly out of scope: the hook must exist and return zero."""
    f, M = A.brake_wrench_frd(6.0, np.array([6.0, 0, 0]), 0.5, 0.2, PP.PEEK_1M)
    check("brake hook present and returns zero (3.5 not yet done)",
          np.allclose(f, 0) and np.allclose(M, 0))
    keys = set(PP.PEEK_1M)
    check("no brake coefficients smuggled into the parameter sets",
          not any(k.startswith(("Cl_d", "Cn_d", "CLd", "CDd")) or k == "d" for k in keys))


if __name__ == "__main__":
    print("\n\033[1maero layer -- plan sections 3.3 and 3.4\033[0m\n")
    print(" prerequisites (3.1)")
    test_fluid_model_is_off(); test_xfrc_acts_at_com(); test_ballistic_without_aero()
    print("\n frame handling (3.3)")
    test_frame_roundtrip(); test_double_count_guard()
    print("\n parafoil lift and drag (3.4)")
    test_static_lift(); test_paper_static_lift(); test_alpha_clamp()
    print("\n pure moments (3.4)")
    test_moment_signs(); test_pitch_stiffness()
    print("\n end to end")
    test_energy_conservation(); test_lift_actually_supports_it()
    test_lift_drag_decomposition(); test_trim_airspeed()
    print("\n strip theory (option 3)")
    test_strip_matches_first_principles(); test_arch_recovery()
    test_strip_generates_roll_physics()
    print("\n free flight")
    test_unpowered_glide(); test_strip_fixes_the_spiral(); test_powered_climb()
    test_validated_thrust_envelope()
    print("\n scope")
    test_brakes_are_stubbed()
    n, tot = sum(_results), len(_results)
    print(f"\n{(GREEN if n==tot else RED)}{n}/{tot} passed{OFF}\n")
    raise SystemExit(0 if n == tot else 1)
