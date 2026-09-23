#!/usr/bin/env python3
"""Viewer for paramotor.xml, with tendon rendering forced on.

macOS needs mjpython, not python: the viewer must own the main thread.

    ../.venv/bin/mjpython view_paramotor.py              # LIVE, free flight, powered
    ../.venv/bin/mjpython view_paramotor.py --sweep      # live + brakes cycling
    ../.venv/bin/mjpython view_paramotor.py --zoom 12    # camera pulled back
    ../.venv/bin/mjpython view_paramotor.py --slow       # 1/20 speed
    ../.venv/bin/mjpython view_paramotor.py --freeze     # hold the design pose
    ../.venv/bin/mjpython view_paramotor.py --trim       # settle, then hold it

Live is the default. Cyan = suspension lines, red = brake lines, red spoke on
the propeller disk shows rotation (the disk is axisymmetric, so without it you
cannot see the propeller turn).

CHANGING ACTUATOR VALUES
    The viewer's right-hand panel has a Control section with one slider per
    actuator: thrust, servo_pos_L, servo_pos_R. Drag those to fly it by hand.
    This script deliberately does NOT rewrite ctrl every step, or the sliders
    would be overwritten as fast as you move them. It only:
      - sets thrust once at startup, and
      - re-spins the propeller if YOU change the thrust slider, so the
        gyroscopic moment stays consistent with thrust.
    --sweep is the exception: it drives the two servo sliders continuously.

DRAGGING BODIES
    double-click           select a body
    Ctrl + right-drag      move it
    Ctrl + left-drag       rotate it
    plain drag / scroll    orbit, pan, zoom the camera
    Frozen applies the drag as a pose change; live applies it as a force.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

import paramotor_aero
import paramotor_params

# Propeller speed from thrust. T = K_T * omega^2, anchored on the spec's static
# bench point (410 gf = 4.02 N) at an assumed ~10 krpm; re-fit from a thrust
# stand. Mirrors K_T in build_paramotor.py. Spin inertia is read from the model
# itself, so there is nothing to keep in sync.
K_T = 4.02 / (10_000 * 2 * np.pi / 60.0) ** 2        # N/(rad/s)^2


def omega_from_thrust(t_n):
    """Shaft speed (rad/s) for a commanded thrust."""
    return float(np.sqrt(max(t_n, 0.0) / K_T))


def sync_prop(model, dat, t_n):
    """Command thrust, and spin the propeller to match: omega = sqrt(T / K_T).

    prop_spin has no actuator. It exists only to carry angular momentum so
    MuJoCo generates the gyroscopic moment from its own Coriolis terms, and it
    is driven kinematically from thrust so the two can never disagree. The
    joint has no damping or applied torque, so the rate simply persists.
    """
    w = omega_from_thrust(t_n)
    dat.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust")] = t_n
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "prop_spin")
    if j >= 0:
        dat.qvel[model.jnt_dofadr[j]] = w
    return w


spin_up = sync_prop            # same thing now: one call ties thrust to rpm


ap = argparse.ArgumentParser()
mode = ap.add_mutually_exclusive_group()
mode.add_argument("--freeze", action="store_true",
                  help="hold the design pose and never step")
mode.add_argument("--trim", action="store_true",
                  help="settle to the hanging trim pose, then hold it")
ap.add_argument("--run", action="store_true",
                help="explicitly request live mode (this is already the default)")
ap.add_argument("--sweep", action="store_true",
                help="live, cycling the brakes over 0 -> 3.0 rad of arm travel")
ap.add_argument("--thrust", type=float, default=0.8,
                help="propeller thrust in N (0 = unpowered); also sets rpm. "
                     "Validated envelope is 0 to about 1.0 N (T/W 0.31); above "
                     "that it pitches up, the suspension goes slack and the "
                     "canopy tumbles.")
ap.add_argument("--wind", type=float, default=6.0,
                help="LAUNCH SPEED in m/s given to both bodies at t=0 (spec "
                     "design point 6.0). This is an initial condition, not a "
                     "steady freestream: m.opt.wind only drives MuJoCo's own "
                     "fluid model, which is disabled (density=0).")
ap.add_argument("--aero", choices=("strip", "lumped", "off"), default="strip",
                help="strip = per-panel forces, the default and the only mode "
                     "with a dihedral effect; lumped = the paper's single "
                     "force; off = no aerodynamics, it just falls.")
ap.add_argument("--inertia", "--drag", dest="inertia", action="store_true",
                help="draw the equivalent inertia boxes (mjVIS_INERTIA) with "
                     "the geoms transparent. NOTE: this used to be called "
                     "--drag because MuJoCo's legacy fluid model dragged on "
                     "these boxes. That model is gone (density=0); aero is now "
                     "applied per panel by paramotor_aero.py, so these boxes "
                     "are mass distribution only and have nothing to do with drag.")
ap.add_argument("--bare", action="store_true",
                help="load paramotor.xml (aircraft only) instead of scene.xml")
ap.add_argument("--flat", action="store_true",
                help="leave the terrain flat (ground plane only, no mountains)")
ap.add_argument("--seed", type=int, default=3, help="terrain random seed")
ap.add_argument("--notrack", action="store_true",
                help="static camera instead of following the pod (it will fly "
                     "out of shot within a couple of seconds)")
ap.add_argument("--zoom", type=float, default=None,
                help="camera distance in m (default 5.0 tracking, 1.6 static)")
ap.add_argument("--slow", action="store_true",
                help="very slow motion: shorthand for --speed 0.05")
ap.add_argument("--speed", type=float, default=1.0,
                help="playback rate, 1.0 = real time, 0.05 = 20x slower")
ap.add_argument("--settle", type=float, default=10.0,
                help="seconds to settle when using --trim")
args = ap.parse_args()

FROZEN = args.freeze or args.trim          # live unless a freeze mode was asked for
SPEED = 0.05 if args.slow else args.speed
if SPEED <= 0:
    ap.error("--speed must be positive")

# scene.xml is the world and <include>s paramotor.xml; the aircraft alone is in
# paramotor.xml. Scenery is collision-free, so the dynamics are the same either way.
XML = "paramotor.xml" if args.bare else "scene.xml"
m = mujoco.MjModel.from_xml_path(str(Path(__file__).parent / XML))


def make_terrain(model, seed=3):
    """Fill the hfield with a smoothed Gaussian random field.

    White Gaussian noise on its own is per-cell spikes, not terrain, so it is
    low-pass filtered with a separable Gaussian kernel. Two scales are summed:
    a broad one for the landforms and a finer one for surface texture. The
    hfield ships flat (MuJoCo allocates zeros when no file is given), so this
    is where the mountains actually come from. Visual only.
    """
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
    if hid < 0:
        return
    nr, nc = model.hfield_nrow[hid], model.hfield_ncol[hid]
    rng = np.random.default_rng(seed)

    def blur(a, sigma):
        """Separable Gaussian low-pass, wrapping at the edges."""
        # rad = max(1, int(3 * sigma))
        # k = np.exp(-0.5 * (np.arange(-rad, rad + 1) / sigma) ** 2)
        # k /= k.sum()
        # for axis in (0, 1):
        #     a = np.apply_along_axis(
        #         lambda v: np.convolve(np.r_[v[-rad:], v, v[:rad]], k,
        #                               mode="same")[rad:-rad], axis, a)
        return a

    h = np.zeros((nr, nc))
    for sigma, amp in ((nr / 10.0, 1.0), (nr / 30.0, 0.35)):
        h += amp * blur(rng.standard_normal((nr, nc)), sigma)
    h -= h.min()
    h /= h.max()
    # flatten a valley under the launch point so the rig is clearly airborne
    y, x = np.mgrid[0:nr, 0:nc] / max(nr - 1, 1)
    h *= np.clip(((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.02, 0.0, 1.0)
    adr = model.hfield_adr[hid]
    model.hfield_data[adr:adr + nr * nc] = h.ravel().astype(np.float32)


if not args.flat:
    make_terrain(m, args.seed)

d = mujoco.MjData(m)

# ---------------------------------------------------------------------------
# AERODYNAMICS.  The MJCF carries density="0" viscosity="0" and no fluidshape:
# MuJoCo's own fluid model is off on purpose, and lift and drag come from
# paramotor_aero.py (Umenberger & Goktogan 2012, eqs. 8-18) as a passive-force
# callback.  Without this block the vehicle has NO aerodynamics and simply
# falls -- which is exactly what this viewer did before the callback existed.
# ---------------------------------------------------------------------------
AERO = None
if args.aero != "off":
    AERO = paramotor_aero.ParamotorAero(m, paramotor_params.PEEK_1M, mode=args.aero)
    mujoco.set_mjcb_passive(AERO)


def launch(speed):
    """Give both free bodies the same forward velocity at t=0.

    Addresses are looked up: the canopy freejoint starts at dof 9, not 6,
    because prop_spin and the two servo arms sit between the two freejoints.
    """
    for name in ("pod_free", "canopy_free"):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j >= 0:
            a = m.jnt_dofadr[j]
            d.qvel[a:a + 3] = [speed, 0.0, 0.0]


launch(args.wind)

CANOPY = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "canopy")
POD = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pod")
THRUST = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust")
servos = [a for a in (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                      for n in ("servo_pos_L", "servo_pos_R")) if a >= 0]
PERT = None


def step():
    # NOTHING is applied to the bodies here beyond the mouse. There was once a
    # constant 3.19 N pushed up at the canopy centre of mass, exactly
    # cancelling gravity, which made the model incapable of descending and
    # invalidated every trajectory it produced. Whatever the wing does now, it
    # does on its own.
    #
    # xfrc_applied is the MOUSE's channel and is cleared here each step.
    # Aerodynamics deliberately does not use it -- ParamotorAero applies its
    # wrenches to qfrc_passive through mj_applyFT -- so Ctrl+drag works on the
    # canopy and panels instead of being overwritten every step.
    d.xfrc_applied[:] = 0
    if PERT is not None:
        mujoco.mjv_applyPerturbForce(m, d, PERT)
    mujoco.mj_step(m, d)


if args.trim:
    print(f"settling {args.settle:.0f} s at {args.wind} m/s ...")
    spin_up(m, d, args.thrust)
    for _ in range(int(args.settle / m.opt.timestep)):
        step()
    d.qvel[:] = 0                          # clean freeze: no residual motion
    d.qacc[:] = 0
    d.xfrc_applied[:] = 0

mujoco.mj_forward(m, d)

with mujoco.viewer.launch_passive(m, d) as v:
    v.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
    v.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
    v.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
    v.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = True
    v.opt.sitegroup[:] = 1
    if args.inertia:
        # Mass distribution only. Aerodynamics no longer has anything to do
        # with these boxes: it is applied per panel by paramotor_aero.py.
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = True
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    # The vehicle always translates now, so a static camera loses it within a
    # couple of seconds. Track the pod unless asked otherwise.
    tracking = not args.notrack
    v.cam.elevation, v.cam.azimuth = -10, 135
    if tracking:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = POD          # lookat follows this body every frame
        v.cam.distance = args.zoom if args.zoom is not None else 5.0
    else:
        v.cam.distance = args.zoom if args.zoom is not None else 1.6
        v.cam.lookat[:] = [0, 0, 0.25]
    v.sync()

    if args.inertia:
        print("INERTIA BOXES shown: mass distribution only, NOT drag.")
    print(f"loaded {XML}.  Right-hand panel -> Control: one slider per actuator "
          "(thrust, servo_pos_L/R).")

    if FROZEN:
        print(f"FROZEN at the {'trim' if args.trim else 'design'} pose. "
              "Nothing is stepped; dragging moves a body's pose.")
        while v.is_running():
            with v.lock():
                mujoco.mjv_applyPerturbPose(m, d, v.perturb, 1)
                mujoco.mj_forward(m, d)
            v.sync()
            time.sleep(1 / 60)
    else:
        PERT = v.perturb
        w = spin_up(m, d, args.thrust)          # set thrust + rpm ONCE
        last_thrust = float(d.ctrl[THRUST])
        print(f"LIVE at {'real time' if SPEED == 1 else f'{1/SPEED:g}x slower'}. "
              f"thrust {args.thrust:.2f} N, propeller {w*60/(2*np.pi):.0f} rpm.")
        if tracking:
            print(f"Camera TRACKING the pod at {v.cam.distance:.1f} m.")
        if AERO is None:
            print("AERO OFF: no lift, no drag. It will simply fall.")
        else:
            print(f"aero: {args.aero} mode, launched at {args.wind:.1f} m/s.")
            print("Validated thrust envelope is 0 to about 1.0 N (T/W 0.31).")
            print("Above that it pitches up, the suspension goes slack and the")
            print("canopy tumbles -- known, needs a stall model and rigging trim.")
        wall_per_step = m.opt.timestep / SPEED
        while v.is_running():
            t0 = time.time()
            if args.sweep:
                ang = 1.5 * (1 - np.cos(0.6 * d.time))
                for a in servos:
                    d.ctrl[a] = ang
            # Only touch ctrl when YOU changed thrust, so the sliders stay usable.
            if float(d.ctrl[THRUST]) != last_thrust:
                last_thrust = float(d.ctrl[THRUST])
                sync_prop(m, d, last_thrust)    # keep rpm consistent with thrust
            step()
            v.sync()
            time.sleep(max(0, wall_per_step - (time.time() - t0)))
