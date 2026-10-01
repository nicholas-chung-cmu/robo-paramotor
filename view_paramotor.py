#!/usr/bin/env python3
"""Viewer for paramotor.xml, with tendon rendering forced on.

macOS needs mjpython, not python: the viewer must own the main thread.

    ../.venv/bin/mjpython view_paramotor.py              # LIVE, free flight, powered
    ../.venv/bin/mjpython view_paramotor.py --csv flight.csv # log odometry
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
    Hold Left / Right to target full left / right brake; release to target zero.
    Both arrows can be held together. Brake commands follow a critically damped
    response, reaching 99% of the target in about one simulation second.
    Sliders and --sweep use the same smoothing. Brake keys stop --sweep.
    Up / Down changes thrust by 0.1 N per press; holding repeats.
    Thrust changes also update propeller speed to keep gyroscopic effects consistent.

DRAGGING BODIES
    double-click           select a body
    Ctrl + right-drag      move it
    Ctrl + left-drag       rotate it
    plain drag / scroll    orbit, pan, zoom the camera
    Frozen applies the drag as a pose change; live applies it as a force.
"""
import argparse
import csv
from contextlib import ExitStack
import time
from queue import SimpleQueue

import glfw
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

import paramotor_aero
import paramotor_params

from paramotor_control import sync_prop, smooth_brakes


spin_up = sync_prop            # same thing now: one call ties thrust to rpm


class OdometryCSV:
    """20 Hz truth data: world XYZ (Z up), FLU body rates, angles in degrees.

    Course is the horizontal CoM velocity heading, not the pod yaw. Course
    rate is a wrapped finite difference between samples; blank at startup,
    resets, or when horizontal speed is below 0.1 m/s. A private MjData copy
    keeps sampling from changing the simulated state.
    """
    def __init__(self, model, stream):
        self.model, self.stream = model, stream
        self.data = mujoco.MjData(model)
        self.writer = csv.writer(stream)
        self.next_time, self.previous = 0.0, None
        self.writer.writerow([
            "time_s", "com_x_m", "com_y_m", "com_z_m",
            "com_vx_mps", "com_vy_mps", "com_vz_mps", "speed_mps",
            "course_deg", "course_rate_degps",
            "pod_roll_deg", "pod_pitch_deg", "pod_yaw_deg",
            "pod_wx_degps", "pod_wy_degps", "pod_wz_degps",
            "canopy_roll_deg", "canopy_pitch_deg", "canopy_yaw_deg",
            "thrust_N", "brake_L_target_rad", "brake_L_command_rad", "brake_L_actual_rad",
            "brake_R_target_rad", "brake_R_command_rad", "brake_R_actual_rad",
        ])
        stream.flush()

    def sample(self, source, brake_states):
        if self.previous is not None and source.time < self.previous[0]:
            self.next_time, self.previous = source.time, None
        if source.time + 1e-9 < self.next_time:
            return
        m, d = self.model, self.data
        mujoco.mj_copyData(d, m, source)
        # mj_step leaves some derived quantities at the preceding state.
        mujoco.mj_forward(m, d)
        mujoco.mj_subtreeVel(m, d)
        velocity = d.subtree_linvel[0]
        course = float(np.arctan2(velocity[1], velocity[0])) if np.linalg.norm(velocity[:2]) >= 0.1 else None
        course_rate = None
        if self.previous is not None:
            last_time, last_course = self.previous
            if course is not None and last_course is not None and d.time > last_time:
                change = np.arctan2(np.sin(course-last_course), np.cos(course-last_course))
                course_rate = float(np.degrees(change) / (d.time-last_time))

        def attitude(body):
            R = d.xmat[m.body(body).id].reshape(3, 3)
            return np.degrees([np.arctan2(R[2, 1], R[2, 2]),
                               np.arcsin(np.clip(-R[2, 0], -1, 1)),
                               np.arctan2(R[1, 0], R[0, 0])]).tolist()

        spatial = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, m.body("pod").id, spatial, 1)
        row = [float(d.time), *d.subtree_com[0], *velocity, float(np.linalg.norm(velocity)),
               None if course is None else float(np.degrees(course)), course_rate,
               *attitude("pod"), *np.degrees(spatial[:3]), *attitude("canopy"),
               float(d.ctrl[m.actuator("thrust").id])]
        for side in ("L", "R"):
            actuator = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "servo_pos_"+side)
            actual = float(d.qpos[m.joint("arm_"+side).qposadr[0]])
            row.extend([brake_states[actuator][2], float(d.ctrl[actuator]), actual]
                       if actuator in brake_states else [None, None, actual])
        self.writer.writerow(row)
        self.stream.flush()
        self.previous = (float(d.time), course)
        self.next_time = float(d.time) + 0.05


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
                     "Powered flight envelope is not yet validated; neutral "
                     "controls can produce a sustained turn.")
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
ap.add_argument("--csv", type=Path, metavar="PATH",
                help="write ground-truth odometry and controls at 20 Hz of simulation time; "
                     "PATH must not already exist")
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
control_keys = SimpleQueue()
THRUST_KEY_STEP = 0.1  # N per press or key-repeat event
brake_actuators = {
    glfw.KEY_LEFT: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "servo_pos_L"),
    glfw.KEY_RIGHT: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "servo_pos_R"),
}
sweep_enabled = args.sweep
# [smoothed position, velocity, target], in servo radians and simulation seconds.
brake_states = {a: [float(d.ctrl[a]), 0.0, float(d.ctrl[a])] for a in servos}


# Keep C callback references alive until the viewer window has been destroyed.
raw_key_callback = None
native_key_callback = None


def on_glfw_key(window, key, scancode, action, mods):
    """Receive actual press/release events, retaining native non-brake controls."""
    if key in brake_actuators:
        control_keys.put((key, action != glfw.RELEASE))
    elif key in (glfw.KEY_UP, glfw.KEY_DOWN):
        if action != glfw.RELEASE:
            control_keys.put((key, True))
    elif native_key_callback:
        native_key_callback(window, key, scancode, action, mods)


def on_key(key):
    """Install the release-aware hook on the viewer thread's first key event.

    MuJoCo's public callback exposes presses only. GLFW's Python wrapper
    cannot return a callback installed by C++, so use the underlying setter
    to preserve and chain the viewer's native callback. GLFW synthesizes key
    releases when focus is lost, which also releases held brakes.
    """
    global raw_key_callback, native_key_callback
    if raw_key_callback is not None:
        return
    window = glfw.get_current_context()
    if not window:
        raise RuntimeError("Cannot attach brake controls: viewer has no GLFW context")
    raw_key_callback = glfw._GLFWkeyfun(on_glfw_key)
    native_key_callback = glfw._glfw.glfwSetKeyCallback(window, raw_key_callback)
    # This first press arrived through the old callback, before our hook.
    if key in brake_actuators or key in (glfw.KEY_UP, glfw.KEY_DOWN):
        control_keys.put((key, True))


def apply_control_keys():
    global sweep_enabled
    # Slider edits are new targets; d.ctrl always receives the smoothed value
    # before physics and the aerodynamic callback run.
    for actuator, state in brake_states.items():
        if d.ctrl[actuator] != state[0]:
            lo, hi = m.actuator_ctrlrange[actuator]
            state[2] = float(np.clip(d.ctrl[actuator], lo, hi))
    while not control_keys.empty():
        key, pressed = control_keys.get_nowait()
        if key in (glfw.KEY_UP, glfw.KEY_DOWN):
            if pressed and THRUST >= 0:
                change = THRUST_KEY_STEP if key == glfw.KEY_UP else -THRUST_KEY_STEP
                lo, hi = m.actuator_ctrlrange[THRUST]
                value = float(np.clip(d.ctrl[THRUST] + change, lo, hi))
                sync_prop(m, d, value)
                print(f"Thrust: {value:.2f} N")
            continue
        sweep_enabled = False
        actuator = brake_actuators[key]
        if actuator >= 0:
            brake_states[actuator][2] = m.actuator_ctrlrange[actuator, 1] if pressed else 0.0



def advance_brakes(dt):
    """Exact critically damped PD response; safe even if the timestep changes.

    x'' = rate**2 * (target - x) - 2 * rate * x'
    Keep velocity across target changes so a quick release stays smooth.
    The physical servo actuator tracks this smoothed control target.
    """
    for actuator, state in brake_states.items():
        position, velocity, target = state
        position, velocity = smooth_brakes(position, velocity, target, dt)
        state[:] = [position, velocity, target]
        d.ctrl[actuator] = position


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

with ExitStack() as stack:
    odometry = None
    if args.csv is not None:
        try:
            stream = stack.enter_context(args.csv.open("x", newline=""))
        except OSError as exc:
            ap.error(f"cannot create CSV {args.csv}: {exc}")
        odometry = OdometryCSV(m, stream)
        print(f"Logging odometry at 20 Hz simulation time to {args.csv.resolve()}")
    v = stack.enter_context(mujoco.viewer.launch_passive(m, d, key_callback=on_key))
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

    print("Thrust: Up/Down changes by 0.1 N; hold to repeat (model limits apply).")
    if servos:
        print("Brakes: hold Left/Right to target full brake; release to target zero. "
              "Smooth response: ~1 simulation second. Brake keys stop --sweep.")
    else:
        print("Brake keys unavailable: this model has no position servo actuators.")

    if FROZEN:
        if odometry is not None:
            odometry.sample(d, brake_states)
        print(f"FROZEN at the {'trim' if args.trim else 'design'} pose. "
              "Nothing is stepped; dragging moves a body's pose.")
        while v.is_running():
            with v.lock():
                apply_control_keys()
                mujoco.mjv_applyPerturbPose(m, d, v.perturb, 1)
                mujoco.mj_forward(m, d)
            v.sync()
            time.sleep(1 / 60)
    else:
        PERT = v.perturb
        w = spin_up(m, d, args.thrust)          # set thrust + rpm ONCE
        last_thrust = float(d.ctrl[THRUST])
        if odometry is not None:
            odometry.sample(d, brake_states)
        print(f"LIVE at {'real time' if SPEED == 1 else f'{1/SPEED:g}x slower'}. "
              f"thrust {args.thrust:.2f} N, propeller {w*60/(2*np.pi):.0f} rpm.")
        if tracking:
            print(f"Camera TRACKING the pod at {v.cam.distance:.1f} m.")
        if AERO is None:
            print("AERO OFF: no lift, no drag. It will simply fall.")
        else:
            print(f"aero: {args.aero} mode, launched at {args.wind:.1f} m/s.")
            print("Powered flight envelope is not yet validated; neutral controls can turn.")
            print("Alpha clipping is a coefficient guard, not a stall model.")
        wall_per_step = m.opt.timestep / SPEED
        while v.is_running():
            # Wall-clock corrections must not turn a frame delay into a long sleep.
            t0 = time.monotonic()
            apply_control_keys()
            if sweep_enabled:
                ang = 1.5 * (1 - np.cos(0.6 * d.time))
                for a in servos:
                    brake_states[a][2] = ang
            advance_brakes(m.opt.timestep)
            # Only touch ctrl when YOU changed thrust, so the sliders stay usable.
            if float(d.ctrl[THRUST]) != last_thrust:
                last_thrust = float(d.ctrl[THRUST])
                sync_prop(m, d, last_thrust)    # keep rpm consistent with thrust
            step()
            if odometry is not None:
                odometry.sample(d, brake_states)
            v.sync()
            time.sleep(max(0, wall_per_step - (time.monotonic() - t0)))
