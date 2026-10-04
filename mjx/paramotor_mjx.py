"""JAX port of the accepted strip aerodynamics on the deformable canopy.

The native model (model/paramotor_aero.py) remains the reference; both share
its elementwise mesh-geometry functions. Physics runs on MJX's MuJoCo Warp
backend (GPU or CPU): each forward/step is one Warp call, including the flex
skin's bending and edge constraints. The aero wrench is computed in JAX from
qpos/qvel alone (canopy vertices are world-axis slide bodies; the pod is a free
body) and applied as xfrc_applied at each body's mass centre. MuJoCo, MJX and
warp-lang are pinned together; run the parity tests before upgrading.
"""

from pathlib import Path
import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import numpy as np
import warp
from mujoco.mjx.third_party.mujoco_warp._src.types import OverflowType
from mujoco.mjx.warp import types as warp_types

from model import paramotor_aero as aero
from model.paramotor_params import PEEK_1M
from model.paramotor_control import K_T

XML = Path(__file__).resolve().parents[1] / "model" / "paramotor.xml"
FLIP = jp.array([1.0, -1.0, -1.0])
# Full float32 products: GPU matmuls default to TF32 (~1e-3 relative error).
EXACT = jax.lax.Precision.HIGHEST
warp.config.quiet = True  # no per-kernel-module load messages


def quat_mat(q):
    w, x, y, z = q / jp.linalg.norm(q)
    return jp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class ParamotorMJX:
    def __init__(self, iterations=10, ls_iterations=5, graph_mode=None):
        """graph_mode: a mujoco.mjx.warp.types.GraphMode name (NONE, JAX, WARP,
        WARP_STAGED, WARP_STAGED_EX); None keeps MJX's default, WARP on a GPU."""
        self.native = m = mujoco.MjModel.from_xml_path(str(XML))
        if m.opt.density or m.opt.viscosity:
            raise ValueError("Built-in fluid forces would double-count aero")
        m.opt.iterations, m.opt.ls_iterations = iterations, ls_iterations
        model = mjx.put_model(m, impl="warp", graph_mode=(
            None if graph_mode is None else getattr(warp_types.GraphMode, graph_mode)))
        # ls_iterations is capped on purpose (EnvConfig.solver_ls_iterations);
        # Warp would print a warning every time a line search hits the cap.
        warp_opt = model.opt._impl
        quiet = int(warp_opt.warn_overflow) & ~int(OverflowType.LS_ITERATIONS)
        self.model = model.replace(opt=model.opt.replace(
            _impl=warp_opt.replace(warn_overflow=quiet)))
        self.p = dict(PEEK_1M)
        self.pod = m.body("pod").id
        self.thrust = m.actuator("thrust").id
        self.servos = jp.array([m.actuator("servo_pos_" + s).id for s in ("L", "R")])
        self.prop_dof = int(m.joint("prop_spin").dofadr[0])
        self.pod_qpos = int(m.joint("pod_free").qposadr[0])
        self.pod_dof = int(m.joint("pod_free").dofadr[0])
        self.pod_com = jp.array(m.body_ipos[self.pod])  # CoM in the pod frame

        mesh = aero.CanopyMesh(m)
        self.vertex_bodies = jp.array(mesh.bodies)
        self.vertex_qpos = jp.array(mesh.qpos)
        self.vertex_dof = jp.array(mesh.dof)
        self.rest = jp.array(mesh.rest)
        self.vertex_mass = jp.array(mesh.mass)
        self.le = mesh.le  # first lifting chord station
        rest = mesh.lifting(mesh.rest)
        _, _, _, R, area = aero.strip_cells(rest, np.zeros_like(rest), self.p)
        self.p["strip_cl_scale"] = aero.ParamotorAero._arch_recovery(R, area)
        # Warp sizes its constraint buffer from this argument, not the XML's
        # <size njmax>; the canopy's edge constraints alone exceed its default.
        self.template = mjx.make_data(m, impl="warp", njmax=int(m.njmax))

    # -- state ---------------------------------------------------------------
    def vertices(self, d):
        """Canopy vertex positions and velocities, (span rows, chord stations, 3)."""
        return self.rest + d.qpos[self.vertex_qpos], d.qvel[self.vertex_dof]

    def canopy_com(self, d):
        P, _ = self.vertices(d)
        m = self.vertex_mass[..., None]
        return (m * P).sum(axis=(0, 1)) / self.vertex_mass.sum()

    def raw_alpha(self, d):
        """Raw angle of attack of each spanwise strip (area-weighted over its chord).

        One value per strip, like the rigid panels this replaced, so a single
        flapping cell does not count as the whole strip leaving the envelope.
        """
        P, V = self.vertices(d)
        _, _, raw, _, area = aero.strip_cells(P[:, self.le:], V[:, self.le:], self.p, jp)
        return (raw * area).sum(axis=1) / area.sum(axis=1)

    def translate(self, qpos, offset):
        """Shift the whole vehicle (pod and every canopy vertex) by offset."""
        qpos = qpos.at[self.pod_qpos : self.pod_qpos + 3].add(offset)
        return qpos.at[self.vertex_qpos].add(offset)

    # -- aerodynamics --------------------------------------------------------
    def wrench(self, d):
        """World force/torque per body: skin cells on the vertices, pod drag."""
        p = self.p
        P, V = self.vertices(d)
        f_cells, alpha, _, R, area = aero.strip_cells(P[:, self.le:], V[:, self.le:], p, jp)
        F = jp.pad(aero.spread_to_corners(f_cells, jp), ((0, 0), (self.le, 0), (0, 0)))
        com, vbar, omega, R_c, inertia, r = aero.canopy_state(
            P, V, self.vertex_mass, R, area, jp)

        # Pure pitch/yaw moments about the skin's mass centre (roll is native
        # to the strip distribution), applied as a zero-net-force couple.
        rates = aero.to_local(R_c, omega, jp) * FLIP
        speed = jp.sqrt((vbar * vbar).sum())
        safe = jp.maximum(speed, 1e-6)
        c, b = p["c"], p["b"]
        mean_alpha = (alpha * area).sum() / area.sum()
        moments = 0.5 * p["rho"] * p["AP"] * speed**2 * jp.array([
            0.0,
            p["Cmq"] * c * c * rates[1] / (2 * safe) + p["Cm0"] * c + p["Cma"] * c * mean_alpha,
            p["Cnr"] * b * b * rates[2] / (2 * safe),
        ])
        moments = jp.where(speed > 1e-6, moments, 0.0)
        F = F + aero.couple_forces(aero.to_world(R_c, moments * FLIP, jp),
                                   inertia, r, self.vertex_mass, jp)
        # add, not set: a merged tip vertex appears twice in the grid
        wrench = jp.zeros((self.native.nbody, 6)).at[self.vertex_bodies, :3].add(F)

        # Fuselage drag at the pod mass centre.
        a, v = self.pod_qpos, self.pod_dof
        pod_r = quat_mat(d.qpos[a + 3 : a + 7])
        pod_w = jp.matmul(pod_r, d.qvel[v + 3 : v + 6], precision=EXACT)
        pod_v = d.qvel[v : v + 3] + jp.cross(
            pod_w, jp.matmul(pod_r, self.pod_com, precision=EXACT))
        vf = jp.matmul(pod_r.T, pod_v, precision=EXACT) * FLIP
        vf_speed = jp.linalg.norm(vf)
        af = jp.arctan2(vf[2], vf[0])
        drag = -0.5 * p["rho"] * p["AF"] * vf_speed * (p["CD0F"] + p["CDaF"] * af**2) * vf
        drag = jp.where(vf_speed > 1e-6, drag, 0.0)
        return wrench.at[self.pod, :3].set(jp.matmul(pod_r, drag * FLIP, precision=EXACT))

    # -- dynamics ------------------------------------------------------------
    def forward(self, d):
        """Forward dynamics with aero, including sensordata."""
        return mjx.forward(self.model, d.replace(xfrc_applied=self.wrench(d)))

    def step(self, d):
        """One physics step. Like mj_step, sensordata is for the pre-step state."""
        return mjx.step(self.model, d.replace(xfrc_applied=self.wrench(d)))

    def command(self, d, thrust, brakes):
        ctrl = d.ctrl.at[self.thrust].set(thrust).at[self.servos].set(brakes)
        # Same behavior as viewer: resync spin only when thrust changes.
        speed = jp.sqrt(jp.maximum(thrust, 0.0) / K_T)
        qvel = d.qvel.at[self.prop_dof].set(
            jp.where(thrust != d.ctrl[self.thrust], speed, d.qvel[self.prop_dof])
        )
        return d.replace(ctrl=ctrl, qvel=qvel)

    def initial(self, speed=6.0, altitude=40.0, thrust=0.8):
        """Design shape at altitude, pod and every vertex moving at speed along +x."""
        d = self.template
        qpos = self.translate(d.qpos, jp.array([0.0, 0.0, altitude]))
        qvel = d.qvel.at[self.pod_dof].set(speed).at[self.vertex_dof[..., 0]].set(speed)
        d = self.command(d.replace(qpos=qpos, qvel=qvel), thrust, jp.zeros(2))
        return self.forward(d)
