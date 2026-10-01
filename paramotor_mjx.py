"""JAX port of the accepted strip aerodynamics; no extra brake-panel forces.

The native model remains the reference. The short forward pipeline mirrors
MuJoCo MJX 3.13.0, inserting aero after velocities and before acceleration.
Private stage APIs are isolated here and the dependency is pinned accordingly.
"""

from pathlib import Path
import jax.numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import forward, sensor, solver
import numpy as np

from paramotor_params import PEEK_1M
from paramotor_control import K_T

XML = Path(__file__).with_name("paramotor.xml")
FLIP = jp.array([1.0, -1.0, -1.0])


class ParamotorMJX:
    def __init__(self, iterations=10, ls_iterations=5):
        self.native = m = mujoco.MjModel.from_xml_path(str(XML))
        if m.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
            raise ValueError("RL expects the accepted implicitfast integrator")
        if m.opt.density or m.opt.viscosity:
            raise ValueError("Built-in fluid forces would double-count aero")
        m.opt.iterations, m.opt.ls_iterations = iterations, ls_iterations
        self.model = mjx.put_model(m, impl="jax")
        self.p = dict(PEEK_1M)
        self.canopy, self.pod = m.body("canopy").id, m.body("pod").id
        self.thrust = m.actuator("thrust").id
        self.servos = jp.array([m.actuator("servo_pos_" + s).id for s in ("L", "R")])
        self.prop_dof = int(m.joint("prop_spin").dofadr[0])
        self.free_qpos = [
            int(m.joint(n).qposadr[0]) for n in ("pod_free", "canopy_free")
        ]
        self.free_dofs = [
            int(m.joint(n).dofadr[0]) for n in ("pod_free", "canopy_free")
        ]
        ids = [i for i in range(m.nbody) if m.body(i).name.startswith("panel_")]
        if any(m.body_dofnum[i] for i in ids):
            raise ValueError("The aero fast path requires rigid panels")
        self.panels = jp.array(ids)
        self.areas = jp.array(
            [4 * np.prod(m.geom_size[m.body_geomadr[i], :2]) for i in ids]
        )
        d = mujoco.MjData(m)
        # Native callback may belong to another model: do not install one here.
        mujoco.mj_forward(m, d)
        projection = np.average(d.xmat[ids, 8], weights=np.asarray(self.areas))
        self.arch_recovery = 1 / float(projection)
        self.roots = jp.array(m.body_rootid)
        self.template = mjx.make_data(self.model)

    def body_velocity(self, d, body):
        omega = d.cvel[body, :3]
        offset = d.xipos[body] - d.subtree_com[self.roots[body]]
        return omega, d.cvel[body, 3:] + jp.cross(omega, offset)

    def panel_flow(self, d):
        omega, velocity = self.canopy_velocity(d)
        v = velocity + jp.cross(omega, d.xipos[self.panels] - d.subtree_com[self.canopy])
        return jp.einsum("nji,nj->ni", d.xmat[self.panels], v) * FLIP

    def canopy_velocity(self, d):
        """Velocity at the rigid canopy assembly CoM, not its massless root."""
        omega, velocity = self.body_velocity(d, self.canopy)
        offset = d.subtree_com[self.canopy] - d.xipos[self.canopy]
        return omega, velocity + jp.cross(omega, offset)

    def raw_alpha(self, d):
        flow = self.panel_flow(d)
        return jp.arctan2(flow[:, 2], flow[:, 0])

    def wrench(self, d):
        p = self.p
        flow = self.panel_flow(d)
        speed = jp.linalg.norm(flow, axis=1)
        alpha = jp.clip(
            jp.arctan2(flow[:, 2], flow[:, 0]), p["alpha_min"], p["alpha_max"]
        )
        cl = (p["CL0"] + p["CLa"] * alpha) * self.arch_recovery
        cd = p["CD0"] + p["CDa"] * alpha**2
        lift = jp.stack((flow[:, 2], jp.zeros_like(speed), -flow[:, 0]), axis=1)
        forces = (
            0.5
            * p["rho"]
            * self.areas[:, None]
            * speed[:, None]
            * (cl[:, None] * lift - cd[:, None] * flow)
        )
        forces = jp.where((speed > 1e-6)[:, None], forces, 0.0)
        world = jp.einsum("nij,nj->ni", d.xmat[self.panels], forces * FLIP)
        wrench = jp.zeros((self.native.nbody, 6)).at[self.panels, :3].set(world)
        omega, velocity = self.canopy_velocity(d)
        rates = (d.xmat[self.canopy].T @ omega) * FLIP
        V = jp.linalg.norm(velocity)
        Vsafe = jp.maximum(V, 1e-6)
        c, b = p["c"], p["b"]
        mean_alpha = jp.sum(alpha * self.areas) / jp.sum(self.areas)
        moments = (
            0.5
            * p["rho"]
            * p["AP"]
            * V**2
            * jp.array(
                [
                    0.0,
                    p["Cmq"] * c * c * rates[1] / (2 * Vsafe)
                    + p["Cm0"] * c
                    + p["Cma"] * c * mean_alpha,
                    p["Cnr"] * b * b * rates[2] / (2 * Vsafe),
                ]
            )
        )
        moments = jp.where(V > 1e-6, moments, 0.0)
        wrench = wrench.at[self.canopy, 3:].set(d.xmat[self.canopy] @ (moments * FLIP))
        _, vp = self.body_velocity(d, self.pod)
        vf = (d.xmat[self.pod].T @ vp) * FLIP
        vf_speed = jp.linalg.norm(vf)
        af = jp.arctan2(vf[2], vf[0])
        drag = (
            -0.5 * p["rho"] * p["AF"] * vf_speed * (p["CD0F"] + p["CDaF"] * af**2) * vf
        )
        drag = jp.where(vf_speed > 1e-6, drag, 0.0)
        return wrench.at[self.pod, :3].set(d.xmat[self.pod] @ (drag * FLIP))

    def forward(self, d, with_sensors=True):
        m = self.model
        d = forward.fwd_position(m, d)
        if with_sensors:
            d = sensor.sensor_pos(m, d)
        d = forward.fwd_velocity(m, d)
        if with_sensors:
            d = sensor.sensor_vel(m, d)
        # MJX maps these CoM wrenches through the same Jacobians as mj_applyFT.
        d = d.replace(xfrc_applied=self.wrench(d))
        d = forward.fwd_actuation(m, d)
        d = forward.fwd_acceleration(m, d)
        d = solver.solve(m, d) if d._impl.efc_J.size else d.replace(qacc=d.qacc_smooth)
        return sensor.sensor_acc(m, d) if with_sensors else d

    def integrate(self, d):
        return forward.implicit(self.model, d)

    def step(self, d, with_sensors=True):
        return self.integrate(self.forward(d, with_sensors=with_sensors))

    def command(self, d, thrust, brakes):
        ctrl = d.ctrl.at[self.thrust].set(thrust).at[self.servos].set(brakes)
        # Same behavior as viewer: resync spin only when thrust changes.
        speed = jp.sqrt(jp.maximum(thrust, 0.0) / K_T)
        qvel = d.qvel.at[self.prop_dof].set(
            jp.where(thrust != d.ctrl[self.thrust], speed, d.qvel[self.prop_dof])
        )
        return d.replace(ctrl=ctrl, qvel=qvel)

    def initial(self, speed=6.0, altitude=40.0, thrust=0.8):
        d = self.template
        qpos, qvel = d.qpos, d.qvel
        for a, b in zip(self.free_qpos, self.free_dofs):
            qpos = qpos.at[a + 2].add(altitude)
            qvel = qvel.at[b].set(speed)
        d = self.command(d.replace(qpos=qpos, qvel=qvel), thrust, jp.zeros(2))
        return self.forward(d)
