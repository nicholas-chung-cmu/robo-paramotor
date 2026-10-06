"""Batched-friendly MJX path following. No Gym wrappers or global callbacks."""

from dataclasses import dataclass, field, fields
from typing import Any
import jax
import jax.numpy as jp
from flax import struct
import mujoco
import numpy as np

from rl import machine, routes
from model.paramotor_control import smooth_brakes
from mjx.paramotor_mjx import ParamotorMJX
from model import paramotor_aero as aero
from mujoco.mjx._src.types import tree_path_to_attr_str
from mujoco.mjx.warp import types as warp_types


def quat_mul(a, b):
    """Hamilton product, wxyz."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return jp.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def rotvec_quat(v):
    """Rotation vector (rad) to unit quaternion, wxyz."""
    angle = jp.linalg.norm(v)
    scale = jp.where(angle > 1e-12, jp.sin(0.5 * angle) / jp.maximum(angle, 1e-12), 0.5)
    return jp.concatenate((jp.cos(0.5 * angle)[None], scale * v))


def quat_mat(q):
    w, x, y, z = q
    return jp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def heading_mat(rotation):
    """Yaw-only part of a body-to-world rotation: heading frame to world."""
    yaw = jp.arctan2(rotation[1, 0], rotation[0, 0])
    co, si = jp.cos(yaw), jp.sin(yaw)
    return jp.array([[co, -si, 0], [si, co, 0], [0, 0, 1]])


@dataclass
class EnvConfig:
    control_hz: int = 25
    sensor_hz: int = 100
    gps_hz: int = 5
    baro_hz: int = 20  # BMP581 output rate; held between samples like GPS
    history_hz: int = 25  # observation history is downsampled from sensor_hz
    history_seconds: float = 2.0  # canopy dynamics are slow and unobserved
    episode_seconds: float = 300.0
    thrust_max: float = 2.0  # hardware thrust clamp; equals the XML ctrlrange
    initial_thrust: float = 0.8
    launch_speed: float = 6.0  # m/s, the canopy's design airspeed
    altitude: float = 40.0
    # Look-ahead, in points past the target. 0 and 1 (the target and the next
    # point, 0-20 m ahead) were added so the policy sees a small sideways
    # offset; with only (2, 4, 8) it drifted ~9 m off straight routes.
    preview_index: tuple = (0, 1, 2, 4, 8)
    success_radius_m: float = 2.0  # full pass reward within this 3D miss distance
    pass_sigma_m: float = 2.0  # pass reward decays as a Gaussian beyond the radius
    progress_reward_per_m: float = 0.1  # shaping per metre of distance-to-go closed
    gps_noise_limit_m: float = 1.0  # hard clip on GNSS horizontal error
    route_points: int = 101  # route stored as route_points, route_spacing_m apart (~1 km)
    route_spacing_m: float = 10.0
    path_kind: str = "random"
    curriculum: bool = True
    # Datasheet values from sensors.csv, keyed by its channel names.
    # Noise is temporarily off by default (see todo/TODO.md). The datasheet
    # values are sensor_spec.defaults("noise_std") and ("bias_std").
    noise_std: dict = field(default_factory=dict)
    bias_std: dict = field(default_factory=dict)
    bias_walk_std: dict = field(default_factory=dict)  # channel units / sqrt(second)
    gps_latency_s: float = 0.0
    gps_dropout: float = 0.0
    position_randomization_m: float = 1.0  # sideways launch offset, +-
    # Launch height offset from the route, +- (uniform): every flight starts
    # off the route's height and has to correct it (was 0.3 m).
    start_height_noise_m: float = 3.0
    max_cross_track_m: float = 35.0
    max_vertical_m: float = 15.0  # off the route vertically ends the episode (was 25)
    # Leaving the aero model's alpha range no longer ends an episode. Instead
    # each control step earns this times the fraction of wing strips inside it
    # (at most 0.125/s at 25 Hz, against ~0.6/s for flying the route at 6 m/s).
    envelope_reward: float = 0.005
    # Penalty on the squared change of action between control steps (was 0.02).
    # Off for now.
    action_change_penalty: float = 0.0
    # Curriculum-scaled terms. `difficulty` (0 -> 1, raised by the training
    # gate) lengthens the route and fades the early helpers out.
    min_route_points: int = 16       # route length at difficulty 0 (150 m at 10 m)
    dense_reward: float = 0.02       # per step: heading along the route, altitude error
    alive_reward: float = 0.01       # per step survival bonus
    failure_penalty_start: float = 2.0  # failure penalty at difficulty 0 ...
    failure_penalty: float = 10.0       # ... rising to this at difficulty 1
    # Height. Unlike the early helpers these never fade with the curriculum.
    # Each step costs altitude_penalty per metre off the route's height beyond
    # the deadband, below_route_factor times as much below it as above, capped
    # per step (against ~0.05 per step for flying the route well). Ending on
    # the ground or past max_vertical_m costs altitude_failure_penalty instead
    # of the failure penalty above, at every difficulty.
    altitude_penalty: float = 0.01
    altitude_deadband_m: float = 1.0
    below_route_factor: float = 3.0
    altitude_penalty_max: float = 0.1
    altitude_failure_penalty: float = 30.0
    # Sideways, also at every difficulty: each step costs lateral_penalty per
    # metre off the route horizontally beyond the deadband, capped per step.
    # Without it nothing pulled the vehicle back once it was past the pass
    # reward's reach (~5 m), and it settled ~9 m off, parallel to the route.
    lateral_penalty: float = 0.005
    lateral_deadband_m: float = 1.0
    lateral_penalty_max: float = 0.05
    # The early heading reward rewards ground velocity toward the route point
    # this many past the target, so turning back to the route scores. 0 is
    # the old reward (along the route segment), which made flying parallel to
    # the route, off to one side, score best.
    heading_lookahead_points: int = 1
    solver_iterations: int = 10
    solver_ls_iterations: int = 5


# Failure reasons, in the order they follow `completed` in step()'s metrics.
ENDING_REASONS = ("ground", "cross_track", "vertical", "nonfinite")

# Checkpoints saved before the route fields existed trained on 513 points 2 m apart.
LEGACY_ROUTE = {"route_points": 513, "route_spacing_m": 2.0}
# Runs saved before the height and sideways terms keep their old rewards: a
# 25 m height band, no height or sideways penalty, the segment heading reward.
LEGACY_HEIGHT = {"max_vertical_m": 25.0, "altitude_penalty": 0.0,
                 "altitude_failure_penalty": 10.0, "start_height_noise_m": 0.3,
                 "lateral_penalty": 0.0, "heading_lookahead_points": 0}


def config_from_saved(env):
    """EnvConfig from a checkpoint's saved env dict, filling fields it predates
    and dropping ones since removed (such as envelope_grace_s)."""
    known = {f.name for f in fields(EnvConfig)}
    saved = {**LEGACY_ROUTE, **LEGACY_HEIGHT, **env}
    return EnvConfig(**{k: v for k, v in saved.items() if k in known})


def _batched(path):
    """MJX Warp keeps some Data fields (shared index arrays) without a batch axis."""
    return warp_types._BATCH_DIM["Data"].get(tree_path_to_attr_str(path), True)


def _map_batched(fn, *states):
    """fn over every per-environment leaf of batched States; shared leaves kept."""
    data = jax.tree_util.tree_map_with_path(
        lambda path, x, *ys: fn(x, *ys) if _batched(path) else x,
        *[s.data for s in states])
    rest = jax.tree.map(fn, *[s.replace(data=None) for s in states])
    return rest.replace(data=data)


def select(mask, new, old):
    """Per environment: new where mask is true, else old (batched States).

    Use this, not jax.tree.map(jp.where, ...): that breaks on MJX Warp Data.
    """
    return _map_batched(
        lambda x, y: jp.where(mask.reshape(mask.shape + (1,) * (x.ndim - mask.ndim)), y, x),
        old, new)


def repeat(states, count):
    """count copies of a batch of States, back to back along the batch axis."""
    return _map_batched(lambda x: jp.concatenate([x] * count), states)


@struct.dataclass
class State:
    data: Any
    key: Any
    sensors: Any
    bias: Any
    gps_buffer: Any
    gps_age: Any
    gps_new: Any
    history: Any
    brakes: Any
    brake_velocity: Any
    action: Any
    points: Any
    origin: Any
    target: Any  # index of the first route point not yet passed
    steps: Any
    difficulty: Any  # curriculum level this episode was started at
    last_point: Any  # index of this episode's final route point
    episode_return: Any
    obs: Any
    privileged: Any  # critic-only features (asymmetric critic); see _privileged


class ParamotorEnv:
    """Actions [-1,1]: thrust, brake (both sides), brake difference (+ = right). z is world-up.

    Measurements (state.sensors) are the XML sensordata, corrupted, followed by
    one barometric altitude. The noise/bias vectors carry three more entries,
    the attitude-estimate error, which rotates pod_quat instead of adding to it.
    Only channels with a real counterpart on the vehicle reach the policy; see
    sensors.csv.
    """

    def __init__(self, config=None):
        self.cfg = c = config or EnvConfig()
        self.physics = ParamotorMJX(c.solver_iterations, c.solver_ls_iterations,
                                    machine.load().get("warp_graph_mode"))
        self.m = self.physics.native
        dt = float(self.m.opt.timestep)
        for hz in (c.control_hz, c.sensor_hz, c.gps_hz, c.baro_hz):
            if hz <= 0 or not np.isclose(1 / (hz * dt), round(1 / (hz * dt))):
                raise ValueError(
                    "Sensor/control periods must be integer multiples of physics dt"
                )
        if c.sensor_hz % c.control_hz or c.sensor_hz % c.gps_hz or c.sensor_hz % c.baro_hz:
            raise ValueError("sensor_hz must be divisible by control_hz, gps_hz and baro_hz")
        if c.history_hz <= 0 or c.sensor_hz % c.history_hz:
            raise ValueError("sensor_hz must be divisible by history_hz")
        if c.path_kind not in routes.KINDS or c.history_seconds <= 0:
            raise ValueError("Invalid path kind or history length")
        if not 0 < c.thrust_max <= self.m.actuator_ctrlrange[self.physics.thrust, 1]:
            raise ValueError("thrust_max must fit the XML actuator range")
        if not 0 <= c.gps_dropout <= 1 or c.gps_latency_s < 0:
            raise ValueError("Invalid GPS dropout/latency")
        self.dt, self.control_dt = dt, 1 / c.control_hz
        self.substeps = round(self.control_dt / dt)
        self.sensor_stride = round(1 / c.sensor_hz / dt)
        self.gps_stride = round(1 / c.gps_hz / dt)
        self.baro_stride = round(1 / c.baro_hz / dt)
        self.history_ratio = c.sensor_hz // c.history_hz
        self.history_count = round(c.history_seconds * c.history_hz)
        if self.history_count < 1 or c.episode_seconds * c.control_hz < 1:
            raise ValueError(
                "History and episodes must contain at least one sample/step"
            )
        if not 0 <= c.initial_thrust <= c.thrust_max or c.launch_speed <= 0:
            raise ValueError(
                "Initial thrust must be in range and launch_speed must be positive"
            )
        if c.route_points < 2 or c.route_spacing_m <= 0:
            raise ValueError("A route needs at least two points and positive spacing")
        if not c.preview_index or any(int(x) != x or x < 0 for x in c.preview_index):
            raise ValueError("Preview offsets must be non-negative integers")
        if c.success_radius_m <= 0 or c.pass_sigma_m <= 0 or c.gps_noise_limit_m < 0:
            raise ValueError("Success radius must be positive, GPS limit non-negative")
        if c.progress_reward_per_m < 0:
            raise ValueError("Progress reward cannot be negative")
        if min(c.position_randomization_m, c.start_height_noise_m, c.envelope_reward, c.action_change_penalty) < 0:
            raise ValueError("Randomization, envelope reward and action penalty cannot be negative")
        self.latency_ticks = round(c.gps_latency_s * c.sensor_hz)
        self.episode_steps = round(c.episode_seconds * c.control_hz)
        self.slices = {
            self.m.sensor(i).name: slice(int(a), int(a + n))
            for i, (a, n) in enumerate(zip(self.m.sensor_adr, self.m.sensor_dim))
        }
        self.gps_indices = jp.array(
            [
                i
                for name in ("pod_pos", "pod_vel")
                for i in range(self.slices[name].start, self.slices[name].stop)
            ]
        )
        # Measurement layout: sensordata, then baro; error vector adds attitude.
        sd = self.m.nsensordata
        self.baro = sd
        self.n_meas = sd + 1
        self.att = slice(self.n_meas, self.n_meas + 3)
        self.n_error = self.n_meas + 3

        def span(name):
            return list(range(self.slices[name].start, self.slices[name].stop))

        pos = span("pod_pos")
        # sensors.csv channel -> entries of the error vector it corrupts.
        self.channels = {
            "gyro": span("gyro"),
            "accel": span("accel"),
            "mag": span("mag"),
            "attitude_roll_pitch": [self.n_meas, self.n_meas + 1],
            "attitude_yaw": [self.n_meas + 2],
            "baro_alt": [self.baro],
            "gps_pos_xy": pos[:2],
            "gps_vel": span("pod_vel"),
            "prop_omega": span("prop_omega"),
            "arm_pos": span("arm_pos_L") + span("arm_pos_R"),
        }
        self.noise = self._sensor_vector(c.noise_std)
        self.bias_scale = self._sensor_vector(c.bias_std)
        self.walk = self._sensor_vector(c.bias_walk_std)
        self.frame_size = int(self._frame(
            jp.zeros(self.n_meas).at[self.slices["pod_quat"].start].set(1.0),
            jp.zeros(3), jp.zeros(()), jp.array(True)).size)
        self.obs_size = self.history_count * self.frame_size + len(c.preview_index) * 3 + 5
        # The launch state depends only on the config, never on the reset key:
        # build it once here instead of re-running the forward pass every reset.
        # Critic-only features: per-strip incidence (S-1), per-row chord (S),
        # and 24 scalars, see _privileged.
        rest = np.asarray(self.physics.rest)
        self.chord0 = jp.array(np.linalg.norm(rest[:, self.physics.le] - rest[:, -1], axis=-1))
        self.pod_site = self.m.site("pod_com").id
        self.tendon_max = jp.array(self.m.tendon_range[:, 1])
        self.privileged_size = 2 * rest.shape[0] - 1 + 24 + self.m.ntendon + 1
        self.launch = jax.jit(lambda: self.physics.initial(
            c.launch_speed, c.altitude, c.initial_thrust))()

    def route(self, key, difficulty, kind):
        """A route at this config's point count and spacing."""
        return routes.make_path(key, difficulty, kind, self.cfg.route_points, self.cfg.route_spacing_m)

    def _sensor_vector(self, mapping):
        unknown = set(mapping) - set(self.channels)
        if unknown:
            raise ValueError(f"Unknown sensor channels: {unknown}")
        out = np.zeros(self.n_error)
        for name, value in mapping.items():
            out[self.channels[name]] = value
        if np.any(out < 0) or not np.all(np.isfinite(out)):
            raise ValueError("Noise standard deviations must be finite and nonnegative")
        return jp.array(out)

    def _frame(self, values, origin, age, fresh):
        """One observation frame: only channels the real vehicle measures.

        Vectors are in the heading frame (world rotated by the estimated yaw), as
        is the route preview, so nothing depends on which way the vehicle points.
        Absolute horizontal position and the compass are left out for that reason.
        """
        s = self.slices
        rotation = quat_mat(values[s["pod_quat"]])
        heading = heading_mat(rotation)
        tilt = heading.T @ rotation  # estimated attitude with yaw removed
        # Accelerometer specific force (gravity included) rotated into world.
        accel = rotation @ values[s["accel"]]
        return jp.concatenate((
            tilt[:, 0], tilt[:, 1],                       # 6-D roll/pitch
            values[s["gyro"]] / 3.0,                      # body rates
            heading.T @ accel / 20.0,
            heading.T @ values[s["pod_vel"]] / 10.0,      # GPS velocity
            values[s["prop_omega"]] / 1000.0,
            values[s["arm_pos_L"]] / 3.0,
            values[s["arm_pos_R"]] / 3.0,
            (values[self.baro, None] - origin[2]) / 20.0,  # barometric altitude
            jp.array([jp.minimum(age, 10.0) / 2.0, fresh.astype(jp.float32)]),
        ))

    def _observation(self, state):
        # Guidance uses measured position/orientation, never true aircraft pose.
        # GNSS gives x/y; the barometer gives altitude.
        pos = state.sensors[self.slices["pod_pos"]].at[2].set(state.sensors[self.baro])
        heading = heading_mat(quat_mat(state.sensors[self.slices["pod_quat"]]))
        offsets = jp.array(self.cfg.preview_index)
        goals = routes.preview(state.points, state.target, offsets) - pos
        local = goals @ heading  # rows: heading.T @ goal
        preview = jp.clip(
            local / (jp.maximum(offsets, 1) * self.cfg.route_spacing_m)[:, None], -3, 3
        ).ravel()
        # Known command-filter states, not unmeasured physical servo states.
        controls = jp.concatenate((state.action, state.brake_velocity / 10))
        obs = jp.concatenate((state.history.ravel(), preview, controls))
        return state.replace(obs=jp.clip(jp.nan_to_num(obs), -10, 10),
                             privileged=self._privileged(state))

    def _privileged(self, state):
        """Ground truth the real vehicle cannot measure, for the critic only.

        The critic exists only in training, so it may see the canopy and true
        state; the policy never does. Vectors are in the TRUE heading frame.
        """
        d, ph, s = state.data, self.physics, self.slices
        heading = heading_mat(quat_mat(d.sensordata[s["pod_quat"]]))
        pod, pod_v = d.site_xpos[self.pod_site], d.sensordata[s["pod_vel"]]
        P, V = ph.vertices(d)
        mass = ph.vertex_mass
        com = (mass[..., None] * P).sum((0, 1)) / mass.sum()
        vel = (mass[..., None] * V).sum((0, 1)) / mass.sum()
        R, area, _ = aero.cell_frames(P[:, ph.le:], jp)
        _, _, omega, frame, _, _ = aero.canopy_state(P, V, mass, R, area, jp)
        tilt = heading.T @ frame                     # canopy axes, heading frame
        chord = jp.linalg.norm(P[:, ph.le] - P[:, -1], axis=-1) / self.chord0
        _, _, closest = routes.nearest_on_segment(state.points, state.target, pod)
        out = jp.concatenate((
            ph.raw_alpha(d) / 0.5,                   # incidence of each strip, rad
            chord - 1.0,                             # chord retention per span row
            tilt[:, 0], tilt[:, 2],                  # canopy chord and normal axes (6)
            heading.T @ (com - pod),                 # canopy offset from the pod, m (3)
            heading.T @ (vel - pod_v) / 5.0,         # canopy velocity relative to pod (3)
            heading.T @ omega / 3.0,                 # canopy angular velocity (3)
            heading.T @ pod_v / 10.0,                # true air velocity, no wind (3)
            jp.linalg.norm(pod_v)[None] / 10.0,      # airspeed (1)
            heading.T @ (pod - closest) / 10.0,      # true route error (3)
            pod[2:3] / 40.0,                         # height above ground (1)
            (state.steps / self.episode_steps)[None],  # episode time used (1)
            d.ten_length / self.tendon_max - 1.0,    # line/brake slack (<0 slack)
            self._distance_to_go(pod, state.points, state.target, state.last_point)[None] / 1000.0,
        ))
        return jp.clip(jp.nan_to_num(out), -10, 10)

    def reset(self, key, difficulty=0.0):
        key, pk, nk, bk = jax.random.split(key, 4)
        points = self.route(pk, difficulty, self.cfg.path_kind)
        data = self.launch
        origin = data.site_xpos[self.m.site("pod_com").id]
        points = points + origin
        # Preserve rigging geometry: shift the pod and the whole skin together.
        offset = jax.random.uniform(nk, (3,), minval=-1.0, maxval=1.0)
        offset = offset * jp.array([0.0, self.cfg.position_randomization_m, self.cfg.start_height_noise_m])
        qpos = self.physics.translate(data.qpos, offset)
        data = self.physics.forward(data.replace(qpos=qpos))
        bias = jax.random.normal(bk, (self.n_error,)) * self.bias_scale
        key, nk = jax.random.split(key)
        sensors = self._corrupt(data.sensordata, bias, nk)
        zero = jp.zeros((), data.qpos.dtype)
        integer_zero = jp.zeros((), jp.int32)
        frame = self._frame(sensors, origin, zero, jp.array(True))
        history = jp.repeat(frame[None, :], self.history_count, axis=0)
        gps = jp.repeat(
            sensors[self.gps_indices][None, :], self.latency_ticks + 1, axis=0
        )
        action = jp.array(
            [2 * self.cfg.initial_thrust / self.cfg.thrust_max - 1, -1.0, 0.0]
        )
        state = State(
            data=data,
            key=key,
            sensors=sensors,
            bias=bias,
            gps_buffer=gps,
            gps_age=zero,
            gps_new=jp.array(True),
            history=history,
            brakes=jp.zeros(2),
            brake_velocity=jp.zeros(2),
            action=action,
            points=points,
            origin=origin,
            target=jp.ones((), jp.int32),  # point 0 is the launch position
            steps=integer_zero,
            difficulty=jp.asarray(difficulty, jp.float32),
            last_point=self._last_point(difficulty),
            episode_return=zero,
            obs=jp.zeros(self.obs_size),
            privileged=jp.zeros(self.privileged_size),
        )
        return self._observation(state)

    def _corrupt(self, truth, bias, key):
        """sensordata -> measurements: additive noise/bias, baro, estimated attitude."""
        error = bias + self.noise * jax.random.normal(key, (self.n_error,))
        gps = self.channels["gps_pos_xy"]
        limit = self.cfg.gps_noise_limit_m
        error = error.at[jp.array(gps)].set(jp.clip(error[jp.array(gps)], -limit, limit))
        baro = truth[self.slices["pod_pos"]][2:3]
        values = jp.concatenate((truth, baro)) + error[: self.n_meas]
        # Estimator error is a small world-frame rotation of the true attitude.
        q = quat_mul(rotvec_quat(error[self.att]), truth[self.slices["pod_quat"]])
        q = q / jp.maximum(jp.linalg.norm(q), 1e-8)
        q = jp.where(q[0] < 0, -q, q)  # avoid quaternion sign jumps in history
        return values.at[self.slices["pod_quat"]].set(q)

    def _last_point(self, difficulty):
        """Final route point for an episode at this curriculum level: short
        routes first, the full route at difficulty 1."""
        full, short = self.cfg.route_points - 1, min(self.cfg.min_route_points, self.cfg.route_points - 1)
        return jp.round(short + (full - short) * jp.clip(difficulty, 0.0, 1.0)).astype(jp.int32)

    def _distance_to_go(self, pos, points, target, last=None):
        """Metres left: to the target point, then along the rest of the route.

        Continuous when the target advances (up to the success radius), so its
        change rewards closing on a point without penalizing passing it.
        """
        last = self.cfg.route_points - 1 if last is None else last
        here = jp.linalg.norm(pos - points[jp.minimum(target, last)])
        return here + jp.maximum(last - target, 0) * self.cfg.route_spacing_m

    def _crossing(self, points, target, start, end, last=None):
        """Did this step cross the target's plane, and how close did it come?

        The plane passes through the target point, perpendicular to the route
        tangent there. The miss distance is the closest approach to the point
        along this step's straight-line motion, start -> end.
        """
        last = self.cfg.route_points - 1 if last is None else last
        index = jp.minimum(target, last)
        point = points[index]
        tangent = points[jp.minimum(index + 1, last)] - points[index - 1]
        tangent = tangent / jp.maximum(jp.linalg.norm(tangent), 1e-9)
        crossed = (target <= last) & (jp.dot(end - point, tangent) >= 0)
        motion = end - start
        u = jp.clip(jp.dot(point - start, motion) / jp.maximum(jp.dot(motion, motion), 1e-12), 0, 1)
        return crossed, jp.linalg.norm(start + u * motion - point)

    def _sample(self, state, tick):
        key, nk, bk, dk = jax.random.split(state.key, 4)
        bias = state.bias + self.walk * jp.sqrt(
            1 / self.cfg.sensor_hz
        ) * jax.random.normal(bk, state.bias.shape)
        values = self._corrupt(state.data.sensordata, bias, nk)
        buffer = jp.concatenate(
            (state.gps_buffer[1:], values[self.gps_indices][None, :]), axis=0
        )
        fix = (tick % self.gps_stride == 0) & (
            jax.random.uniform(dk) >= self.cfg.gps_dropout
        )
        values = values.at[self.gps_indices].set(
            jp.where(fix, buffer[0], state.sensors[self.gps_indices])
        )
        # The barometer reports at baro_hz; between reports the last value holds.
        values = values.at[self.baro].set(
            jp.where(tick % self.baro_stride == 0, values[self.baro], state.sensors[self.baro])
        )
        age = jp.where(
            fix,
            self.latency_ticks / self.cfg.sensor_hz,
            state.gps_age + 1 / self.cfg.sensor_hz,
        )
        frame = self._frame(values, state.origin, age, fix)
        # History keeps every history_ratio-th sample (history_hz).
        push = (tick // self.sensor_stride) % self.history_ratio == 0
        history = jp.where(
            push,
            jp.concatenate((state.history[1:], frame[None, :]), axis=0),
            state.history,
        )
        return state.replace(
            key=key,
            sensors=values,
            bias=bias,
            gps_buffer=buffer,
            gps_age=age,
            gps_new=fix,
            history=history,
        )

    def step(self, state, action):
        action = jp.clip(action, -1, 1)
        # Symmetric brake in [0, 1] and a differential (positive = right brake),
        # mixed into per-side commands and clipped to the line travel.
        brake, diff = 0.5 * (action[1] + 1), action[2]
        target = 3.0 * jp.clip(jp.array([brake - diff, brake + diff]), 0.0, 1.0)
        thrust = 0.5 * (action[0] + 1) * self.cfg.thrust_max

        def substep(s, i):
            brakes, rate = smooth_brakes(
                s.brakes, s.brake_velocity, target, self.dt, jp
            )
            data = self.physics.command(s.data, thrust, brakes)
            s = s.replace(
                data=self.physics.step(data),
                brakes=brakes, brake_velocity=rate,
            )
            tick = s.steps * self.substeps + i + 1

            def sample(x):
                # Refresh sensors at the actual sample time, after integration.
                x = x.replace(data=self.physics.forward(x.data))
                return self._sample(x, tick)

            # All environments share the fast-sensor phase within a control step.
            # An unbatched predicate keeps vmap from evaluating both cond branches.
            s = jax.lax.cond((i + 1) % self.sensor_stride == 0, sample, lambda x: x, s)
            return s, None

        previous_action = state.action
        start = state.data.site_xpos[self.m.site("pod_com").id]
        before = self._distance_to_go(start, state.points, state.target, state.last_point)
        state, _ = jax.lax.scan(substep, state, jp.arange(self.substeps))
        data = state.data  # the last substep always refreshed the sensors
        pos = data.site_xpos[self.m.site("pod_com").id]
        last = state.last_point
        # The target is passed by crossing its plane (true position), however far
        # off; the reward is graded by the miss distance, so a miss never stalls.
        reached, miss = self._crossing(state.points, state.target, start, pos, last)
        excess = jp.maximum(miss - self.cfg.success_radius_m, 0) / self.cfg.pass_sigma_m
        pass_reward = jp.where(reached, jp.exp(-excess**2), 0.0)
        target = state.target + reached
        # Tracking error against the segment into the target, for metrics and failure.
        index, fraction, closest = routes.nearest_on_segment(state.points, target, pos)
        progress = (index + fraction) * self.cfg.route_spacing_m
        error = pos - closest
        lateral = jp.linalg.norm(error[:2])
        vertical = jp.abs(error[2])
        raw = self.physics.raw_alpha(data)
        within = (raw >= self.physics.p["alpha_min"]) & (raw <= self.physics.p["alpha_max"])
        outside = ~jp.all(within)  # for metrics only; it no longer ends the episode
        finite = jp.all(jp.isfinite(data.qpos)) & jp.all(jp.isfinite(data.qvel))
        # Why an episode fails; more than one can trip on the same step.
        reasons = jp.stack((
            pos[2] < 0,                              # ground
            lateral > self.cfg.max_cross_track_m,    # off route sideways
            vertical > self.cfg.max_vertical_m,      # off route vertically
            ~finite,                                 # non-finite physics
        ))
        failed = jp.any(reasons)
        completed = target > last
        terminated = failed | completed
        truncated = state.steps + 1 >= self.episode_steps
        # Graded pass reward, plus progress shaping: the drop in distance-to-go.
        # A plain difference with no terminal term, so over a flight it sums to
        # the distance gained and a crash earns nothing extra.
        closed = before - self._distance_to_go(pos, state.points, target, last)
        progress_reward = self.cfg.progress_reward_per_m * closed
        # Only while closing on the route, so a slow glide cannot farm it.
        envelope_reward = self.cfg.envelope_reward * jp.mean(within) * (closed > 0)
        # Early helpers, faded out by the curriculum: fly along the route
        # (true ground track against the segment into the target), stay at its
        # height, and stay alive.
        early = 1.0 - jp.clip(state.difficulty, 0.0, 1.0)
        if self.cfg.heading_lookahead_points:  # toward a point ahead: corrects offsets
            aim = state.points[jp.minimum(target + self.cfg.heading_lookahead_points, last)] - pos
        else:  # along the segment: indifferent to a sideways offset
            aim = state.points[jp.minimum(target, last)] - state.points[jp.maximum(jp.minimum(target, last) - 1, 0)]
        ground = data.sensordata[self.slices["pod_vel"]][:2]
        heading = jp.dot(ground, aim[:2]) / jp.maximum(
            jp.linalg.norm(ground) * jp.linalg.norm(aim[:2]), 1e-6)
        dense = self.cfg.dense_reward * (heading - jp.clip(vertical / 10.0, 0.0, 1.0))
        # Height off the route, worse below it (error[2] < 0), at every difficulty.
        off = jp.maximum(vertical - self.cfg.altitude_deadband_m, 0.0)
        height = self.cfg.altitude_penalty * off * jp.where(
            error[2] < 0, self.cfg.below_route_factor, 1.0)
        sideways = self.cfg.lateral_penalty * jp.maximum(lateral - self.cfg.lateral_deadband_m, 0.0)
        reward = pass_reward + progress_reward + envelope_reward + early * (
            dense + self.cfg.alive_reward) - jp.minimum(height, self.cfg.altitude_penalty_max) - jp.minimum(
            sideways, self.cfg.lateral_penalty_max) - (
            self.cfg.action_change_penalty * jp.sum((action - previous_action) ** 2))
        # Failure costs little while episodes earn little, the full penalty later;
        # leaving the height band (or hitting the ground) costs the most throughout.
        penalty = self.cfg.failure_penalty_start + (
            self.cfg.failure_penalty - self.cfg.failure_penalty_start) * (1.0 - early)
        reward = jp.nan_to_num(reward, nan=-penalty, posinf=-penalty, neginf=-penalty)
        penalty = jp.where(reasons[0] | reasons[2], self.cfg.altitude_failure_penalty, penalty)
        reward = jp.where(failed, -penalty, reward) + jp.where(completed, 10.0, 0.0)
        state = state.replace(
            data=data,
            steps=state.steps + 1,
            action=action,
            target=target,
            episode_return=state.episode_return + reward,
        )
        state = self._observation(state)
        lateral = jp.nan_to_num(
            lateral, nan=self.cfg.max_cross_track_m, posinf=self.cfg.max_cross_track_m
        )
        vertical = jp.nan_to_num(
            vertical, nan=self.cfg.max_vertical_m, posinf=self.cfg.max_vertical_m
        )
        progress = jp.nan_to_num(progress)
        metrics = jp.array(
            [
                lateral,
                vertical,
                progress,
                outside.astype(jp.float32),
                state.episode_return,
                failed.astype(jp.float32),
                completed.astype(jp.float32),
                *reasons.astype(jp.float32),         # ENDING_REASONS, in order
            ]
        )
        return state, reward, terminated, truncated, metrics
