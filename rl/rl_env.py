"""Batched-friendly MJX path following. No Gym wrappers or global callbacks."""

from dataclasses import dataclass, field
import math
from typing import Any
import jax
import jax.numpy as jp
from flax import struct
import mujoco
import numpy as np

from rl import routes
from model import sensor_spec
from model.paramotor_control import smooth_brakes
from mjx.paramotor_mjx import ParamotorMJX


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


@dataclass
class EnvConfig:
    control_hz: int = 25
    sensor_hz: int = 100
    gps_hz: int = 5
    baro_hz: int = 20  # BMP581 output rate; held between samples like GPS
    history_hz: int = 25  # observation history is downsampled from sensor_hz
    history_seconds: float = 1.0
    episode_seconds: float = 60.0
    thrust_max: float = 1.0  # hardware thrust clamp; equals the XML ctrlrange
    initial_thrust: float = 0.8
    launch_speed: float = 6.0
    altitude: float = 40.0
    preview_m: tuple = (20.0, 40.0, 80.0)  # look-ahead points along the route
    route_points: int = 101  # route stored as route_points, route_spacing_m apart (~1 km)
    route_spacing_m: float = 10.0
    path_kind: str = "random"
    curriculum: bool = True
    # Datasheet values from sensors.csv, keyed by its channel names.
    noise_std: dict = field(default_factory=lambda: sensor_spec.defaults("noise_std"))
    bias_std: dict = field(default_factory=lambda: sensor_spec.defaults("bias_std"))
    bias_walk_std: dict = field(default_factory=dict)  # channel units / sqrt(second)
    gps_latency_s: float = 0.0
    gps_dropout: float = 0.0
    position_randomization_m: float = 1.0
    max_cross_track_m: float = 35.0
    envelope_grace_s: float = 1.0
    solver_iterations: int = 10
    solver_ls_iterations: int = 5


# Checkpoints saved before the route fields existed trained on 513 points 2 m apart.
LEGACY_ROUTE = {"route_points": 513, "route_spacing_m": 2.0}


def config_from_saved(env):
    """EnvConfig from a checkpoint's saved env dict, filling fields it predates."""
    return EnvConfig(**{**LEGACY_ROUTE, **env})


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
    arc: Any
    origin: Any
    path_index: Any
    observed_index: Any
    progress: Any
    steps: Any
    envelope_time: Any
    episode_return: Any
    obs: Any


class ParamotorEnv:
    """Actions [-1,1]: thrust, left brake, right brake. z is world-up.

    Measurements (state.sensors) are the XML sensordata, corrupted, followed by
    one barometric altitude. The noise/bias vectors carry three more entries,
    the attitude-estimate error, which rotates pod_quat instead of adding to it.
    Only channels with a real counterpart on the vehicle reach the policy; see
    sensors.csv.
    """

    def __init__(self, config=None):
        self.cfg = c = config or EnvConfig()
        self.physics = ParamotorMJX(c.solver_iterations, c.solver_ls_iterations)
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
        # Projection searches a fixed distance around the last match (about
        # 6 m back, 24 m ahead), whatever the route's point spacing.
        self.project_back = max(1, math.ceil(6.0 / c.route_spacing_m))
        self.project_ahead = max(1, math.ceil(24.0 / c.route_spacing_m))
        if not c.preview_m or any(x <= 0 for x in c.preview_m):
            raise ValueError("Preview distances must be positive")
        if c.position_randomization_m < 0 or c.envelope_grace_s < 0:
            raise ValueError("Randomization and envelope grace cannot be negative")
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
        self.obs_size = self.history_count * self.frame_size + len(c.preview_m) * 3 + 5
        # The launch state depends only on the config, never on the reset key:
        # build it once here instead of re-running the forward pass every reset.
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
        """One observation frame: only channels the real vehicle measures."""
        rotation = quat_mat(values[self.slices["pod_quat"]])
        s = self.slices
        return jp.concatenate((
            rotation[:, 0], rotation[:, 1],               # 6-D estimated attitude
            values[s["gyro"]] / 3.0,
            values[s["accel"]] / 20.0,
            values[s["mag"]] / 0.5,
            (values[s["pod_pos"]][:2] - origin[:2]) / 100.0,  # GPS horizontal
            values[s["pod_vel"]] / 10.0,
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
        quat = state.sensors[self.slices["pod_quat"]]
        w, x, y, z = quat
        yaw = jp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        i, s, _, _ = routes.project(state.points, state.arc, pos, state.observed_index,
                                    self.project_back, self.project_ahead)
        goals = (
            routes.preview(state.points, state.arc, s, jp.array(self.cfg.preview_m))
            - pos
        )
        co, si = jp.cos(yaw), jp.sin(yaw)
        local = goals @ jp.array([[co, -si, 0], [si, co, 0], [0, 0, 1]])
        preview = jp.clip(local / jp.array(self.cfg.preview_m)[:, None], -3, 3).ravel()
        # Known command-filter states, not unmeasured physical servo states.
        controls = jp.concatenate((state.action, state.brake_velocity / 10))
        obs = jp.concatenate((state.history.ravel(), preview, controls))
        return state.replace(observed_index=i, obs=jp.clip(jp.nan_to_num(obs), -10, 10))

    def reset(self, key, difficulty=0.0):
        key, pk, nk, bk = jax.random.split(key, 4)
        points, arc = self.route(pk, difficulty, self.cfg.path_kind)
        data = self.launch
        origin = data.site_xpos[self.m.site("pod_com").id]
        points = points + origin
        # Preserve rigging geometry: shift both free bodies together.
        offset = jax.random.uniform(nk, (3,), minval=-1.0, maxval=1.0)
        offset = offset * jp.array([0.0, 1.0, 0.3]) * self.cfg.position_randomization_m
        qpos = data.qpos
        for a in self.physics.free_qpos:
            qpos = qpos.at[a : a + 3].add(offset)
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
            [2 * self.cfg.initial_thrust / self.cfg.thrust_max - 1, -1.0, -1.0]
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
            arc=arc,
            origin=origin,
            path_index=integer_zero,
            observed_index=integer_zero,
            progress=zero,
            steps=integer_zero,
            envelope_time=zero,
            episode_return=zero,
            obs=jp.zeros(self.obs_size),
        )
        return self._observation(state)

    def _corrupt(self, truth, bias, key):
        """sensordata -> measurements: additive noise/bias, baro, estimated attitude."""
        error = bias + self.noise * jax.random.normal(key, (self.n_error,))
        baro = truth[self.slices["pod_pos"]][2:3]
        values = jp.concatenate((truth, baro)) + error[: self.n_meas]
        # Estimator error is a small world-frame rotation of the true attitude.
        q = quat_mul(rotvec_quat(error[self.att]), truth[self.slices["pod_quat"]])
        q = q / jp.maximum(jp.linalg.norm(q), 1e-8)
        q = jp.where(q[0] < 0, -q, q)  # avoid quaternion sign jumps in history
        return values.at[self.slices["pod_quat"]].set(q)

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
        target = 1.5 * (action[1:] + 1)
        thrust = 0.5 * (action[0] + 1) * self.cfg.thrust_max

        def substep(s, i):
            brakes, rate = smooth_brakes(
                s.brakes, s.brake_velocity, target, self.dt, jp
            )
            data = self.physics.command(s.data, thrust, brakes)
            data = self.physics.forward(data, with_sensors=False)
            s = s.replace(
                data=self.physics.integrate(data), brakes=brakes, brake_velocity=rate
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

        previous_action, previous_progress = state.action, state.progress
        state, _ = jax.lax.scan(substep, state, jp.arange(self.substeps))
        data = state.data  # the last substep always refreshed the sensors
        pos = data.site_xpos[self.m.site("pod_com").id]
        index, progress, closest, tangent = routes.project(
            state.points, state.arc, pos, state.path_index,
            self.project_back, self.project_ahead,
        )
        error = pos - closest
        lateral = jp.linalg.norm(error[:2])
        vertical = jp.abs(error[2])
        raw = self.physics.raw_alpha(data)
        outside = jp.any(
            (raw < self.physics.p["alpha_min"]) | (raw > self.physics.p["alpha_max"])
        )
        envelope = jp.where(outside, state.envelope_time + self.control_dt, 0.0)
        finite = jp.all(jp.isfinite(data.qpos)) & jp.all(jp.isfinite(data.qvel))
        canopy_below = (
            data.xipos[self.physics.canopy, 2] < data.xipos[self.physics.pod, 2]
        )
        failed = (
            (~finite)
            | (pos[2] < 0)
            | (lateral > self.cfg.max_cross_track_m)
            | (vertical > 25)
            | canopy_below
            | (envelope > self.cfg.envelope_grace_s)
        )
        completed = progress >= state.arc[-1] - 2
        terminated = failed | completed
        truncated = state.steps + 1 >= self.episode_steps
        delta = jp.clip(
            progress - previous_progress,
            -2 * self.cfg.launch_speed * self.control_dt,
            2 * self.cfg.launch_speed * self.control_dt,
        )
        gate = jp.exp(-0.5 * ((lateral / 5) ** 2 + (vertical / 3) ** 2))
        reward = (
            gate * delta / (self.cfg.launch_speed * self.control_dt)
            - 0.15 * (lateral / 5) ** 2
            - 0.15 * (vertical / 3) ** 2
            - 0.02 * jp.sum((action - previous_action) ** 2)
        )
        reward = jp.nan_to_num(reward, nan=-10.0, posinf=-10.0, neginf=-10.0)
        reward = jp.where(failed, -10.0, reward) + jp.where(completed, 10.0, 0.0)
        state = state.replace(
            data=data,
            steps=state.steps + 1,
            action=action,
            path_index=index,
            progress=progress,
            envelope_time=envelope,
            episode_return=state.episode_return + reward,
        )
        state = self._observation(state)
        lateral = jp.nan_to_num(
            lateral, nan=self.cfg.max_cross_track_m, posinf=self.cfg.max_cross_track_m
        )
        vertical = jp.nan_to_num(vertical, nan=25.0, posinf=25.0)
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
            ]
        )
        return state, reward, terminated, truncated, metrics
