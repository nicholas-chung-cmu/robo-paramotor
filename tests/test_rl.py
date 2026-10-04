"""Regression checks for physics parity, sensor timing, guidance, and PPO math."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import numpy as np
import pytest

from rl import routes
from model import sensor_spec
from model.paramotor_aero import ParamotorAero
from model.paramotor_control import smooth_brakes
from model.paramotor_params import PEEK_1M
from rl.rl_env import repeat, select
from rl.rl_env import EnvConfig, ParamotorEnv, quat_mat, quat_mul, rotvec_quat
from rl.train import advantages, log_probability


@pytest.fixture(scope="module")
def env():
    mujoco.set_mjcb_passive(None)
    # Noise off: these tests check timing and layout, not noise statistics.
    return ParamotorEnv(EnvConfig(episode_seconds=0.2, position_randomization_m=0.0,
                                  noise_std={}, bias_std={}))


@pytest.fixture(scope="module")
def initial(env):
    return jax.jit(env.reset)(jax.random.PRNGKey(4))


def _rigid_state(m, d, velocity, rates, tilt):
    """Pod and skin moving together: same translation, rotation and attitude."""
    from model.paramotor_aero import CanopyMesh
    from scipy.spatial.transform import Rotation
    mesh = CanopyMesh(m)
    rot = Rotation.from_rotvec(tilt)
    pod = m.joint("pod_free")
    a, b = pod.qposadr[0], pod.dofadr[0]
    d.qpos[a + 2] += 40
    q = rot.as_quat()  # x y z w
    d.qpos[a + 3 : a + 7] = [q[3], q[0], q[1], q[2]]
    d.qvel[b : b + 3] = velocity
    d.qvel[b + 3 : b + 6] = rates  # local frame for a free joint
    w = rot.apply(rates)
    centre = mesh.rest.reshape(-1, 3).mean(axis=0)
    P = rot.apply(mesh.rest.reshape(-1, 3) - centre) + centre + [0, 0, 40]
    d.qpos[mesh.qpos.reshape(-1, 3)] = P - mesh.rest.reshape(-1, 3)
    d.qvel[mesh.dof.reshape(-1, 3)] = np.asarray(velocity) + np.cross(w, P - (centre + [0, 0, 40]))


def test_native_aero_and_acceleration_parity(env):
    physics = env.physics
    m = physics.native
    aero = ParamotorAero(m, PEEK_1M)
    calculate = jax.jit(physics.forward)
    for velocity, rates in [
        ([6.0, 0.0, -0.3], [0.1, -0.2, 0.3]),
        ([5.0, 0.8, -1.0], [-0.3, 0.2, -0.1]),
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ]:
        d = mujoco.MjData(m)
        # A nonzero attitude, unequal arm positions and nonzero controls.
        _rigid_state(m, d, velocity, rates, [0.1, -0.06, 0.2])
        d.qpos[m.joint("arm_L").qposadr[0]] = 0.4
        d.ctrl[:] = [0.8, 0.2, 0.1]
        try:
            mujoco.set_mjcb_passive(aero)
            mujoco.mj_forward(m, d)
            expected = aero.wrench.copy()
        finally:
            mujoco.set_mjcb_passive(None)
        result = calculate(mjx.put_data(m, d, impl="warp"))
        np.testing.assert_allclose(result.xfrc_applied, expected, atol=2e-5, rtol=2e-4)
        # Compare accelerations as force (x each dof's inertia): canopy vertices
        # weigh under a gram, so float32 force rounding of 1e-4 N alone moves
        # their accelerations by ~0.2 m/s^2.
        np.testing.assert_allclose(
            (np.asarray(result.qacc) - d.qacc) * m.dof_M0, 0.0, atol=5e-4)
        np.testing.assert_allclose(
            result.sensordata, d.sensordata, atol=0.01, rtol=3e-3
        )


def test_native_short_rollout_parity(env):
    physics = env.physics
    m = physics.native
    start = jax.jit(physics.initial)(
        env.cfg.launch_speed, env.cfg.altitude, env.cfg.initial_thrust)
    d = mujoco.MjData(m)
    d.qpos[:], d.qvel[:], d.ctrl[:] = start.qpos, start.qvel, start.ctrl
    aero = ParamotorAero(m, PEEK_1M)
    try:
        mujoco.set_mjcb_passive(aero)
        for _ in range(400):
            mujoco.mj_step(m, d)
    finally:
        mujoco.set_mjcb_passive(None)
    advance = jax.jit(
        lambda data: jax.lax.fori_loop(
            0, 400, lambda _, x: physics.step(x), data
        )
    )
    result = advance(start)
    np.testing.assert_allclose(result.qpos, d.qpos, atol=1e-3, rtol=1e-4)
    np.testing.assert_allclose(result.qvel, d.qvel, atol=0.05, rtol=5e-3)


def test_brake_step_reaches_99_percent_at_one_second():
    value, rate = smooth_brakes(np.zeros(2), np.zeros(2), np.array([3.0, 0.0]), 1.0)
    assert 0.989 < value[0] / 3 < 0.991
    assert value[1] == 0 and rate[1] == 0
    # Closed-form update must not depend on subdivision.
    p, v = np.zeros(2), np.zeros(2)
    for _ in range(2000):
        p, v = smooth_brakes(p, v, np.array([3.0, 0.0]), 0.0005)
    np.testing.assert_allclose(p, value, atol=1e-12)
    np.testing.assert_allclose(v, rate, atol=1e-12)


def test_random_path_limits():
    p = routes.make_path(jax.random.PRNGKey(2))
    delta = np.diff(p, axis=0)
    horizontal = np.linalg.norm(delta[:, :2], axis=1)
    grade = delta[:, 2] / horizontal
    heading = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    assert abs(grade).max() <= 0.10001
    assert abs(np.diff(heading) / 10.0).max() <= 0.04001
    np.testing.assert_allclose(horizontal, 10.0, atol=1e-3)


def test_figure_eight_points_are_evenly_spaced():
    p = routes.make_path(jax.random.PRNGKey(0), kind="figure_eight")
    np.testing.assert_allclose(np.linalg.norm(np.diff(p, axis=0), axis=1), 10.0, rtol=0.03)


def test_target_segment_and_preview_clamp():
    straight = routes.make_path(jax.random.PRNGKey(0), kind="straight")
    i, fraction, closest = routes.nearest_on_segment(
        straight, jp.int32(4), jp.array([31.0, 3.0, -2.0])
    )
    assert int(i) == 3  # segment 30-40 m, into target point 4
    np.testing.assert_allclose([fraction, *closest], [0.1, 31.0, 0.0, 0.0], atol=1e-5)
    offsets = jp.array([2, 4, 8])
    np.testing.assert_allclose(
        routes.preview(straight, jp.int32(4), offsets)[:, 0], [60.0, 80.0, 120.0]
    )
    # Past the end the preview holds the last point.
    np.testing.assert_allclose(
        routes.preview(straight, jp.int32(98), offsets)[:, 0], [100 * 10.0] * 3
    )


def test_target_passes_at_its_plane_with_graded_reward(env, initial):
    assert int(initial.target) == 1
    pos = initial.data.site_xpos[env.m.site("pod_com").id]
    radius, sigma = env.cfg.success_radius_m, env.cfg.pass_sigma_m
    # Isolate the pass reward: no shaping, no alpha-range reward, and the
    # action is unchanged.
    old = env.cfg.progress_reward_per_m, env.cfg.envelope_reward
    env.cfg.progress_reward_per_m = env.cfg.envelope_reward = 0.0
    try:
        # The level, straight launch route runs along +x at 6 m/s (0.24 m per
        # step): put point 1 just ahead, so this step crosses its plane, offset
        # vertically by the intended miss distance.
        for miss in (0.5, 4.0, 8.0):
            shift = pos + jp.array([0.03, 0.0, miss]) - initial.points[1]
            state = initial.replace(points=initial.points + shift)
            new, reward, *_ = env.step(state, state.action)
            assert int(new.target) == 2
            expected = np.exp(-(max(miss - radius, 0) / sigma) ** 2)
            np.testing.assert_allclose(float(reward), expected, rtol=0.05, atol=0.01)
        # A point still ahead is not passed, however close.
        shift = pos + jp.array([1.0, 0.0, 0.0]) - initial.points[1]
        state = initial.replace(points=initial.points + shift)
        new, reward, *_ = env.step(state, state.action)
        assert int(new.target) == 1 and abs(float(reward)) < 1e-6
    finally:
        env.cfg.progress_reward_per_m, env.cfg.envelope_reward = old


def test_progress_shaping(env, initial):
    site = env.m.site("pod_com").id
    # Isolate the progress term: the alpha-range reward is checked separately.
    old, env.cfg.envelope_reward = env.cfg.envelope_reward, 0.0
    try:
        new, reward, *_ = env.step(initial, initial.action)
    finally:
        env.cfg.envelope_reward = old
    assert int(new.target) == 1  # no point passed, no action change
    before = env._distance_to_go(initial.data.site_xpos[site], initial.points, 1)
    after = env._distance_to_go(new.data.site_xpos[site], new.points, 1)
    assert float(before - after) > 0  # flying toward the route
    np.testing.assert_allclose(
        float(reward), env.cfg.progress_reward_per_m * float(before - after), rtol=1e-5)
    # Distance-to-go barely moves when the target advances at a route point.
    points = initial.points
    for t in (1, 5, env.cfg.route_points - 2):
        np.testing.assert_allclose(env._distance_to_go(points[t], points, t),
                                   env._distance_to_go(points[t], points, t + 1), atol=0.1)


def test_gps_hold_freshness_and_history(env, initial):
    sample = jax.jit(env._sample)
    s = initial
    gps0 = np.asarray(s.sensors[env.gps_indices]).copy()
    for i in range(1, 21):
        truth = initial.data.sensordata.at[env.gps_indices].set(i)
        truth = truth.at[env.slices["gyro"]].set(i)
        s = sample(
            s.replace(data=s.data.replace(sensordata=truth)), i * env.sensor_stride
        )
        np.testing.assert_allclose(s.sensors[env.slices["gyro"]], i)
        if i < 20:
            np.testing.assert_allclose(s.sensors[env.gps_indices], gps0)
            assert not bool(s.gps_new)
            np.testing.assert_allclose(s.gps_age, i * 0.01, atol=1e-6)
    np.testing.assert_allclose(s.sensors[env.gps_indices], 20.0)
    assert bool(s.gps_new) and float(s.gps_age) == 0.0
    np.testing.assert_allclose(s.history[-1, -2:], [0.0, 1.0])
    assert s.history.shape == (25, env.frame_size)  # 1 s at history_hz = 25
    assert set(env.slices) == {env.m.sensor(i).name for i in range(env.m.nsensor)}


def test_noise_reproducibility_and_units(env, initial):
    old = env.noise
    try:
        env.noise = env._sensor_vector({"gyro": [0.1, 0.2, 0.3]})
        zero = jp.zeros(env.n_error)
        one = env._corrupt(initial.data.sensordata, zero, jax.random.PRNGKey(5))
        two = env._corrupt(initial.data.sensordata, zero, jax.random.PRNGKey(5))
        other = env._corrupt(initial.data.sensordata, zero, jax.random.PRNGKey(6))
        np.testing.assert_array_equal(one, two)
        assert not np.allclose(one[env.slices["gyro"]], other[env.slices["gyro"]])
        np.testing.assert_allclose(
            one[env.slices["accel"]], initial.data.sensordata[env.slices["accel"]]
        )
        np.testing.assert_allclose(
            np.linalg.norm(one[env.slices["pod_quat"]]), 1.0, atol=1e-6
        )
    finally:
        env.noise = old
    with pytest.raises(ValueError):
        env._sensor_vector({"unknown": 1})
    with pytest.raises(ValueError):
        env._sensor_vector({"gyro": -1})


def test_gps_dropout_and_latency(env, initial):
    old_dropout, old_latency = env.cfg.gps_dropout, env.latency_ticks
    try:
        env.latency_ticks = 2
        s = initial.replace(
            gps_buffer=jp.repeat(initial.sensors[env.gps_indices][None], 3, axis=0)
        )
        for i in range(1, 21):
            s = s.replace(
                data=s.data.replace(
                    sensordata=s.data.sensordata.at[env.gps_indices].set(i)
                )
            )
            s = env._sample(s, i * env.sensor_stride)
        np.testing.assert_allclose(s.sensors[env.gps_indices], 18.0)
        np.testing.assert_allclose(s.gps_age, 0.02)
        env.cfg.gps_dropout = 1.0
        old = np.asarray(s.sensors[env.gps_indices]).copy()
        s = env._sample(s, 2 * env.gps_stride)
        np.testing.assert_array_equal(s.sensors[env.gps_indices], old)
        assert not bool(s.gps_new)
        assert float(s.gps_age) > 0.02
    finally:
        env.cfg.gps_dropout, env.latency_ticks = old_dropout, old_latency


def test_gae_bootstrap_and_episode_boundaries():
    # First transition truncates: bootstrap 10, but never include the next episode.
    adv, returns = advantages(
        jp.array([[1.0], [2.0]]),
        jp.zeros((2, 1)),
        jp.array([[10.0], [20.0]]),
        jp.array([[False], [True]]),
        jp.array([[True], [False]]),
        0.9,
        0.95,
    )
    np.testing.assert_allclose(adv[:, 0], [10.0, 2.0])
    np.testing.assert_array_equal(adv, returns)
    assert np.isfinite(
        log_probability(jp.array([100.0, -100.0, 0.0]), jp.zeros(3), jp.zeros(3))
    )


def test_batched_step_and_time_limit(env):
    reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    step = jax.jit(jax.vmap(env.step))
    states = reset(jax.random.split(jax.random.PRNGKey(8), 2), 0.0)
    for i in range(env.episode_steps):
        states, reward, term, trunc, metrics = step(states, states.action)
        assert np.all(np.isfinite(states.obs)) and np.all(np.isfinite(reward))
        assert not np.any(term)
        assert np.all(trunc == (i == env.episode_steps - 1))
    np.testing.assert_allclose(states.data.time, 0.2, atol=1e-5)
    assert np.all(metrics[:, 2] > 0.0)
    assert states.obs.shape == (2, env.obs_size)
    np.testing.assert_allclose(
        states.sensors[:, env.gps_indices],
        states.data.sensordata[:, env.gps_indices],
        atol=1e-5,
    )


def test_sensor_csv_is_complete_and_sourced(env):
    rows = sensor_spec.load()
    observed = {n for n, r in rows.items() if r["observed"] == "yes"}
    assert observed == set(env.channels)
    for name in observed:
        r = rows[name]
        assert r["source"] and r["datasheet_noise"], name
        assert float(r["noise_std"]) > 0, name
        assert float(r["rate_hz"]) > 0, name
    defaults = EnvConfig()
    # Noise is temporarily off by default (todo/TODO.md); restore these then:
    # defaults.noise_std == sensor_spec.defaults("noise_std"), same for bias_std.
    assert defaults.noise_std == {} and defaults.bias_std == {}
    # The CSV's GNSS rate is the env's GNSS rate.
    assert float(rows["gps_pos_xy"]["rate_hz"]) == defaults.gps_hz
    assert float(rows["baro_alt"]["rate_hz"]) == defaults.baro_hz


def test_barometer_holds_between_samples(env, initial):
    # 100 Hz sensor ticks, 20 Hz barometer: a new altitude only every 5th sample.
    ratio = env.baro_stride // env.sensor_stride
    assert ratio == env.cfg.sensor_hz // env.cfg.baro_hz
    z = env.slices["pod_pos"].start + 2
    s, seen = initial, []
    for i in range(1, 2 * ratio + 1):
        s = s.replace(data=s.data.replace(sensordata=s.data.sensordata.at[z].set(100.0 + i)))
        s = env._sample(s, i * env.sensor_stride)
        seen.append(float(s.sensors[env.baro]))
    fresh = [i for i in range(1, 2 * ratio + 1) if (i * env.sensor_stride) % env.baro_stride == 0]
    assert fresh == [ratio, 2 * ratio]
    for i, value in enumerate(seen, start=1):
        latest = max([f for f in fresh if f <= i], default=None)
        if latest is None:
            assert value == float(initial.sensors[env.baro])  # still the reset reading
        else:
            assert abs(value - (100.0 + latest)) < 10.0  # that sample's truth + noise/bias
            assert value == seen[latest - 1]  # held, bit for bit


def test_observation_excludes_unmeasurable_channels(env, initial):
    assert env.frame_size == 6 + 3 + 3 + 3 + 1 + 2 + 1 + 2
    assert env.obs_size == 25 * env.frame_size + 3 * len(env.cfg.preview_index) + 5
    # Perturbing channels the vehicle cannot measure must not change the frame.
    truth = initial.data.sensordata
    hidden = [i for n in ("vel_body", "pod_angvel", "brake_len_L", "brake_len_R")
              for i in range(env.slices[n].start, env.slices[n].stop)]
    hidden += [env.slices["pod_pos"].start + i for i in range(3)]  # GNSS position
    hidden += list(range(env.slices["mag"].start, env.slices["mag"].stop))
    values = env._corrupt(truth, jp.zeros(env.n_error), jax.random.PRNGKey(0))
    frame = env._frame(values, initial.origin, jp.zeros(()), jp.array(True))
    moved = values.at[jp.array(hidden)].add(5.0)
    np.testing.assert_array_equal(
        frame, env._frame(moved, initial.origin, jp.zeros(()), jp.array(True)))


def test_frame_is_heading_invariant(env, initial):
    # Yawing the whole vehicle (attitude and world velocity) leaves the frame alone.
    values = env._corrupt(initial.data.sensordata, jp.zeros(env.n_error), jax.random.PRNGKey(0))
    yaw = 1.3
    turn = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    q = quat_mul(rotvec_quat(jp.array([0.0, 0.0, yaw])), values[env.slices["pod_quat"]])
    vel = env.slices["pod_vel"]
    turned = values.at[env.slices["pod_quat"]].set(q).at[vel].set(turn @ values[vel])
    np.testing.assert_allclose(
        env._frame(values, initial.origin, jp.zeros(()), jp.array(True)),
        env._frame(turned, initial.origin, jp.zeros(()), jp.array(True)), atol=1e-5)
    # Gravity is kept: a vehicle at rest reads 1 g straight up, at any attitude.
    gravity = np.asarray(env.m.opt.gravity)
    rest = values.at[env.slices["accel"]].set(
        quat_mat(values[env.slices["pod_quat"]]).T @ -gravity)
    frame = env._frame(rest, initial.origin, jp.zeros(()), jp.array(True))
    np.testing.assert_allclose(frame[9:12], -gravity / 20.0, atol=1e-5)


def test_default_noise_statistics(initial):
    env = ParamotorEnv(EnvConfig(episode_seconds=0.2,
                                 noise_std=sensor_spec.defaults("noise_std"),
                                 bias_std=sensor_spec.defaults("bias_std")))
    truth = initial.data.sensordata
    keys = jax.random.split(jax.random.PRNGKey(1), 4000)
    zero = jp.zeros(env.n_error)
    samples = jax.vmap(lambda k: env._corrupt(truth, zero, k))(keys)
    rows = sensor_spec.load()
    # White noise matches the CSV for additive channels.
    for name, index in (("gyro", env.slices["gyro"].start),
                        ("baro_alt", env.baro),
                        ("gps_pos_xy", env.slices["pod_pos"].start)):
        std = float(np.std(samples[:, index]))
        expected = float(rows[name]["noise_std"])
        if name == "gps_pos_xy":
            # Clipped at +-1 m (2 sigma at 0.5 m): std of a clipped normal.
            expected *= 0.9594
            assert float(np.max(np.abs(samples[:, index] - truth[index]))) <= 1.0 + 1e-6
        np.testing.assert_allclose(std, expected, rtol=0.06)
    # Attitude error: angle between true and measured quaternions.
    q = truth[env.slices["pod_quat"]]
    dots = np.abs(np.asarray(samples[:, env.slices["pod_quat"]]) @ np.asarray(q))
    angle = 2 * np.arccos(np.clip(dots, -1, 1))
    expected = np.sqrt(2 * float(rows["attitude_roll_pitch"]["noise_std"]) ** 2
                       + float(rows["attitude_yaw"]["noise_std"]) ** 2)
    np.testing.assert_allclose(np.sqrt(np.mean(angle**2)), expected, rtol=0.06)
    # Per-episode bias is drawn at reset with the CSV's standard deviation.
    biases = jax.vmap(lambda k: env.reset(k).bias)(jax.random.split(jax.random.PRNGKey(2), 400))
    np.testing.assert_allclose(float(np.std(biases[:, env.baro])),
                               float(rows["baro_alt"]["bias_std"]), rtol=0.15)


def test_select_and_repeat_batched_states(env):
    reset = jax.jit(jax.vmap(env.reset))
    a = reset(jax.random.split(jax.random.PRNGKey(1), 3))
    b = reset(jax.random.split(jax.random.PRNGKey(2), 3))
    mask = jp.array([True, False, True])
    merged = select(mask, b, a)
    np.testing.assert_array_equal(merged.data.qpos, jp.where(mask[:, None], b.data.qpos, a.data.qpos))
    np.testing.assert_array_equal(merged.points, jp.where(mask[:, None, None], b.points, a.points))
    tiled = repeat(a, 2)
    assert tiled.data.qpos.shape[0] == 6 and tiled.obs.shape[0] == 6
    # The merged batch still steps on MJX Warp.
    jax.jit(jax.vmap(env.step))(merged, jp.zeros((3, 3)))
    jax.jit(jax.vmap(env.step))(tiled, jp.zeros((6, 3)))


def test_thrust_clamp_matches_xml(env):
    assert env.cfg.thrust_max == 2.0
    assert env.m.actuator_ctrlrange[env.physics.thrust, 1] == 2.0
    with pytest.raises(ValueError):
        ParamotorEnv(EnvConfig(thrust_max=2.1))
    states = jax.jit(env.reset)(jax.random.PRNGKey(3))
    states, *_ = jax.jit(env.step)(states, jp.array([1.0, -1.0, -1.0]))
    assert float(states.data.ctrl[env.physics.thrust]) <= 2.0 + 1e-6
