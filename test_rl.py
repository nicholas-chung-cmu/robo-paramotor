"""Regression checks for physics parity, sensor timing, guidance, and PPO math."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import numpy as np
import pytest

import paths
from paramotor_aero import ParamotorAero
from paramotor_control import smooth_brakes
from paramotor_params import PEEK_1M
from rl_env import EnvConfig, ParamotorEnv
from train import advantages, log_probability


@pytest.fixture(scope="module")
def env():
    mujoco.set_mjcb_passive(None)
    return ParamotorEnv(EnvConfig(episode_seconds=0.2, position_randomization_m=0.0))


@pytest.fixture(scope="module")
def initial(env):
    return jax.jit(env.reset)(jax.random.PRNGKey(4))


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
        for a, b in zip(physics.free_qpos, physics.free_dofs):
            d.qpos[a + 2] += 40
            d.qvel[b : b + 3] = velocity
            d.qvel[b + 3 : b + 6] = rates
        # Include a nonzero attitude and unequal arm positions.
        delta = np.zeros(m.nv)
        for b in physics.free_dofs:
            delta[b + 3 : b + 6] = [0.1, -0.06, 0.2]
        mujoco.mj_integratePos(m, d.qpos, delta, 1.0)
        d.ctrl[:] = [0.8, 0.2, 0.1]
        try:
            mujoco.set_mjcb_passive(aero)
            mujoco.mj_forward(m, d)
            expected = aero.wrench.copy()
        finally:
            mujoco.set_mjcb_passive(None)
        result = calculate(mjx.put_data(m, d))
        np.testing.assert_allclose(result.xfrc_applied, expected, atol=2e-5, rtol=2e-4)
        np.testing.assert_allclose(result.qacc, d.qacc, atol=0.03, rtol=3e-3)
        np.testing.assert_allclose(
            result.sensordata, d.sensordata, atol=0.01, rtol=3e-3
        )


def test_native_short_rollout_parity(env, initial):
    physics = env.physics
    m = physics.native
    d = mujoco.MjData(m)
    d.qpos[:], d.qvel[:], d.ctrl[:] = (
        initial.data.qpos,
        initial.data.qvel,
        initial.data.ctrl,
    )
    aero = ParamotorAero(m, PEEK_1M)
    try:
        mujoco.set_mjcb_passive(aero)
        for _ in range(400):
            mujoco.mj_step(m, d)
    finally:
        mujoco.set_mjcb_passive(None)
    advance = jax.jit(
        lambda data: jax.lax.fori_loop(
            0, 400, lambda _, x: physics.step(x, with_sensors=False), data
        )
    )
    result = advance(initial.data)
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


def test_random_path_limits_and_projection():
    p, arc = paths.make_path(jax.random.PRNGKey(2))
    delta = np.diff(p, axis=0)
    horizontal = np.linalg.norm(delta[:, :2], axis=1)
    grade = delta[:, 2] / horizontal
    heading = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    assert abs(grade).max() <= 0.10001
    assert abs(np.diff(heading) / 2).max() <= 0.04001
    np.testing.assert_allclose(horizontal, 2.0, atol=1e-4)
    straight, s = paths.make_path(jax.random.PRNGKey(0), kind="straight")
    i, progress, closest, tangent = paths.project(
        straight, s, jp.array([11.0, 3.0, -2.0]), 4
    )
    assert int(i) == 5
    np.testing.assert_allclose(progress, 11.0)
    np.testing.assert_allclose(closest, [11.0, 0.0, 0.0])
    np.testing.assert_allclose(tangent, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        paths.preview(straight, s, progress, jp.array([5.0, 10.0]))[:, 0], [16.0, 21.0]
    )


def test_figure_eight_projection_stays_on_current_branch():
    p, arc = paths.make_path(jax.random.PRNGKey(0), kind="figure_eight")
    index, progress, _, _ = paths.project(p, arc, jp.zeros(3), 0)
    assert int(index) == 0 and float(progress) == 0
    index, progress, _, _ = paths.project(p, arc, jp.zeros(3), 90)
    assert 87 <= int(index) <= 102 and float(progress) > 100


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
    assert s.history.shape == (100, env.m.nsensordata + 2)
    assert set(env.slices) == {env.m.sensor(i).name for i in range(env.m.nsensor)}


def test_noise_reproducibility_and_units(env, initial):
    old = env.noise
    try:
        env.noise = env._sensor_vector({"gyro": [0.1, 0.2, 0.3]})
        one = env._corrupt(
            initial.data.sensordata, jp.zeros(env.m.nsensordata), jax.random.PRNGKey(5)
        )
        two = env._corrupt(
            initial.data.sensordata, jp.zeros(env.m.nsensordata), jax.random.PRNGKey(5)
        )
        other = env._corrupt(
            initial.data.sensordata, jp.zeros(env.m.nsensordata), jax.random.PRNGKey(6)
        )
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
