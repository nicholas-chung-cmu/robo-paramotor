"""Actuator conventions shared by the interactive viewer and training."""

import numpy as np

# T = K_T * omega², anchored at 4.02 N and an assumed 10 krpm.
# Refit this coefficient when thrust-stand measurements are available.
K_T = 4.02 / (10_000 * 2 * np.pi / 60.0) ** 2
BRAKE_RESPONSE_RATE = 6.64  # critically damped; 99% of a step after 1 s


def smooth_brakes(position, velocity, target, dt, xp=np):
    """Exact PD step, accepting either NumPy or jax.numpy arrays."""
    error = position - target
    slope = velocity + BRAKE_RESPONSE_RATE * error
    decay = xp.exp(-BRAKE_RESPONSE_RATE * dt)
    value = target + (error + slope * dt) * decay
    rate = (velocity - BRAKE_RESPONSE_RATE * slope * dt) * decay
    rate = xp.where((value < 0) | (value > 3), 0, rate)
    return xp.clip(value, 0, 3), rate


def omega_from_thrust(thrust):
    return float(np.sqrt(max(thrust, 0.0) / K_T))


def sync_prop(model, data, thrust):
    """Set native MuJoCo thrust and its matching propeller spin."""
    data.ctrl[model.actuator("thrust").id] = thrust
    speed = omega_from_thrust(thrust)
    data.qvel[model.joint("prop_spin").dofadr[0]] = speed
    return speed
