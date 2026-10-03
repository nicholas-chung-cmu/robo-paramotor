"""Fixed-shape, untimed routes. Distances and coordinates are in meters."""

import jax
import jax.numpy as jp

KINDS = (
    "random",
    "straight",
    "left",
    "right",
    "s_turn",
    "figure_eight",
    "climb",
    "descend",
)


def make_path(key, difficulty=1.0, kind="random", count=101, spacing=10.0):
    """Integrate smooth curvature and grade; no sharp waypoint corners.

    Positive grade is a climb. Curriculum 0 is straight and level; horizontal
    curvature grows first, vertical variation starts above difficulty 0.5.
    The default is ~1 km stored every 10 m; projection and preview interpolate
    linearly between points.
    """
    s = jp.arange(count) * spacing
    phase = jax.random.uniform(key, (3,), minval=-jp.pi, maxval=jp.pi)
    curvature = (
        difficulty
        * 0.04
        * (0.6 * jp.sin(s / 35 + phase[0]) + 0.4 * jp.sin(s / 70 + phase[1]))
    )
    grade = jp.maximum(2 * difficulty - 1, 0) * 0.10 * jp.sin(s / 80 + phase[2])
    if kind != "random":
        curvature = jp.zeros_like(s)
        grade = jp.zeros_like(s)
        if kind in ("left", "right"):
            curvature = jp.full_like(s, (1 if kind == "left" else -1) / 60)
        elif kind == "s_turn":
            curvature = 0.02 * jp.sin(s / 30)
        elif kind == "figure_eight":
            # A closed smooth figure eight, with initial tangent along +x.
            angle = s / 60
            xy = jp.stack((60 * jp.sin(angle), 30 * jp.sin(2 * angle)), -1)
            rot = jp.array([[1.0, -1.0], [1.0, 1.0]]) / jp.sqrt(2.0)
            xy = xy @ rot
            points = jp.concatenate((xy, jp.zeros((count, 1))), -1)
            arc = jp.concatenate(
                (
                    jp.zeros(1),
                    jp.cumsum(jp.linalg.norm(jp.diff(points, axis=0), axis=1)),
                )
            )
            return points, arc
        elif kind == "climb":
            grade = 0.08 * (1 - jp.exp(-s / 30)) * 0.5 * (1 - jp.tanh((s - 200) / 40))
        elif kind == "descend":
            grade = -0.15 * (1 - jp.exp(-s / 30)) * 0.5 * (1 - jp.tanh((s - 140) / 25))
    heading = jp.cumsum(curvature * spacing) - curvature[0] * spacing
    increments = spacing * jp.stack((jp.cos(heading), jp.sin(heading), grade), -1)
    points = jp.concatenate((jp.zeros((1, 3)), jp.cumsum(increments[:-1], axis=0)))
    return points, s  # progress is horizontal distance, including on slopes


def project(points, arc, position, previous, back=3, ahead=12):
    """Nearest segment in a bounded neighborhood; cannot jump route branches."""
    index = jp.clip(
        previous + jp.arange(-back, ahead + 1, dtype=jp.int32), 0, len(points) - 2
    )
    start = points[index]
    delta = points[index + 1] - start
    fraction = jp.clip(
        jp.sum((position - start) * delta, -1)
        / jp.maximum(jp.sum(delta * delta, -1), 1e-8),
        0,
        1,
    )
    closest = start + fraction[:, None] * delta
    best = jp.argmin(jp.sum((position - closest) ** 2, -1))
    i = index[best]
    progress = arc[i] + fraction[best] * (arc[i + 1] - arc[i])
    tangent = delta[best] / jp.maximum(jp.linalg.norm(delta[best]), 1e-8)
    return i, progress, closest[best], tangent


def preview(points, arc, progress, distances):
    return jax.vmap(
        lambda s: jp.array([jp.interp(s, arc, points[:, a]) for a in range(3)])
    )(progress + distances)
