"""Fixed-shape, untimed routes. Distances and coordinates are in meters.

A route is just its points, spaced `spacing` meters apart horizontally (along the
path for the figure eight). The vehicle works through them in order: the target is
the first point it has not yet passed, and it never looks anywhere else.
"""

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
    s = jp.arange(count) * spacing  # distance of each point along the route
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
            # Sampled finely, then resampled at equal path length so the
            # points are `spacing` apart like every other route.
            angle = jp.linspace(0, count * spacing / 30, 80 * count)
            xy = jp.stack((60 * jp.sin(angle), 30 * jp.sin(2 * angle)), -1)
            rot = jp.array([[1.0, -1.0], [1.0, 1.0]]) / jp.sqrt(2.0)
            fine = jp.concatenate((xy @ rot, jp.zeros((len(angle), 1))), -1)
            length = jp.concatenate(
                (jp.zeros(1), jp.cumsum(jp.linalg.norm(jp.diff(fine, axis=0), axis=1)))
            )
            return jax.vmap(lambda a: jp.interp(s, length, a), 1, 1)(fine)
        elif kind == "climb":
            grade = 0.08 * (1 - jp.exp(-s / 30)) * 0.5 * (1 - jp.tanh((s - 200) / 40))
        elif kind == "descend":
            grade = -0.15 * (1 - jp.exp(-s / 30)) * 0.5 * (1 - jp.tanh((s - 140) / 25))
    heading = jp.cumsum(curvature * spacing) - curvature[0] * spacing
    increments = spacing * jp.stack((jp.cos(heading), jp.sin(heading), grade), -1)
    points = jp.concatenate((jp.zeros((1, 3)), jp.cumsum(increments[:-1], axis=0)))
    return points


def nearest_on_segment(points, target, position):
    """Closest point on the segment ending at `target` (the first unpassed point).

    No search: the segment comes from the target index alone. Returns
    (segment index, fraction along it, closest point).
    """
    i = jp.clip(target - 1, 0, len(points) - 2)
    start, end = points[i], points[i + 1]
    delta = end - start
    fraction = jp.clip(
        jp.dot(position - start, delta) / jp.maximum(jp.dot(delta, delta), 1e-8), 0, 1
    )
    return i, fraction, start + fraction * delta


def preview(points, target, offsets):
    """The points `offsets` indices past the target, clamped to the last point."""
    return points[jp.clip(target + offsets, 0, len(points) - 1)]
