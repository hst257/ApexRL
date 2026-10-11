import math
import numpy as np


def closed_spline(control_points: np.ndarray, spacing: float = 2.0) -> np.ndarray:
    """Centripetal Catmull-Rom interpolation, then uniform arc-length resampling."""
    dense = []
    for i in range(len(control_points)):
        p0, p1, p2, p3 = [control_points[(i + k) % len(control_points)] for k in (-1, 0, 1, 2)]
        t0 = 0.0
        t1 = t0 + math.sqrt(float(np.linalg.norm(p1 - p0)))
        t2 = t1 + math.sqrt(float(np.linalg.norm(p2 - p1)))
        t3 = t2 + math.sqrt(float(np.linalg.norm(p3 - p2)))
        for t in np.linspace(t1, t2, max(16, int(np.linalg.norm(p2 - p1))), endpoint=False):
            a1 = (t1-t)/(t1-t0)*p0 + (t-t0)/(t1-t0)*p1
            a2 = (t2-t)/(t2-t1)*p1 + (t-t1)/(t2-t1)*p2
            a3 = (t3-t)/(t3-t2)*p2 + (t-t2)/(t3-t2)*p3
            b1 = (t2-t)/(t2-t0)*a1 + (t-t0)/(t2-t0)*a2
            b2 = (t3-t)/(t3-t1)*a2 + (t-t1)/(t3-t1)*a3
            dense.append((t2-t)/(t2-t1)*b1 + (t-t1)/(t2-t1)*b2)
    dense = np.array(dense)
    closed = np.vstack((dense, dense[0]))
    distances = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))))
    samples = np.linspace(0, distances[-1], max(32, math.ceil(distances[-1]/spacing)), endpoint=False)
    return np.column_stack([np.interp(samples, distances, closed[:, axis]) for axis in (0, 1)])


def segment_intersection(a, b, c, d) -> bool:
    def cross(p, q, r):
        return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
    return cross(a, b, c) * cross(a, b, d) < -1e-10 and cross(c, d, a) * cross(c, d, b) < -1e-10


def validate_closed_boundary(points: np.ndarray, name: str) -> None:
    """Check all nonadjacent segments, including locally folded offset curves."""
    ends = np.roll(points, -1, axis=0)
    lows, highs = np.minimum(points, ends), np.maximum(points, ends)
    n = len(points)
    for i in range(n-2):
        candidate = np.flatnonzero(np.all(highs[i] >= lows, axis=1) & np.all(highs >= lows[i], axis=1))
        for j in candidate:
            if j <= i+1 or (i == 0 and j == n-1):
                continue
            if segment_intersection(points[i], ends[i], points[j], ends[j]):
                raise ValueError(f"{name}: self-intersecting boundary segments {i}, {j}")


def gate_crossing(previous, current, center, tangent, half_width: float) -> float | None:
    """Return forward crossing fraction of movement through a finite gate."""
    before = (previous[0]-center[0])*tangent[0] + (previous[1]-center[1])*tangent[1]
    after = (current[0]-center[0])*tangent[0] + (current[1]-center[1])*tangent[1]
    if before >= 0 or after < 0 or after - before < 1e-10:
        return None
    fraction = -before / (after-before)
    x = previous[0] + fraction * (current[0]-previous[0]) - center[0]
    y = previous[1] + fraction * (current[1]-previous[1]) - center[1]
    if abs(-x*tangent[1] + y*tangent[0]) > half_width:
        return None
    return fraction
