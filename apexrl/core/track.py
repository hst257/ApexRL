from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
import numpy as np

from .geometry import closed_spline, segment_intersection, validate_closed_boundary


@dataclass(frozen=True, slots=True)
class TrackProjection:
    x: float
    y: float
    distance: float
    lateral_distance: float
    progress: float
    tangent: tuple[float, float]
    width: float
    segment: int


@dataclass(frozen=True, slots=True)
class Gate:
    progress: float
    center: tuple[float, float]
    tangent: tuple[float, float]
    half_width: float


class Track:
    """Read-only derived circuit geometry. Distances and positions are meters."""

    CELL_SIZE = 30.0
    CURB_WIDTH = 1.2
    RUNOFF_WIDTH = 10.0

    def __init__(self, definition: dict):
        self.id = str(definition["id"])
        self.name = str(definition["name"])
        self.difficulty = str(definition["difficulty"])
        self.theme = str(definition.get("theme", "meadow"))
        self.description = str(definition.get("description", ""))
        controls = np.array(definition["control_points"], dtype=float)
        if controls.ndim != 2 or controls.shape[1] != 2 or len(controls) < 4 or not np.isfinite(controls).all():
            raise ValueError(f"{self.id}: need at least four finite 2D control points")
        if np.min(np.linalg.norm(np.roll(controls, -1, axis=0)-controls, axis=1)) < 1:
            raise ValueError(f"{self.id}: duplicate or too-close control points")
        if len(np.unique(controls, axis=0)) != len(controls) or np.linalg.matrix_rank(controls-controls[0]) < 2:
            raise ValueError(f"{self.id}: duplicate or collinear control points")
        self.points = closed_spline(controls)
        target = float(definition.get("target_length", 0))
        if not math.isfinite(target) or target < 0:
            raise ValueError("target_length must be finite and nonnegative")
        if target:
            length = np.linalg.norm(np.roll(self.points, -1, axis=0)-self.points, axis=1).sum()
            self.points = closed_spline(controls * target / length)
        self.vectors = np.roll(self.points, -1, axis=0)-self.points
        self.lengths = np.linalg.norm(self.vectors, axis=1)
        self.length_squared = self.lengths ** 2
        self.total_length = float(self.lengths.sum())
        self.cumulative = np.concatenate(([0.0], np.cumsum(self.lengths)))
        self.tangents = self.vectors / self.lengths[:, None]
        smooth_tangents = self.tangents + np.roll(self.tangents, 1, axis=0)
        smooth_tangents /= np.linalg.norm(smooth_tangents, axis=1)[:, None]
        self.normals = np.column_stack((-smooth_tangents[:, 1], smooth_tangents[:, 0]))
        road_width = float(definition.get("road_width", 13))
        if not math.isfinite(road_width) or road_width < 6:
            raise ValueError(f"{self.id}: road width must be at least 6m")
        profile = definition.get("width_profile", [[0, road_width], [1, road_width]])
        profile = np.asarray(profile, dtype=float)
        if profile.ndim != 2 or profile.shape[1] != 2 or len(profile) < 2 or not np.isfinite(profile).all():
            raise ValueError("Invalid width profile")
        if profile[0, 0] != 0 or profile[-1, 0] != 1 or np.any(np.diff(profile[:, 0]) <= 0) or np.any(profile[:, 1] < 6) or profile[0, 1] != profile[-1, 1]:
            raise ValueError("Width profile must be ordered, positive and continuous across the seam")
        self.widths = np.interp(self.cumulative[:-1]/self.total_length, profile[:, 0], profile[:, 1])
        self.minimum_width = float(self.widths.min())
        self.left_boundary = self.points - self.normals*self.widths[:, None]/2
        self.right_boundary = self.points + self.normals*self.widths[:, None]/2
        self.left_barrier = self.points - self.normals*(self.widths[:, None]/2 + self.RUNOFF_WIDTH)
        self.right_barrier = self.points + self.normals*(self.widths[:, None]/2 + self.RUNOFF_WIDTH)
        angles = np.arctan2(self.tangents[:, 1], self.tangents[:, 0])
        self.curvatures = (np.roll(angles, -1)-angles+math.pi) % (2*math.pi)-math.pi
        self.curvatures /= self.lengths
        self.bounds = (float(self.points[:, 0].min()), float(self.points[:, 1].min()),
                       float(self.points[:, 0].max()), float(self.points[:, 1].max()))
        grid: dict[tuple[int, int], set[int]] = {}
        for i, (a, b) in enumerate(zip(self.points, np.roll(self.points, -1, axis=0))):
            lo = np.floor(np.minimum(a, b)/self.CELL_SIZE).astype(int)
            hi = np.floor(np.maximum(a, b)/self.CELL_SIZE).astype(int)
            for gx in range(lo[0], hi[0]+1):
                for gy in range(lo[1], hi[1]+1):
                    grid.setdefault((gx, gy), set()).add(i)
        self._grid = MappingProxyType({cell: np.array(sorted(indices), dtype=int) for cell, indices in grid.items()})
        self.start_progress = float(definition.get("start_progress", 0.0)) % self.total_length
        self.finish = self.gate(self.start_progress)
        fractions = definition.get("checkpoint_fractions", [i/12 for i in range(1, 12)])
        if not fractions or any(not 0 < f < 1 for f in fractions) or sorted(set(fractions)) != fractions:
            raise ValueError("Checkpoint fractions must be strictly ordered inside (0, 1)")
        self.checkpoints = tuple(self.gate(self.start_progress + f*self.total_length) for f in fractions)
        self.version = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()[:12]
        self.metadata = MappingProxyType({"name": self.name, "difficulty": self.difficulty, "theme": self.theme})
        self.validate()
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
        for value in self._grid.values():
            value.flags.writeable = False

    @classmethod
    def load(cls, path: str | Path) -> "Track":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def pose(self, progress: float) -> tuple[float, float, float]:
        progress %= self.total_length
        i = min(len(self.points)-1, int(np.searchsorted(self.cumulative, progress, side="right"))-1)
        fraction = (progress-self.cumulative[i])/self.lengths[i]
        p = self.points[i] + fraction*self.vectors[i]
        return float(p[0]), float(p[1]), math.atan2(self.tangents[i, 1], self.tangents[i, 0])

    def gate(self, progress: float) -> Gate:
        x, y, heading = self.pose(progress)
        i = max(0, int(np.searchsorted(self.cumulative, progress % self.total_length, side="right"))-1)
        return Gate(progress % self.total_length, (x, y), (math.cos(heading), math.sin(heading)), float(self.widths[i]/2+self.CURB_WIDTH))

    def project(self, x: float, y: float) -> TrackProjection:
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Projection position must be finite")
        gx, gy = math.floor(x/self.CELL_SIZE), math.floor(y/self.CELL_SIZE)
        candidates: set[int] = set()
        best = None
        for radius in range(1, 5):
            for ix in range(gx-radius, gx+radius+1):
                for iy in range(gy-radius, gy+radius+1):
                    indices = self._grid.get((ix, iy))
                    if indices is not None:
                        candidates.update(indices)
            if candidates:
                best = self._project_indices(x, y, np.fromiter(sorted(candidates), dtype=int))
                edge_distance = min(x-(gx-radius)*self.CELL_SIZE, (gx+radius+1)*self.CELL_SIZE-x,
                                    y-(gy-radius)*self.CELL_SIZE, (gy+radius+1)*self.CELL_SIZE-y)
                if best.distance <= edge_distance:
                    return best
        return self._project_indices(x, y, np.arange(len(self.points)))

    def _project_indices(self, x: float, y: float, indices: np.ndarray) -> TrackProjection:
        relative = np.array((x, y))-self.points[indices]
        fraction = np.clip(np.einsum("ij,ij->i", relative, self.vectors[indices])/self.length_squared[indices], 0, 1)
        projected = self.points[indices]+fraction[:, None]*self.vectors[indices]
        delta = np.array((x, y))-projected
        distances = np.einsum("ij,ij->i", delta, delta)
        local = int(np.argmin(distances))
        i, f = int(indices[local]), float(fraction[local])
        tangent = self.tangents[i]
        lateral = -delta[local, 0]*tangent[1]+delta[local, 1]*tangent[0]
        width = self.widths[i]*(1-f)+self.widths[(i+1)%len(self.points)]*f
        return TrackProjection(float(projected[local, 0]), float(projected[local, 1]), math.sqrt(float(distances[local])),
                               float(lateral), float((self.cumulative[i]+f*self.lengths[i]) % self.total_length),
                               (float(tangent[0]), float(tangent[1])), float(width), i)

    def surface(self, x: float, y: float) -> str:
        projection = self.project(x, y)
        if projection.distance <= projection.width/2:
            return "asphalt"
        if projection.distance <= projection.width/2+self.CURB_WIDTH:
            return "curb"
        return "grass"

    def curvature(self, progress: float, lookahead: float = 30.0) -> float:
        samples = (progress + np.linspace(0, max(0, lookahead), 8)) % self.total_length
        indices = np.searchsorted(self.cumulative, samples, side="right")-1
        return float(self.curvatures[indices[np.argmax(np.abs(self.curvatures[indices]))]])

    def validate(self) -> None:
        if not np.isfinite(self.points).all() or np.min(self.lengths) < 0.01:
            raise ValueError(f"{self.id}: invalid centerline segments")
        if np.max(np.abs(self.curvatures)*(self.widths/2+self.CURB_WIDTH)) >= 0.9:
            raise ValueError(f"{self.id}: corner too tight for road width")
        if np.max(np.abs(self.curvatures)*(self.widths/2+self.RUNOFF_WIDTH)) >= 0.9:
            raise ValueError(f"{self.id}: corner too tight for runoff barriers")
        for name, path in (("centerline", self.points), ("left road", self.left_boundary), ("right road", self.right_boundary),
                           ("left barrier", self.left_barrier), ("right barrier", self.right_barrier)):
            validate_closed_boundary(path, f"{self.id}: {name}")
        n = len(self.points)
        for i in range(n):
            gx, gy = np.floor(self.points[i]/self.CELL_SIZE).astype(int)
            nearby = set()
            for ix in range(gx-1, gx+2):
                for iy in range(gy-1, gy+2):
                    nearby.update(self._grid.get((ix, iy), ()))
            for j in nearby:
                if j <= i:
                    continue
                separation = min(abs(self.cumulative[i]-self.cumulative[j]), self.total_length-abs(self.cumulative[i]-self.cumulative[j]))
                if separation < 3*max(self.widths[i], self.widths[j]):
                    continue
                if np.linalg.norm(self.points[i]-self.points[j]) < (self.widths[i]+self.widths[j])/2+2*self.CURB_WIDTH:
                    raise ValueError(f"{self.id}: disconnected sections overlap at {i}, {j}")
                for path in (self.points, self.left_boundary, self.right_boundary, self.left_barrier, self.right_barrier):
                    if segment_intersection(path[i], path[(i+1)%n], path[j], path[(j+1)%n]):
                        raise ValueError(f"{self.id}: self-intersecting boundaries")


def load_tracks(directory: str | Path) -> tuple[Track, ...]:
    paths = [Path(directory)/f"{name}.json" for name in ("rookie_ring", "apex_valley", "velocity_circuit")]
    return tuple(Track.load(path) for path in paths)
