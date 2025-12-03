from dataclasses import dataclass
from pathlib import Path
from random import sample
from typing import Mapping, Sequence, Optional, Tuple, List, Dict, Set

import numpy as np
import math
from dg_commons import PlayerName
from dg_commons.sim import InitSimGlobalObservations, InitSimObservations, SharedGoalObservation, SimObservations
from dg_commons.sim.agents import Agent, GlobalPlanner
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.diff_drive import DiffDriveCommands
from dg_commons.sim.models.diff_drive_structures import DiffDriveGeometry, DiffDriveParameters
from dg_commons.sim.shared_goals import CollectionPoint, SharedPolygonGoal
from dg_commons.sim.models.obstacles import StaticObstacle
from numpydantic import NDArray
from pydantic import BaseModel

from shapely import unary_union
from shapely.geometry import Point, Polygon, LineString
from scipy.spatial import KDTree


class FastMarchingTree:

    sample_points: List[Tuple[float, float]]
    neighbors: List[List[int]]
    indices_path: List[int]
    optimal_path: List[Tuple[float, float]]
    connection_radius: float

    def __init__(
        self,
        initObservations: InitSimGlobalObservations,
        n_samples: int = 500,
    ):
        self.static_obstacles: Sequence[StaticObstacle] = initObservations.dg_scenario.static_obstacles
        self.shared_goals: Optional[Mapping[str, SharedPolygonGoal]] = initObservations.shared_goals
        self.collection_points: Optional[Mapping[str, CollectionPoint]] = initObservations.collection_points
        self.n_samples = n_samples

    # Plan a path from start to goal using FMT*
    def plan_path(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:

        self.sample_points = self._sample_workspace(self.n_samples)

        self.indices_path = self._fmt_star(start, goal)
        self.optimal_path = self._indices_path_to_path()

        self.export_to_json()

        return self.optimal_path

    # FMT* core algorithm
    def _fmt_star(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[int]:
        self.sample_points.append(start)
        self.sample_points.append(goal)
        x_init_idx = len(self.sample_points) - 2
        x_goal_idx = len(self.sample_points) - 1

        self.connection_radius = self._compute_connection_radius(len(self.sample_points))

        n = len(self.sample_points)
        self._build_neighbors()

        V = set(range(n))

        Vopen: Set[int] = {x_init_idx}
        Vclosed: Set[int] = set()
        Vunvisited: Set[int] = V - {x_init_idx}

        cost: Dict[int, float] = {i: math.inf for i in V}
        cost[x_init_idx] = 0.0

        parent: Dict[int, Optional[int]] = {x_init_idx: None}

        while Vopen:
            z = min(Vopen, key=lambda i: cost[i])
            if z == x_goal_idx:
                break
            X_near = [x for x in self.neighbors[z] if x in Vunvisited]

            for x in X_near:
                Y_near = [y for y in self.neighbors[x] if y in Vopen]
                if not Y_near:
                    continue

                best_y = None
                best_cost = math.inf
                px = self.sample_points[x]

                for y in Y_near:
                    py = self.sample_points[y]
                    c = cost[y] + self._distance(py, px)
                    if c < best_cost:
                        best_cost = c
                        best_y = y

                if best_y is None:
                    continue

                if not self._collision_free(self.sample_points[best_y], px):
                    continue
                parent[x] = best_y
                cost[x] = best_cost

                Vopen.add(x)
                Vunvisited.remove(x)

            Vopen.remove(z)
            Vclosed.add(z)

        if x_goal_idx not in parent:
            return []

        path_indices: List[int] = []
        cur = x_goal_idx
        while cur is not None:
            path_indices.append(cur)
            cur = parent.get(cur)

        path_indices.reverse()
        return path_indices

    # Compute free space area μ(X_free)
    def _compute_free_space_area(self) -> float:
        """
        Compute μ(X_free): area of workspace minus polygonal obstacles.
        Used in the FMT* connection radius formula.
        """
        # Workspace boundary as polygon
        ring = self._get_workspace_linearring()
        workspace_poly = Polygon(ring)

        # Collect all obstacle geometries
        obstacle_geoms = []
        for obs in self.static_obstacles:
            geom = obs.shape
            if geom.geom_type == "LinearRing":
                continue
            if geom.geom_type in ("Polygon", "MultiPolygon"):
                obstacle_geoms.append(geom)

        if not obstacle_geoms:
            return workspace_poly.area

        # Union obstacles and intersect with workspace (in case some extend outside)
        obstacles_union = unary_union(obstacle_geoms)
        obstacles_in_ws = obstacles_union.intersection(workspace_poly)

        free_area = workspace_poly.area - obstacles_in_ws.area
        # guard against tiny negative due to numerical issues
        return max(free_area, 0.0)

    # Compute FMT* connection radius r_n
    def _compute_connection_radius(self, n_vertices: int) -> float:
        """
        Compute the FMT* connection radius r_n for d = 2:

            r_n = ((3/2) * μ(X_free) / π)^(1/2) * sqrt(log n / n)

        where μ(X_free) is the free-space area.
        """
        d = 2
        mu_free = self._compute_free_space_area()
        zeta_d = math.pi
        gamma = ((1.0 + 1.0 / d) * mu_free / zeta_d) ** (1.0 / d)

        if n_vertices <= 1:
            return 0.0

        rn = gamma * (math.log(n_vertices) / n_vertices) ** (1.0 / d)
        return rn

    # Convert list of indices to list of points
    def _indices_path_to_path(self) -> List[Tuple[float, float]]:
        return [self.sample_points[idx] for idx in self.indices_path]

    # get the LinearRing polygon defining the workspace boundary
    def _get_workspace_linearring(self):
        for obs in self.static_obstacles:
            geom = obs.shape
            if geom.geom_type == "LinearRing":
                return geom
        raise RuntimeError("No LinearRing workspace boundary found.")

    # sample random points within the workspace and non colliding using the Halton sequence
    def _sample_workspace(self, n_samples: int) -> List[Tuple[float, float]]:
        boundary_linearring = self._get_workspace_linearring()
        boundary_polygon = Polygon(boundary_linearring)
        minx, miny, maxx, maxy = boundary_polygon.bounds

        halton = self._halton_sequence(size=n_samples, dim=2)
        samples = []
        for h in halton:
            # scale Halton point into workspace bounds
            x = minx + h[0] * (maxx - minx)
            y = miny + h[1] * (maxy - miny)

            p = Point(x, y)

            # (1) must be inside workspace
            if not boundary_polygon.contains(p):
                continue
            # (2) must NOT be inside any obstacle
            inside_obstacle = False
            for obs in self.static_obstacles:
                geom = obs.shape
                if geom.geom_type == "LinearRing":
                    continue

                if geom.contains(p) or geom.touches(p):
                    inside_obstacle = True
                    break

            if inside_obstacle:
                continue

            samples.append((x, y))

        return samples

    # The Halton Sequence, Lecture 9: Sampling-Based Methods p. 15
    @staticmethod
    def _halton_sequence(size, dim, bases=None):
        if bases is None:
            # First d primes, as suggested in the lecture
            bases = [2, 3, 5, 7, 11][:dim]

        def vdc(n, base):
            vdc, denom = 0.0, 1.0
            while n:
                n, remainder = divmod(n, base)
                denom *= base
                vdc += remainder / denom
            return vdc

        seq = np.zeros((size, dim))
        for i in range(size):
            for d in range(dim):
                seq[i, d] = vdc(i + 1, bases[d])
        return seq

    # Euclidean distance between two points
    @staticmethod
    def _distance(p: Tuple[float, float], q: Tuple[float, float]) -> float:
        return float(math.hypot(p[0] - q[0], p[1] - q[1]))

    # build neighbor connections within connection radius, with KDTree acc. to Lecture
    def _build_neighbors(self):
        pts = np.array(self.sample_points)
        tree = KDTree(pts)

        neighbors = []
        for i, p in enumerate(pts):
            idxs = tree.query_ball_point(p, self.connection_radius)
            idxs.remove(i)  # remove itself
            neighbors.append(idxs)

        self.neighbors = neighbors

    # check if the line segment between p and q is collision-free
    def _collision_free(self, p, q):
        segment = LineString([p, q])

        # Check intersection with polygon obstacles
        for obs in self.static_obstacles:
            geom = obs.shape
            if geom.geom_type in ("Polygon", "MultiPolygon"):
                if segment.intersects(geom):
                    return False
        return True

    # For debugging: export workspace, obstacles, and samples to JSON
    def export_to_json(self) -> str:
        import json

        """
        Export workspace polygon, obstacles, and sample points in JSON format.
        Coordinates are represented as lists of [x, y].
        """

        # Workspace boundary
        ring = self._get_workspace_linearring()
        workspace_poly = Polygon(ring)
        workspace_coords = list(map(list, workspace_poly.exterior.coords))

        # Obstacles
        obstacles_json = []
        for obs in self.static_obstacles:
            geom = obs.shape
            if geom.geom_type == "Polygon":
                obstacles_json.append(
                    {
                        "type": "Polygon",
                        "exterior": list(map(list, geom.exterior.coords)),
                        "interiors": [list(map(list, interior.coords)) for interior in geom.interiors],
                    }
                )

            elif geom.geom_type == "MultiPolygon":
                multi_list = []
                for poly in geom.geoms:
                    multi_list.append(
                        {
                            "type": "Polygon",
                            "exterior": list(map(list, poly.exterior.coords)),
                            "interiors": [list(map(list, interior.coords)) for interior in poly.interiors],
                        }
                    )
                obstacles_json.append({"type": "MultiPolygon", "polygons": multi_list})

        # Samples
        samples_json = [list(pt) for pt in self.sample_points]

        # Optimal path
        if hasattr(self, "optimal_path"):
            optimal_path_json = [list(pt) for pt in self.optimal_path]
        else:
            optimal_path_json = []

        # Construct JSON-friendly object
        data = {
            "workspace_polygon": workspace_coords,
            "obstacles": obstacles_json,
            "samples": samples_json,
            # "indices_path": self.indices_path if hasattr(self, "indices_path") else [],
            "optimal_path": optimal_path_json,
        }

        json_str = json.dumps(data, indent=2)

        # persist to /out/14 so it shows up next to the rendered report
        project_root = Path(__file__).resolve().parents[4]
        output_dir = project_root / "out" / "14"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "fmt_samples.json"
        output_file.write_text(json_str, encoding="utf-8")

        return json_str
