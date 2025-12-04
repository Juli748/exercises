from dataclasses import dataclass
from mimetypes import init
from os import path
from pathlib import Path
from random import sample
from tracemalloc import start
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
    start: Tuple[float, float]
    goal: Tuple[float, float]

    _start: Optional[Tuple[float, float]] = None
    _goal: Optional[Tuple[float, float]] = None

    # Tuneable parameters
    robot_radius: float = 0.6
    robot_clearance: float = 0.1

    treat_foreign_goals_as_obstacles: bool = True

    n_samples: int = 5000
    connection_radius: float = 2.5  # ≥2 required for asymptotic optimality (Karaman & Frazzoli 2011)

    def __init__(self, initObservations: InitSimGlobalObservations):
        self.static_obstacles: Sequence[StaticObstacle] = initObservations.dg_scenario.static_obstacles
        self.shared_goals: Optional[Mapping[str, SharedPolygonGoal]] = initObservations.shared_goals
        self.collection_points: Optional[Mapping[str, CollectionPoint]] = initObservations.collection_points

        self.sample_points = self._sample_workspace(self.n_samples)
        pts = np.array(self.sample_points)
        self.kdtree = KDTree(pts)
        self.neighbors = self._build_neighbor_graph(pts)

    def plan_path(
        self, start: Tuple[float, float], goal: Tuple[float, float]
    ) -> Tuple[List[Tuple[float, float]], float]:
        self._start = start
        self._goal = goal
        self._start_point = Point(start)
        self._goal_point = Point(goal)

        indices, points_snapshot = self._fmt_star(start, goal)
        self.indices_path = indices
        self.optimal_path = self._indices_path_to_path(points_snapshot)
        path_length = self.get_optimal_path_length()

        # self.export_to_json()
        return self.optimal_path, path_length

    def get_optimal_path_length(self) -> float:
        if not hasattr(self, "optimal_path") or len(self.optimal_path) < 2:
            return 0.0
        total = 0.0
        for i in range(len(self.optimal_path) - 1):
            p = self.optimal_path[i]
            q = self.optimal_path[i + 1]
            total += self._distance(p, q)
        return total

    def _fmt_star(
        self, start: Tuple[float, float], goal: Tuple[float, float]
    ) -> Tuple[List[int], List[Tuple[float, float]]]:
        base_points = list(self.sample_points)
        points = base_points + [start, goal]

        start_idx = len(base_points)
        goal_idx = start_idx + 1

        neighbors: List[List[int]] = [list(nbs) for nbs in self.neighbors]
        neighbors.append([])
        neighbors.append([])

        def connect_new_vertex(vertex_idx: int, vertex_point: Tuple[float, float]):
            nearby = self.kdtree.query_ball_point(vertex_point, self.connection_radius)
            for idx in nearby:
                if vertex_idx not in neighbors[idx]:
                    neighbors[idx].append(vertex_idx)
                if idx not in neighbors[vertex_idx]:
                    neighbors[vertex_idx].append(idx)

        connect_new_vertex(start_idx, start)
        connect_new_vertex(goal_idx, goal)

        if self._distance(start, goal) <= self.connection_radius:
            neighbors[start_idx].append(goal_idx)
            neighbors[goal_idx].append(start_idx)

        x_init_idx = start_idx
        x_goal_idx = goal_idx

        n = len(points)
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

            X_near = [x for x in neighbors[z] if x in Vunvisited]

            for x in X_near:
                Y_near = [y for y in neighbors[x] if y in Vopen]
                if not Y_near:
                    continue

                best_y = None
                best_cost = math.inf
                px = points[x]

                for y in Y_near:
                    py = points[y]
                    c = cost[y] + self._distance(py, px)
                    if c < best_cost:
                        best_cost = c
                        best_y = y

                if best_y is None:
                    continue

                if not self._collision_free(points[best_y], px):
                    continue

                parent[x] = best_y
                cost[x] = best_cost

                Vopen.add(x)
                Vunvisited.remove(x)

            Vopen.remove(z)
            Vclosed.add(z)

        if x_goal_idx not in parent:
            return [], points

        points_snapshot = list(points)
        path_indices: List[int] = []
        cur = x_goal_idx
        while cur is not None:
            path_indices.append(cur)
            cur = parent.get(cur)

        path_indices.reverse()
        return path_indices, points_snapshot

    def _compute_free_space_area(self) -> float:
        ring = self._get_workspace_linearring()
        workspace_poly = Polygon(ring)

        obstacle_geoms = []
        for obs in self.static_obstacles:
            geom = obs.shape
            if geom.geom_type == "LinearRing":
                continue
            if geom.geom_type in ("Polygon", "MultiPolygon"):
                obstacle_geoms.append(geom)

        if not obstacle_geoms:
            return workspace_poly.area

        obstacles_union = unary_union(obstacle_geoms)
        obstacles_in_ws = obstacles_union.intersection(workspace_poly)
        free_area = workspace_poly.area - obstacles_in_ws.area
        return max(free_area, 0.0)

    def _compute_connection_radius(self, n_vertices: int) -> float:
        d = 2
        mu_free = self._compute_free_space_area()
        zeta_d = math.pi
        gamma = ((1.0 + 1.0 / d) * mu_free / zeta_d) ** (1.0 / d)

        if n_vertices <= 1:
            return 0.0

        rn = gamma * (math.log(n_vertices) / n_vertices) ** (1.0 / d)
        return rn

    def _indices_path_to_path(self, points_snapshot: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
        return [points_snapshot[idx] for idx in self.indices_path]

    def _build_neighbor_graph(self, pts: NDArray) -> List[List[int]]:
        neighbors: List[List[int]] = []
        for i in range(len(pts)):
            idxs = self.kdtree.query_ball_point(pts[i], self.connection_radius)
            if i in idxs:
                idxs.remove(i)
            neighbors.append(idxs)
        return neighbors

    def _get_workspace_linearring(self):
        for obs in self.static_obstacles:
            geom = obs.shape
            if geom.geom_type == "LinearRing":
                return geom
        raise RuntimeError("No LinearRing workspace boundary found.")

    def _sample_workspace(self, n_samples: int) -> List[Tuple[float, float]]:
        boundary_linearring = self._get_workspace_linearring()
        boundary_polygon = Polygon(boundary_linearring)
        minx, miny, maxx, maxy = boundary_polygon.bounds

        halton = self._halton_sequence(size=n_samples, dim=2)
        samples = []

        for h in halton:
            x = minx + h[0] * (maxx - minx)
            y = miny + h[1] * (maxy - miny)
            p = Point(x, y)

            if not boundary_polygon.contains(p):
                continue

            if boundary_linearring.distance(p) <= self.robot_radius + self.robot_clearance:
                continue

            colliding_obstacle = False
            for obs in self.static_obstacles:
                geom = obs.shape
                if geom.geom_type == "LinearRing":
                    continue
                if geom.distance(p) <= self.robot_radius + self.robot_clearance:
                    colliding_obstacle = True
                    break

            colliding_goal = False
            if self.shared_goals is not None and self.treat_foreign_goals_as_obstacles:
                for goal in self.shared_goals.values():
                    if hasattr(self, "_start_point") and hasattr(self, "_goal_point"):
                        if goal.polygon.contains(self._start_point) or goal.polygon.contains(self._goal_point):
                            continue
                    if goal.polygon.distance(p) <= self.robot_radius + self.robot_clearance:
                        colliding_goal = True
                        break

            if colliding_obstacle or colliding_goal:
                continue

            samples.append((x, y))

        return samples

    @staticmethod
    def _halton_sequence(size, dim, bases=None):
        if bases is None:
            bases = [2, 3, 5, 7, 11][:dim]

        def vdc(n, base):
            vdc_val, denom = 0.0, 1.0
            while n:
                n, remainder = divmod(n, base)
                denom *= base
                vdc_val += remainder / denom
            return vdc_val

        seq = np.zeros((size, dim))
        for i in range(size):
            for d in range(dim):
                seq[i, d] = vdc(i + 1, bases[d])
        return seq

    @staticmethod
    def _distance(p: Tuple[float, float], q: Tuple[float, float]) -> float:
        return float(math.hypot(p[0] - q[0], p[1] - q[1]))

    def _collision_free(self, p, q):
        segment = LineString([p, q])
        ring = self._get_workspace_linearring()

        if segment.distance(ring) <= self.robot_radius + self.robot_clearance:
            return False

        for obs in self.static_obstacles:
            geom = obs.shape
            if geom.geom_type in ("Polygon", "MultiPolygon"):
                if segment.distance(geom) <= self.robot_radius + self.robot_clearance:
                    return False

        if self.shared_goals is not None and self.treat_foreign_goals_as_obstacles:
            for goal in self.shared_goals.values():
                if hasattr(self, "_start_point") and hasattr(self, "_goal_point"):
                    if goal.polygon.contains(self._start_point) or goal.polygon.contains(self._goal_point):
                        continue
                if goal.polygon.distance(segment) <= self.robot_radius + self.robot_clearance:
                    return False

        return True

    def export_to_json(self) -> str:
        import json
        import re
        import shutil

        ring = self._get_workspace_linearring()
        workspace_poly = Polygon(ring)
        workspace_coords = list(map(list, workspace_poly.exterior.coords))

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
                polys = []
                for poly in geom.geoms:
                    polys.append(
                        {
                            "type": "Polygon",
                            "exterior": list(map(list, poly.exterior.coords)),
                            "interiors": [list(map(list, interior.coords)) for interior in poly.interiors],
                        }
                    )
                obstacles_json.append({"type": "MultiPolygon", "polygons": polys})

        samples_json = [list(pt) for pt in self.sample_points]
        optimal_path_json = [list(pt) for pt in getattr(self, "optimal_path", [])]

        data = {
            "workspace_polygon": workspace_coords,
            "obstacles": obstacles_json,
            "samples": samples_json,
            "optimal_path": optimal_path_json,
            "start": list(self._start),
            "goal": list(self._goal),
        }

        json_str = json.dumps(data, indent=2)

        project_root = Path(__file__).resolve().parents[4]
        base_dir = project_root / "out" / "14"
        base_dir.mkdir(parents=True, exist_ok=True)

        json_dir = base_dir / "json_paths"
        json_dir.mkdir(parents=True, exist_ok=True)

        if not hasattr(self, "_json_folder_cleaned"):
            for f in json_dir.iterdir():
                if f.is_file():
                    f.unlink()
            self._json_folder_cleaned = True

        pattern = re.compile(r"^fmt_samples_(\d+)\.json$")
        existing = [f for f in json_dir.iterdir() if f.is_file()]

        used_nums = []
        for f in existing:
            m = pattern.match(f.name)
            if m:
                used_nums.append(int(m.group(1)))

        next_num = 0 if not used_nums else max(used_nums) + 1
        filename = f"fmt_samples_{next_num}.json"

        output_file = json_dir / filename
        output_file.write_text(json_str, encoding="utf-8")

        return json_str
