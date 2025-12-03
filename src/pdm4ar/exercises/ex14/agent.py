from gc import collect
from mimetypes import init
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from dg_commons import PlayerName
from dg_commons.sim import InitSimGlobalObservations, InitSimObservations, SharedGoalObservation, SimObservations
from dg_commons.sim.agents import Agent, GlobalPlanner
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.diff_drive import DiffDriveCommands
from dg_commons.sim.models.diff_drive_structures import DiffDriveGeometry, DiffDriveParameters
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.utils import extract_2d_position_from_state
from numpydantic import NDArray
from pydantic import BaseModel

from pdm4ar.exercises.ex14 import fmt
from pdm4ar.exercises.ex14.fmt import FastMarchingTree


class GlobalPlanMessage(BaseModel):
    # TODO: modify/add here the fields you need to send your global plan
    fake_id: int
    fake_name: str
    fake_np_data: NDArray  # If you need to send numpy arrays, annotate them with NDArray


@dataclass(frozen=True)
class Pdm4arAgentParams:
    param1: float = 10


class Pdm4arAgent(Agent):
    """This is the PDM4AR agent.
    Do *NOT* modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task"""

    name: PlayerName
    goal: PlanningGoal
    static_obstacles: Sequence[StaticObstacle]
    sg: DiffDriveGeometry
    sp: DiffDriveParameters

    def __init__(self):
        # feel free to remove/modify  the following
        self.params = Pdm4arAgentParams()

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        pass

    def on_receive_global_plan(
        self,
        serialized_msg: str,
    ):
        # TODO: process here the received global plan
        global_plan = GlobalPlanMessage.model_validate_json(serialized_msg)

    def get_commands(self, sim_obs: SimObservations) -> DiffDriveCommands:
        """This method is called by the simulator every dt_commands seconds (0.1s by default).
        Do not modify the signature of this method.

        For instance, this is how you can get your current state from the observations:
        my_current_state: DiffDriveState = sim_obs.players[self.name].state
        :param sim_obs:
        :return:
        """

        # TODO: implement here your planning stack
        omega1 = random.random() * self.params.param1
        omega2 = random.random() * self.params.param1

        return DiffDriveCommands(omega_l=omega1, omega_r=omega2)


class Pdm4arGlobalPlanner(GlobalPlanner):
    """
    This is the Global Planner for PDM4AR
    Do *NOT* modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task
    """

    def __init__(self):
        pass

    def send_plan(self, init_sim_obs: InitSimGlobalObservations) -> str:

        adj_mat, labels = self.create_adjacancy_matrix(init_sim_obs)
        print("Adjacency Matrix:")
        print(adj_mat)
        print("Labels:")
        print(labels)

        """import random

        # number of random tests
        num_tests = 20

        fmt = FastMarchingTree(initObservations=init_sim_obs, n_samples=10000)

        for i in range(num_tests):
            # generate random points
            start = (random.uniform(-10, 10), random.uniform(-10, 10))
            goal = (random.uniform(-10, 10), random.uniform(-10, 10))

            path, length = fmt.plan_path(start, goal)
            print(f"Test {i+1}: start={start}, goal={goal}, length={length}")"""

        # TODO: implement here your global planning stack.
        global_plan_message = GlobalPlanMessage(
            fake_id=1,
            fake_name="agent_1",
            fake_np_data=np.array([[1, 2, 3], [4, 5, 6]]),
        )
        return global_plan_message.model_dump_json(round_trip=True)

    def create_adjacancy_matrix(self, init_sim_obs: InitSimGlobalObservations) -> tuple[np.ndarray, list[str]]:
        """
        Create a symmetric distance matrix between players, shared goals and collection points.
        Distances are computed via Fast Marching Tree paths. Returns the matrix and the labels.
        """

        fmt = FastMarchingTree(initObservations=init_sim_obs, n_samples=5000)

        labeled_points: list[tuple[str, np.ndarray]] = []

        for player_name in sorted(init_sim_obs.initial_states.keys()):
            state = init_sim_obs.initial_states[player_name]
            labeled_points.append((f"player:{player_name}", extract_2d_position_from_state(state)))

        if init_sim_obs.shared_goals:
            for goal_id in sorted(init_sim_obs.shared_goals.keys()):
                goal = init_sim_obs.shared_goals[goal_id]
                centroid = goal.polygon.centroid
                labeled_points.append((f"goal:{goal_id}", np.array([centroid.x, centroid.y], dtype=float)))

        if init_sim_obs.collection_points:
            for collection_id in sorted(init_sim_obs.collection_points.keys()):
                collection = init_sim_obs.collection_points[collection_id]
                centroid = collection.polygon.centroid
                labeled_points.append((f"collection:{collection_id}", np.array([centroid.x, centroid.y], dtype=float)))

        n = len(labeled_points)
        labels = [label for label, _ in labeled_points]
        adjacency = np.zeros((n, n), dtype=float)

        for i in range(n):
            pi = tuple(labeled_points[i][1].tolist())
            for j in range(i + 1, n):
                pj = tuple(labeled_points[j][1].tolist())
                _, length = fmt.plan_path(pi, pj)
                adjacency[i, j] = length
                adjacency[j, i] = length

        return adjacency, labels
