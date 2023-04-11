from agents.polycraft_agent import PolycraftAgent
from tqdm import tqdm
import config as CONFIG

import os
import sys
import json
import shutil
from pathlib import Path

sys.path.append("planning")
sys.path.append(CONFIG.NSAM_PATH)
from enhsp import ENHSP
from observations.experiments_trajectories_creator import (
    ExperimentTrajectoriesCreator,
    SolverType,
)
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser
from sam_learning.learners.numeric_sam import NumericSAMLearner
import logging

os.environ["CONVEX_HULL_ERROR_PATH"] = "tests/temp_files/ch_error.txt"
logging.root.setLevel(logging.ERROR)


class ExploringSam(PolycraftAgent):
    """
    Agent that learn planning action models.

    args:
        env: environment
        problem: path to the problem file
        fluents_map: path to the fluents map file
        explorer: agent to explore the environment
        save_path: path to save the model
        save_interval: interval to save the model
        output_dir: path to save the trajectories
    """

    def __init__(
        self,
        env,
        domain: str = "planning/basic_minecraft_domain.pddl",
        problem: str = "planning/basic_minecraft_problem.pddl",
        fluents_map: str = "planning/basic_minecraft_fluents_map.json",
        explorer: PolycraftAgent = None,
        save_path: str = "ESAM",
        save_interval: int = 100,
        output_dir: str = "solutions",
    ):
        super().__init__(env)
        self.explorer = explorer

        # N-SAM learner
        self.nsam = CONFIG.NSAM_PATH

        self.output_dir = output_dir
        shutil.copyfile(domain, f"{output_dir}/domain.pddl")
        shutil.copyfile(problem, f"{output_dir}/problem.pddl")
        shutil.copyfile(fluents_map, f"{output_dir}/fluents_map.json")

        # logger
        self.save_path = save_path
        self.save_interval = save_interval
        self.num_timesteps = 0
        self.episode = 0

    # overriding abstract method
    def choose_action(self, state=None) -> int:
        """Use explorer to sample action"""
        return self.explorer.choose_action(state)

    # add learn action, add each episode active sam, if got expert active diffrent choose action
    def learn(self, n_episodes: int):
        """Explore the environment and search for new state to explore.
        if got expert, use expert to choose action"""

        if self.explorer is None:
            raise Exception("E-SAM must have an explorer agent in order to work.")

        file = open(f"{self.output_dir}/pfile0.solution", "w")

        for _ in tqdm(range(n_episodes)):
            state = self.env.reset()
            done = False

            self.last_state = None
            self.last_action = None
            self.pre_state = None
            self.pre_action = None

            # play one episode
            while not done:
                action = self.choose_action(state)

                # save action
                act = self.env.decoder.decode_to_planning(action)
                file.write(f"({act})\n")

                next_state, _, done, _ = self.env.step(action)

                # update if the environment is done and the current obs
                state = next_state

            self.episode += 1

            file.close()
            file = open(f"{self.output_dir}/pfile{self.episode}.solution", "w")

            # add to logger
            self.num_timesteps += 1
            if self.num_timesteps % self.save_interval == 0:
                plan = self.active_nsam()
                self.save(plan)

    def active_nsam(self):
        """Active N-SAM to learn the action model and return the plan"""

        SOLUTIONS_PATH = Path(self.output_dir)

        # create pddl trajectories
        trajectory_creator = ExperimentTrajectoriesCreator(
            domain_file_name="domain.pddl", working_directory_path=SOLUTIONS_PATH
        )
        selected_solver = SolverType.enhsp
        trajectory_creator.fix_solution_files(selected_solver)
        trajectory_creator.create_domain_trajectories(
            problem_path=SOLUTIONS_PATH / "problem.pddl"
        )

        # run nsam
        depot_partial_domain = DomainParser(
            SOLUTIONS_PATH / "domain.pddl", partial_parsing=True
        ).parse_domain()
        depot_problem = ProblemParser(
            SOLUTIONS_PATH / f"problem.pddl", domain=depot_partial_domain
        ).parse_problem()
        with open(SOLUTIONS_PATH / "fluents_map.json", "rt") as json_file:
            depot_fluents_map = json.load(json_file)

        observation_list = []

        i = 0
        while (
            os.path.exists(trajectory := SOLUTIONS_PATH / f"pfile{i}.trajectory")
            and os.path.getsize(trajectory) > 0
        ):
            depot_observation = TrajectoryParser(
                depot_partial_domain, depot_problem
            ).parse_trajectory(SOLUTIONS_PATH / f"pfile{i}.trajectory")
            observation_list.append(depot_observation)
            i += 1

        numeric_sam = NumericSAMLearner(depot_partial_domain, depot_fluents_map)
        learned_model, _ = numeric_sam.learn_action_model(observation_list)
        domain = learned_model.to_pddl()
        domain_location = SOLUTIONS_PATH / f"domain{i}.pddl"
        with open(domain_location, "w") as f:
            f.write(domain)

        # run enhsp
        enhsp = ENHSP()

        plan = enhsp.create_plan(
            domain=f"{os.getcwd()}/{str(domain_location)}",
            problem=f"{os.getcwd()}/{str(SOLUTIONS_PATH / 'problem.pddl')}",
        )

        return plan

    def save(self, plan):
        """Save the plan."""
        if len(plan) == 0:
            print(f"score: 0, length: {self.env.max_rounds}")
        else:
            print(f"score: 1, length: {len(plan)}")

        with open(f"{self.save_path}/plan{self.episode}.txt", "w") as file:
            file.write("\n".join(plan))

    def load(self, path):
        """Change the output_dir to path."""
        self.output_dir = path
