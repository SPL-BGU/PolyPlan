from agents.polycraft_agent import PolycraftAgent
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.type_aliases import MaybeCallback
from utils import Logger
import config as CONFIG

import os
import sys
import json
import shutil
from pathlib import Path

sys.path.append("planning")
sys.path.append(CONFIG.NSAM_PATH)
from metric_ff import MetricFF as solver
from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser
from sam_learning.numeric_sam import NumericSAMLearner
import logging

os.environ["CONVEX_HULL_ERROR_PATH"] = "temp_files/ch_error.txt"
logging.root.setLevel(logging.ERROR)


class Explore(BaseCallback):
    """
    auxiliary function for exploring the environment

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    :param save_interval: interval to activate NSAM
    """

    def __init__(self, verbose=0, save_interval=1):
        super().__init__(verbose)
        self.episodes = 0
        self.save_interval = save_interval

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.episodes += 1
        if self.episodes % self.save_interval == 0:
            plan = self.model.active_nsam()
            self.model.save_plan(plan)


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
        self.solver = solver()
        self.eval = False

        self.output_dir = output_dir
        shutil.copyfile(domain, f"{output_dir}/domain.pddl")
        shutil.copyfile(problem, f"{output_dir}/problem.pddl")
        shutil.copyfile(fluents_map, f"{output_dir}/fluents_map.json")

        # logger
        self.save_path = save_path
        self.save_interval = save_interval

    # overriding abstract method
    def choose_action(self, state=None) -> int:
        """Use explorer to sample action"""
        # TODO: add if in eval mode use solver
        return self.explorer.choose_action(state)

    def learn(self, total_timesteps: int, callback: MaybeCallback = None):
        """Learn the policy for a total of `total_timesteps` steps."""

        if self.explorer is None:
            raise Exception("E-SAM must have an explorer agent in order to work.")

        explore_callback = Explore(save_interval=self.save_interval)
        logger_callback = Logger.RecordTrajectories(output_dir=self.output_dir)
        self.eval = False

        if callback is None:
            callback = [logger_callback, explore_callback]
        else:
            if isinstance(callback, CallbackList):
                callback = callback.callbacks
            if isinstance(callback, list):
                for cb in callback:
                    if isinstance(cb, Logger.RecordTrajectories):
                        callback.remove(cb)  # remove duplicate callback
                callback.append(logger_callback)
                callback.append(explore_callback)

        super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
        )

    def active_nsam(self) -> list:
        """Active N-SAM to learn the action model and return the plan"""

        SOLUTIONS_PATH = Path(self.output_dir)

        # run nsam
        domain = DomainParser(
            SOLUTIONS_PATH / "domain.pddl", partial_parsing=True
        ).parse_domain()
        with open(SOLUTIONS_PATH / "fluents_map.json", "rt") as json_file:
            fluents_map = json.load(json_file)

        observation_list = []

        i = 1
        while (
            os.path.exists(trajectory := SOLUTIONS_PATH / f"pfile{i}.trajectory")
            and os.path.getsize(trajectory) > 0
        ):
            observation = TrajectoryParser(domain).parse_trajectory(
                SOLUTIONS_PATH / f"pfile{i}.trajectory"
            )
            observation_list.append(observation)
            i += 1

        numeric_sam = NumericSAMLearner(domain, fluents_map)
        learned_model, _ = numeric_sam.learn_action_model(observation_list)
        learned_domain = learned_model.to_pddl()
        domain_location = SOLUTIONS_PATH / f"domain{i-1}.pddl"
        with open(domain_location, "w") as f:
            f.write(learned_domain)

        # run solver
        plan = self.solver.create_plan(
            domain=f"{os.getcwd()}/{str(domain_location)}",
            problem=f"{os.getcwd()}/{str(SOLUTIONS_PATH / 'problem.pddl')}",
        )

        return plan

    def save_plan(self, plan):
        """Save the plan"""
        if len(plan) == 0:
            print(f"score: 0, length: {self.env.max_rounds}")
        else:
            print(f"score: 1, length: {len(plan)}")

        with open(f"{self.save_path}/plan{self.episodes}.txt", "w") as file:
            file.write("\n".join(plan))

    def load_plan(self, path):
        """Change the output_dir to path"""
        self.output_dir = path
