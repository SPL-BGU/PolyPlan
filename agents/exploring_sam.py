from agents import PolycraftAgent, FixedScriptAgent
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.type_aliases import MaybeCallback
from utils import Logger
import config as CONFIG
from config import ErrorFlag


import os
import re
import sys
import json
import time
import shutil
from pathlib import Path

sys.path.append("planning")
sys.path.append(CONFIG.NSAM_PATH)
from planning import MetricFF, validator, shorter_plan
from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser
from trajectory_creators import ExperimentTrajectoriesCreator
from utilities import SolverType
from sam_learning.learners import NumericSAMLearner
import logging

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
        self.error_flag = ErrorFlag.NO_ERROR

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
        self.saved_explorer = None

        self.observation_count = 0

        self.output_dir = Path(output_dir).absolute()
        shutil.copyfile(domain, f"{output_dir}/domain.pddl")
        shutil.copyfile(problem, f"{output_dir}/problem.pddl")
        shutil.copyfile(fluents_map, f"{output_dir}/fluents_map.json")
        self.domain_parser = DomainParser(
            Path(self.output_dir) / "domain.pddl", partial_parsing=True
        ).parse_domain()
        with open(Path(self.output_dir) / "fluents_map.json", "rt") as json_file:
            self.fluents_map = json.load(json_file)

        # N-SAM learner
        self.numeric_sam = NumericSAMLearner(self.domain_parser)
        self.last_short_observation = None
        self.planner = MetricFF()
        self.eval_mode = False

        # logger
        self.save_path = save_path
        self.save_interval = save_interval
        self.error_flag = ErrorFlag.NO_ERROR
        self.time_to_plan = -1
        self.time_to_learn = -1

    def update_problem(self, problem: str) -> None:
        """Update the problem file"""
        shutil.copyfile(problem, f"{self.output_dir}/problem.pddl")

    def eval(self, toggle: bool = True) -> None:
        if toggle == self.eval_mode:
            return

        if toggle:
            self.saved_explorer = self.explorer
            self.explorer = FixedScriptAgent(
                self.env,
                script=self.active_nsam(),
            )
        else:
            self.explorer = self.saved_explorer
            self.saved_explorer = None

        self.eval_mode = toggle

    def update_fixed_explorer(
        self,
        use_fluents_map=False,
        env_is_reset=False,
        run_planner=True,
        run_shortening=False,
    ) -> bool:
        try:
            # run nsam
            plan = self.active_nsam(
                use_fluents_map=use_fluents_map,
                run_planner=run_planner,
                run_shortening=run_shortening,
            )
            if len(plan) == 0:
                return False

            # gym agent to execute the plan
            self.explorer = FixedScriptAgent(
                self.env,
                script=plan,
                env_is_reset=env_is_reset,
            )
            return True
        except:
            return False

    # overriding abstract method
    def choose_action(self, state=None) -> int:
        """Use explorer to sample action"""
        return self.explorer.choose_action(state)

    def learn(self, total_timesteps: int, callback: MaybeCallback = None):
        """
        Learn the policy for a total of `total_timesteps` steps.
        Standart RL learn method with callback to save the plan.
        """

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

    def active_nsam(
        self,
        parser_trajectories=False,
        use_fluents_map=False,
        run_planner=True,
        run_shortening=False,
    ) -> list:
        """Active N-SAM to learn the action model and return the plan"""

        SOLUTIONS_PATH = Path(self.output_dir)
        self.time_to_plan = -1
        self.time_to_learn = -1

        # create pddl trajectories
        if parser_trajectories:
            trajectory_creator = ExperimentTrajectoriesCreator(
                domain_file_name="domain.pddl", working_directory_path=SOLUTIONS_PATH
            )
            selected_solver = SolverType.enhsp
            trajectory_creator.fix_solution_files(selected_solver)
            trajectory_creator.create_domain_trajectories()

        # load trajectories
        all_files = os.listdir(SOLUTIONS_PATH)
        trajectory_files = sorted(
            [file for file in all_files if file.endswith(".trajectory")],
            key=lambda s: [
                int(text) if text.isdigit() else text.lower()
                for text in re.split(r"(\d+)", s)
            ],
        )
        if (
            trajectory_files
            and os.path.getsize(SOLUTIONS_PATH / trajectory_files[-1]) == 0
        ):
            trajectory_files.pop()

        if len(trajectory_files) == 0:
            self.error_flag = ErrorFlag.ERROR
            return []

        # parse trajectories
        observation_list = []
        for trajectory in trajectory_files[self.observation_count :]:
            observation = TrajectoryParser(self.domain_parser).parse_trajectory(
                SOLUTIONS_PATH / f"{trajectory}"
            )
            observation_list.append(observation)

        self.observation_count = len(trajectory_files)

        # update numeric sam
        if use_fluents_map:
            self.numeric_sam.relevant_fluents = self.fluents_map
        else:
            self.numeric_sam.relevant_fluents = None
        learned_model, learning_metadata = self.numeric_sam.learn_action_model(
            observation_list
        )
        self.time_to_learn = learning_metadata.pop("learning_time", 0)

        # output learned model
        learned_domain = learned_model.to_pddl()
        domain_location = SOLUTIONS_PATH / f"domain{len(trajectory_files)}.pddl"
        with open(domain_location, "w") as f:
            f.write(learned_domain)

        # update problem files
        plan = []
        self.error_flag = ErrorFlag.NO_ERROR
        problem = f"{str(SOLUTIONS_PATH / 'problem.pddl')}"
        domain = f"{str(domain_location)}"
        tloc = f"{SOLUTIONS_PATH}/temp_plan.txt"
        odomain = f"{SOLUTIONS_PATH}/domain.pddl"

        # shorter observation list
        if run_shortening:
            goal = "craft_wooden_pogo"  # "craft_wooden_sword" "craft_wooden_pogo"
            if len(observation_list) != 0:
                shorter_observation_list = []
                for observation in observation_list:
                    if (
                        (
                            goal
                            not in observation.components[-1].grounded_action_call.name
                        )
                        or (
                            observation.components[-1].previous_state
                            == observation.components[-1].next_state
                        )
                        or (observation == self.last_short_observation)
                    ):
                        continue

                    observation = shorter_plan(
                        self.env,
                        observation,
                        problem,
                        self.domain_parser,
                        learned_model,
                    )

                    shorter_observation_list.append(observation)
                    self.last_short_observation = observation

            # if we didn't found a goal
            if self.last_short_observation == None:
                self.error_flag = ErrorFlag.ERROR
                return []

        # run planner
        if run_planner:
            for numeric_precision in [0.1]:
                self.time_to_plan = time.time()
                plan = self.planner.create_plan(
                    domain=domain,
                    problem=problem,
                    timeout=300,
                    flag=f"-t {numeric_precision}",
                )
                self.time_to_plan = time.time() - self.time_to_plan

                self.error_flag = self.planner.error_flag

                if len(plan) > 0:
                    # Validate the plan
                    with open(tloc, "w") as tfile:
                        tfile.write("\n".join(plan))

                    valid = validator(Path(odomain), Path(problem), Path(tloc))
                    if valid:
                        self.error_flag = ErrorFlag.NO_ERROR
                        break
                    else:
                        self.error_flag = ErrorFlag.INVALID_PLAN
                        plan = []
                        break

        # if plan is empty, return the shorter trajectory
        if run_shortening and (len(plan) == 0):
            if (goal not in observation.components[-1].grounded_action_call.name) or (
                observation.components[-1].previous_state
                == observation.components[-1].next_state
            ):
                return []

            self.error_flag = ErrorFlag.FOUND_BY_SHORTEN
            plan = [
                str(component.grounded_action_call)
                for component in observation.components
            ]

            with open(tloc, "w") as tfile:
                tfile.write("\n".join(plan))

            valid = validator(Path(odomain), Path(problem), Path(tloc))
            if not valid:
                self.error_flag = ErrorFlag.NO_SOLUTION
                plan = []

        return plan

    def save_plan(self, plan):
        """Save the plan"""
        if len(plan) == 0:
            print(f"score: 0, length: {self.env.max_steps}")
        else:
            print(f"score: 1, length: {len(plan)}")

        with open(f"{self.save_path}/plan{self.episodes}.txt", "w") as file:
            file.write("\n".join(plan))

    def load_plan(self, path):
        """Change the output_dir to path"""
        self.output_dir = path
