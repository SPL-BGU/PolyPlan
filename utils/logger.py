import os
import numpy as np
import pandas as pd
from gym import Wrapper
from gym.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


class RecordTrajectories(BaseCallback):
    """
    Record trajectories for planning.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, verbose=0, output_dir="solutions"):
        super().__init__(verbose)
        self.episodes = 1
        self.output_dir = output_dir
        self.file = open(f"{output_dir}/pfile1.trajectory", "w")
        self.env = None

    def translate(self, state):
        output = ""
        for key, value in state.items():
            if key == "gameMap":
                for i, cell_value in enumerate(value):
                    if cell_value == 0:
                        output += "(air_cell cell{}) ".format(i)
                    elif cell_value == 1:
                        output += "(tree_cell cell{}) ".format(i)
                    elif cell_value == 2:
                        output += "(crafting_table_cell crafting_table) "
            elif key == "inventory":
                output += "(= (count_log_in_inventory ) {}) ".format(int(value[0]))
                output += "(= (count_planks_in_inventory ) {}) ".format(int(value[1]))
                output += "(= (count_stick_in_inventory ) {}) ".format(int(value[2]))
                output += (
                    "(= (count_sack_polyisoprene_pellets_in_inventory ) {}) ".format(
                        int(value[3])
                    )
                )
                output += "(= (count_tree_tap_in_inventory ) {}) ".format(int(value[4]))
                if int(value[5]) > 0:
                    output += "(have_pogo_stick) "
            elif key == "position":
                output += "(position cell{})".format(int(value[0]))
            elif key == "treeCount":
                output += "(= (trees_in_map ) {}) ".format(int(value[0]))

        output += ")"

        return output

    def get_env(self):
        """Get the current environment"""
        if self.env is None:
            if "env" in self.locals:
                env = self.locals["env"].envs[0]
            else:
                env = self.locals["self"].env.envs[0]
            while issubclass(type(env), Wrapper):
                env = env.env
            self.env = env
        return self.env

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        env = self.get_env()

        action = self.locals["actions"][0]
        action = env.decoder.decode_to_planning(action)
        self.file.write(f"(operator: ({action}))\n")
        translate = self.translate(env._state)
        self.file.write(f"(:state {translate}\n")

        if env.rounds_left == env.max_rounds or env.done:
            self.file.write(f")\n")
            self.file.close()
            self.episodes += 1
            self.file = open(f"{self.output_dir}/pfile{self.episodes}.trajectory", "w")

        return True

    def _on_rollout_start(self) -> None:
        env = self.get_env()

        translate = self.translate(env._state)
        self.file.write(f"((:init {translate}\n")

    def _on_rollout_end(self) -> None:
        env = self.locals["env"].envs[0]

        if type(env) == Monitor:
            env = env.env

        if type(env) == RecordEpisodeStatistics:
            score = np.array(env.return_queue)
            length = np.array(env.length_queue)

            results = np.array([score, length]).transpose()
            df = pd.DataFrame(results, columns=["score", "length"])
            df.to_csv(f"{self.output_dir}/summary.csv")

    def _on_training_end(self) -> None:
        os.remove(f"{self.output_dir}/pfile{self.episodes}.trajectory")
