import numpy as np
import pandas as pd
from gym.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.callbacks import BaseCallback


def save_log(env: RecordEpisodeStatistics, filename: str):
    score = np.array(env.return_queue)
    length = np.array(env.length_queue)

    results = np.array([score, length]).transpose()
    df = pd.DataFrame(results, columns=["score", "length"])
    df.to_csv(filename)


class RecordTrajectories(BaseCallback):
    """
    Record trajectories for planning.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, verbose=0, output_dir="solutions"):
        super().__init__(verbose)
        self.episode = 0
        self.output_dir = output_dir
        self.file = open(f"{output_dir}/pfile0.solution", "w")

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        env = self.locals["env"].envs[0]

        action = self.locals["actions"][0]
        action = env.decoder.decode_to_planning(action)
        self.file.write(f"({action})\n")

        if env.rounds_left == env.max_rounds:
            self.file.close()
            self.episode += 1
            self.file = open(f"{self.output_dir}/pfile{self.episode}.solution", "w")

        return True
