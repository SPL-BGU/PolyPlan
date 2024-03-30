import os
import sys

import time
from stable_baselines3.common.evaluation import evaluate_policy

from envs import MaskedMinecraft

from stable_baselines3 import DQN
from sb3_contrib.ppo_mask import MaskablePPO
from gym.wrappers import RecordEpisodeStatistics

from utils import Logger

import numpy as np
import random
import torch

SEED = 63
np.random.seed(SEED)  # random seed for reproducibility
random.seed(SEED)
torch.manual_seed(SEED)


def main(map_type, map_size, learning_method, steps_per_episode, steps_per_map):
    minecraft = MaskedMinecraft

    if map_type == "basic":
        env = minecraft(
            visually=False,
            start_pal=True,
            keep_alive=False,
            max_steps=steps_per_episode,
        )
    elif map_type == "advanced":
        env = minecraft(
            visually=False,
            start_pal=True,
            keep_alive=False,
            max_steps=steps_per_episode,
            map_size=map_size,
        )

    map_size = f"{map_size}X{map_size}"
    maps = 50

    train_idx = list(range(maps))

    output_directory_path = f"{os.getcwd()}/dataset/{map_size}"

    # make log directory
    postfix = f"{learning_method}/{map_type}_{map_size}"
    logdir, models_dir = Logger.create_logdir(postfix, indexing=True)

    rec_dir = f"{logdir}/solutions"
    if not os.path.exists(rec_dir):
        os.makedirs(rec_dir)

    renv = RecordEpisodeStatistics(env, deque_size=5000)

    if learning_method == "DQN":
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            learning_starts=32,
            exploration_fraction=1,
            train_freq=(1, "episode"),
            target_update_interval=32,
            batch_size=32,
            tensorboard_log=logdir,
            seed=SEED,
        )
    elif learning_method == "PPO":
        model = MaskablePPO(
            "MlpPolicy",
            renv,
            verbose=1,
            n_steps=steps_per_episode,
            batch_size=steps_per_episode,
            stats_window_size=1,
            ent_coef=0.01,
            gamma=0.999,
            vf_coef=0.65,
            max_grad_norm=1.0,
            tensorboard_log=logdir,
            seed=SEED,
        )

    callback = Logger.RecordTrajectories(output_dir=rec_dir)

    for index, problem_index in enumerate(train_idx):
        rec_dir_index = f"{rec_dir}/{index}"
        if not os.path.exists(rec_dir_index):
            os.makedirs(rec_dir_index)
        callback.update_output_dir(rec_dir_index)

        t1 = time.time()
        domain_path = f"{output_directory_path}/map_instance_{problem_index}.json"
        env.set_domain(domain_path)
        renv.reset()
        model.learn(
            total_timesteps=steps_per_map,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=True,
        )
        model.save(f"{models_dir}/{index}.zip")
        t2 = time.time()
        print(f"Time to train: {t2 - t1}")

    env.close()


if __name__ == "__main__":
    if len(sys.argv) == 6:
        map_type = sys.argv[1]
        map_size = int(sys.argv[2])
        learning_method = sys.argv[3]
        steps_per_episode = int(sys.argv[4])
        steps_per_map = int(sys.argv[5])

        if (
            not os.path.isdir(f"{os.getcwd()}/dataset/{map_size}X{map_size}")
            or map_type not in ["basic", "advanced"]
            or learning_method not in ["PPO", "DQN"]
            or steps_per_map % steps_per_episode != 0
        ):
            print("Please provide valid command-line argument.")
            print(
                "Example: python playground_online.py map_type[basic/advanced] map_size[<int>] algorithm[PPO/DQN] steps_per_episode[<int>] steps_per_map[<int>]"
            )
        else:
            main(map_type, map_size, learning_method, steps_per_episode, steps_per_map)
    else:
        print("Please provide a variable as a command-line argument.")
        print(
            "Example: python playground_online.py map_type[basic/advanced] map_size[<int>] algorithm[PPO/DQN] steps_per_episode[<int>] steps_per_map[<int>]"
        )
