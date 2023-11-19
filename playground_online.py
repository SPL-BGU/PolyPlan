import os
import sys

import time
from stable_baselines3.common.evaluation import evaluate_policy

from envs import (
    BasicMinecraft,
    IntermediateMinecraft,
    AdvancedMinecraft,
    MaskedMinecraft,
)
from polycraft_policy import (
    PolycraftPPOPolicy,
    PolycraftMaskedPPOPolicy,
    PolycraftDQNPolicy,
)

from stable_baselines3 import PPO, DQN
from sb3_contrib.ppo_mask import MaskablePPO

from utils import Logger

import pandas as pd
import numpy as np
import random
import torch

SEED = 63
np.random.seed(SEED)  # random seed for reproducibility
random.seed(SEED)
torch.manual_seed(SEED)


def evaluate(env, model, test_set, id, map_size):
    """Evaluate the trained model"""
    avg = []
    output_directory_path = f"{os.getcwd()}/dataset/{map_size}"
    for problem_index in test_set:
        domain_path = f"{output_directory_path}/map_instance_{problem_index}.json"
        env.set_domain(domain_path)

        rewards, _ = evaluate_policy(
            model=model,
            env=env,
            n_eval_episodes=1,
            return_episode_rewards=True,
            deterministic=True,
        )
        avg.append(sum(rewards) / len(rewards))

    # print("Average Reward:", avg)
    print(f"{id}.Total Average Reward:", sum(avg) / len(avg))
    return sum(avg) / len(avg)


def main(map_type, map_size, learning_method, fold, max_steps):
    if map_type == "basic":
        env_index = 0  # 0: BasicMinecraft, 1: IntermediateMinecraft, 2: AdvancedMinecraft, 3: MaskedMinecraft
    elif map_type == "advanced":
        env_index = 2  # 0: BasicMinecraft, 1: IntermediateMinecraft, 2: AdvancedMinecraft, 3: MaskedMinecraft

    minecraft = [
        BasicMinecraft,
        IntermediateMinecraft,
        AdvancedMinecraft,
        MaskedMinecraft,
    ][env_index]

    if map_type == "basic":
        env = minecraft(
            visually=False, start_pal=True, keep_alive=False, max_steps=max_steps
        )
    elif map_type == "advanced":
        env = minecraft(
            visually=False,
            start_pal=True,
            keep_alive=False,
            max_steps=max_steps,
            map_size=map_size,
        )

    map_size = f"{map_size}X{map_size}"
    chunk_size = 160
    timesteps = 128 * chunk_size

    j = -1
    df = pd.read_csv("kfolds.csv")
    for _, row in df.iterrows():
        # skip to the fold
        j += 1
        if j < fold:
            continue

        train_idx = eval(row["train_idx"])
        val_idx = eval(row["val_idx"])

        output_directory_path = f"{os.getcwd()}/dataset/{map_size}"

        # make log directory
        postfix = f"{learning_method}/{map_type}_{map_size}/fold_{fold}"
        logdir, models_dir = Logger.create_logdir(postfix)

        rec_dir = f"{logdir}/solutions"
        if not os.path.exists(rec_dir):
            os.makedirs(rec_dir)
        callback = Logger.RecordTrajectories(output_dir=rec_dir)

        if learning_method == "DQN":
            model = DQN(
                PolycraftDQNPolicy,  # "MlpPolicy"
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
            model = PPO(
                PolycraftPPOPolicy,
                env,
                verbose=1,
                n_steps=32,
                batch_size=32,
                tensorboard_log=logdir,
                seed=SEED,
            )

        file = open(f"{logdir}/results.txt", "w", encoding="utf-8")

        for index, problem_start_index in enumerate(
            range(0, len(train_idx), chunk_size)
        ):
            t1 = time.time()
            for problem_index in train_idx[
                problem_start_index : problem_start_index + chunk_size
            ]:
                domain_path = (
                    f"{output_directory_path}/map_instance_{problem_index}.json"
                )
                env.set_domain(domain_path)
                model.learn(
                    total_timesteps=timesteps,
                    callback=callback,
                )
            model.save(f"{models_dir}/{index}.zip")
            t2 = time.time()
            print(f"Time to train: {t2 - t1}")
            t1 = time.time()
            avg = evaluate(env, model, val_idx, index, map_size)
            t2 = time.time()
            print(f"Time to evaluate: {t2 - t1}")
            file.write(f"{avg}\n")

        file.close()
        break
    env.close()


if __name__ == "__main__":
    if len(sys.argv) == 6:
        max_steps = int(sys.argv[5])
    else:
        max_steps = 32

    if len(sys.argv) == 5:
        map_type = sys.argv[1]
        map_size = int(sys.argv[2])
        learning_method = sys.argv[3]
        fold = int(sys.argv[4])

        if (
            map_type not in ["basic", "advanced"]
            or map_size not in [6, 10]
            or (map_type == "basic" and map_size != 6)
            or learning_method not in ["PPO", "DQN"]
            or fold not in list(range(5))
            or max_steps % 32 != 0
        ):
            print("Please provide valid command-line argument.")
            print(
                "Example: python playground_online.py map_type[basic/advanced] map_size[6/10] algorithm[PPO/DQN] fold[0-4] optional_max_steps[32*X]"
            )
        else:
            main(map_type, map_size, learning_method, fold, max_steps)
    else:
        print("Please provide a variable as a command-line argument.")
        print(
            "Example: python playground_online.py map_type[basic/advanced] map_size[6/10] algorithm[PPO/DQN] fold[0-4] optional_max_steps[32*X]"
        )
