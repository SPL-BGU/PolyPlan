import os
import sys
import pickle

import time
from stable_baselines3.common.evaluation import evaluate_policy

from utils.wooden_sword.macro_actions_decoder import MacroActionsDecoder
from utils.wooden_sword.advanced_actions_decoder import AdvancedActionsDecoder

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
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.algorithms import bc
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util import logger as imit_logger

from stable_baselines3.common.buffers import ReplayBuffer

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
    output_directory_path = f"{os.getcwd()}/dataset/wooden_sword/{map_size}"
    for problem_index in test_set:
        domain_path = f"{output_directory_path}/map_instance_{problem_index}.json"
        env.set_domain(domain_path)

        rewards, _ = evaluate_policy(
            model=model,
            env=env,
            n_eval_episodes=1,
            return_episode_rewards=True,
            deterministic=False,
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
            decoder_class=MacroActionsDecoder,
            visually=False,
            start_pal=False,
            keep_alive=True,
            max_steps=max_steps,
        )
    elif map_type == "advanced":
        env = minecraft(
            decoder_class=AdvancedActionsDecoder,
            visually=False,
            start_pal=False,
            keep_alive=True,
            max_steps=max_steps,
            map_size=map_size,
        )

    map_size = f"{map_size}X{map_size}"
    chunk_size = 32
    timesteps = 128 * chunk_size

    # TODO: remove
    output_directory_path = f"{os.getcwd()}/dataset/wooden_sword/{map_size}"
    domain_path = f"{output_directory_path}/map_instance_0.json"
    env.set_domain(domain_path)
    env.reset()
    # TODO: remove

    j = -1
    df = pd.read_csv("modified_kfolds2.csv")
    for _, row in df.iterrows():
        # skip to the fold
        j += 1
        if j < fold:
            continue

        train_idx = eval(row["train_idx"])
        val_idx = eval(row["val_idx"])

        # make log directory
        postfix = f"{learning_method}/{map_type}_{map_size}/fold_{fold}"
        logdir, models_dir = Logger.create_logdir(postfix)

        rollouts = []

        if learning_method == "BC":
            policy = PolycraftPPOPolicy(
                env.observation_space, env.action_space, lambda _: 3e-4
            )

            bc_trainer = bc.BC(
                batch_size=32,
                observation_space=env.observation_space,
                action_space=env.action_space,
                custom_logger=imit_logger.configure(folder=logdir),
                policy=policy,
                rng=SEED,
            )
        elif learning_method == "DQN":
            dqn_model = DQN(
                PolycraftDQNPolicy,  # "MlpPolicy"
                env,
                verbose=1,
                learning_rate=3e-4,
                learning_starts=256,
                exploration_fraction=1,
                train_freq=(1, "episode"),
                target_update_interval=32,
                batch_size=32,
                tensorboard_log=logdir,
                seed=SEED,
            )

        elif learning_method == "GAIL":
            ppo_model = PPO(
                PolycraftPPOPolicy,
                env,
                verbose=1,
                n_steps=32,
                batch_size=32,
                tensorboard_log=logdir,
                seed=SEED,
            )

            buffer_size = 2048
            venv = DummyVecEnv([lambda: RolloutInfoWrapper(env)])
            reward_net = BasicRewardNet(
                observation_space=env.observation_space,
                action_space=env.action_space,
                use_next_state=True,  # determinestic world
                use_done=True,
                normalize_input_layer=RunningNorm,
            )
            gail_trainer = GAIL(
                demonstrations=None,
                demo_batch_size=1,
                gen_replay_buffer_capacity=buffer_size,
                venv=venv,
                gen_algo=ppo_model,
                reward_net=reward_net,
                log_dir=logdir,
                init_tensorboard=True,
                init_tensorboard_graph=True,
                allow_variable_horizon=True,
                custom_logger=imit_logger.configure(folder=logdir),
            )

        file = open(f"{logdir}/results.txt", "w", encoding="utf-8")

        for index, problem_start_index in enumerate(
            range(0, len(train_idx), chunk_size)
        ):
            for problem_index in train_idx[
                problem_start_index : problem_start_index + chunk_size
            ]:
                with open(
                    f"{os.getcwd()}/dataset/wooden_sword/{map_size}/{map_type}_map_instance_{problem_index}.pkl",
                    "rb",
                ) as fp:
                    pk = pickle.load(fp)
                    rollouts += pk

            if learning_method == "BC":
                bc_trainer.set_demonstrations(rollouts)
                t1 = time.time()
                bc_trainer.train(n_batches=timesteps)  # n_batches n_epochs
                t2 = time.time()
                print(f"Time to train: {t2 - t1}")
                bc_trainer.save_policy(f"{models_dir}/{index}.zip")
                model = bc.reconstruct_policy(f"{models_dir}/{index}.zip")
            elif learning_method == "DQN":
                replay_buffer = ReplayBuffer(
                    buffer_size=len(rollouts) * 16,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                )

                for trajectory in rollouts:
                    # unwrafted trajectory
                    obs = trajectory.obs
                    action = trajectory.acts
                    reward = trajectory.rews

                    unwrafted_trajectory = [
                        (obs[i], action[i], reward[i], obs[i + 1], bool(reward[i]))
                        for i in range(len(obs) - 1)
                    ]

                    # add to replay buffer
                    for step in unwrafted_trajectory:
                        obs, action, reward, next_obs, done = step
                        replay_buffer.add(obs, next_obs, action, reward, done, [{}])
                dqn_model.replay_buffer = replay_buffer

                dqn_model.learn(total_timesteps=0)
                t1 = time.time()
                dqn_model.train(timesteps, 32)
                t2 = time.time()
                print(f"Time to train: {t2 - t1}")
                dqn_model.save(f"{models_dir}/{index}.zip")

                model = DQN.load(f"{models_dir}/{index}.zip", env=env)

            elif learning_method == "GAIL":
                gail_trainer.set_demonstrations(rollouts)
                t1 = time.time()
                gail_trainer.train(total_timesteps=timesteps)
                t2 = time.time()
                print(f"Time to train: {t2 - t1}")
                ppo_model.save(f"{models_dir}/{index}.zip")
                model = PPO.load(f"{models_dir}/{index}.zip", env=env)

            t1 = time.time()
            avg = evaluate(env, model, val_idx, index, map_size)
            t2 = time.time()
            print(f"Time to evaluate: {t2 - t1}")
            file.write(f"{avg}\n")
            file.flush()

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
            not os.path.isdir(
                f"{os.getcwd()}/dataset/wooden_sword/{map_size}X{map_size}"
            )
            or map_type not in ["basic", "advanced"]
            or learning_method not in ["BC", "DQN", "GAIL"]
            or fold not in list(range(5))
            or max_steps % 32 != 0
        ):
            print("Please provide valid command-line argument.")
            print(
                "Example: python playground_offline.py map_type[basic/advanced] map_size[M] algorithm[BC/DQN/GAIL] fold[0-4] optional_max_steps[32*X]"
            )
        else:
            main(map_type, map_size, learning_method, fold, max_steps)
    else:
        print("Please provide a variable as a command-line argument.")
        print(
            "Example: python playground_offline.py map_type[basic/advanced] map_size[M] algorithm[BC/DQN/GAIL] fold[0-4] optional_max_steps[32*X]"
        )
