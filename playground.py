import os
import pickle
import shutil

import time
import cProfile, pstats
import config as CONFIG
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

from agents import QLearningAgent
from agents import FixedScriptAgent

from planning.metric_ff import MetricFF as solver

from stable_baselines3 import PPO, DQN
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from imitation.algorithms import bc
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util import logger as imit_logger

from gym.wrappers import RecordEpisodeStatistics
from utils import Logger

import numpy as np
import random
import torch

SEED = 63
np.random.seed(SEED)  # random seed for reproducibility
random.seed(SEED)
torch.manual_seed(SEED)


def train_rl_agent(
    env,
    learning_method: str,
    timesteps: int = 1024,
    epoch: int = 256,
    batch_size: int = 64,
    record_trajectories: bool = False,
    save_freq: int = 1024,
):
    """Train RL agent using PPO, BC, or GAIL

    Args:
        learning_method (str): one of "BC", "PPO", "GAIL"
        timesteps (int, optional): number of timesteps to train. Defaults to 1024.
        epoch (int, optional): how much steps to until update the net. Defaults to 256.
        batch_size (int, optional): size of the sub-updated in each epoch. Defaults to 64.
        record_trajectories (bool, optional): whether to record trajectories for planning. Defaults to False.
        save_freq (int, optional): how often to save the model. Defaults to 1024.
    """

    if learning_method not in ["BC", "DQN", "PPO", "GAIL", "Masked-PPO"]:
        raise ValueError(
            "learning method must be one of BC, DQN, PPO, GAIL, Masked-PPO"
        )

    env = RecordEpisodeStatistics(env, deque_size=5000)

    # make log directory
    logdir = f"logs/{learning_method}"
    dir_index = 1
    while os.path.exists(f"{logdir}/{dir_index}") and len(
        os.listdir(f"{logdir}/{dir_index}")
    ):
        dir_index += 1
    logdir = f"{logdir}/{dir_index}"
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    print("Start training")

    if learning_method not in ["PPO", "DQN", "Masked-PPO"]:
        # load expert trajectory
        with open("expert_trajectory.pkl", "rb") as fp:
            rollouts = pickle.load(fp)

    models_dir = f"models/{learning_method}/{dir_index}"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq, save_path=f"{models_dir}/", name_prefix=learning_method
    )

    if learning_method == "BC":
        # basic behavior cloning
        bc_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=rollouts,
            custom_logger=imit_logger.configure(folder=logdir),
            rng=SEED,
        )  # using default policy for better performance

        bc_trainer.train(n_epochs=timesteps)
        bc_trainer.save_policy(f"{models_dir}/BC_{timesteps}_steps.zip")

    elif learning_method == "DQN":
        if record_trajectories:
            rec_dir = f"{logdir}/solutions"
            if not os.path.exists(rec_dir):
                os.makedirs(rec_dir)
            callback = CallbackList(
                [
                    checkpoint_callback,
                    Logger.RecordTrajectories(output_dir=rec_dir),
                ]
            )
        else:
            callback = checkpoint_callback

        model = DQN(
            PolycraftDQNPolicy,  # "MlpPolicy"
            env,
            verbose=1,
            learning_rate=3e-4,
            learning_starts=2048,
            exploration_fraction=1,
            train_freq=(1, "episode"),
            target_update_interval=epoch,
            batch_size=batch_size,
            tensorboard_log=logdir,
            seed=SEED,
        )

        model.learn(
            total_timesteps=timesteps,
            callback=callback,
        )

    else:  # PPO or GAIL
        # agent
        if learning_method == "Masked-PPO":
            model = MaskablePPO(
                PolycraftMaskedPPOPolicy,
                env,
                verbose=1,
                n_steps=epoch,
                batch_size=batch_size,
                tensorboard_log=logdir,
                seed=SEED,
            )
        else:
            model = PPO(
                PolycraftPPOPolicy,
                env,
                verbose=1,
                n_steps=epoch,
                batch_size=batch_size,
                tensorboard_log=logdir,
                seed=SEED,
            )

        if learning_method == "GAIL":
            # behavior cloning using GAIL
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
                demonstrations=rollouts,
                demo_batch_size=1,
                gen_replay_buffer_capacity=buffer_size,
                venv=venv,
                gen_algo=model,
                reward_net=reward_net,
                log_dir=logdir,
                init_tensorboard=True,
                init_tensorboard_graph=True,
                allow_variable_horizon=True,
                custom_logger=imit_logger.configure(folder=logdir),
            )

            checkpoint_callback.model = model
            checkpoint_callback.save_freq = int(save_freq / epoch)

            gail_trainer.train(
                total_timesteps=timesteps,
                callback=lambda step: checkpoint_callback.on_step(),
            )
        else:
            if record_trajectories:
                rec_dir = f"{logdir}/solutions"
                if not os.path.exists(rec_dir):
                    os.makedirs(rec_dir)
                callback = CallbackList(
                    [
                        checkpoint_callback,
                        Logger.RecordTrajectories(output_dir=rec_dir),
                    ]
                )
            else:
                callback = checkpoint_callback

            # RL using PPO
            model.learn(
                total_timesteps=timesteps,
                callback=callback,
            )

    print("Done training")


def train_with_qlearning(
    env,
    learning_method: str,
    timesteps: int = 1024,
    record_trajectories: bool = False,
    save_freq: int = 1024,
):
    """Train Q-Learning agent

    Args:
        learning_method (str): one of "offline", "online"
        timesteps (int, optional): number of timesteps to train. Defaults to 1024.
        record_trajectories (bool, optional): whether to record trajectories for planning. Defaults to False.
        save_freq (int, optional): frequency to save model. Defaults to 1024.
    """

    if learning_method not in ["offline", "online"]:
        raise ValueError("learning method must be offline or online")

    env = RecordEpisodeStatistics(env, deque_size=5000)

    # make log directory
    logdir = f"logs/qlearning_{learning_method}"
    dir_index = 1
    while os.path.exists(f"{logdir}/{dir_index}") and len(
        os.listdir(f"{logdir}/{dir_index}")
    ):
        dir_index += 1
    logdir = f"{logdir}/{dir_index}"
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    print("Start training")

    models_dir = f"models/qlearning_{learning_method}/{dir_index}"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq, save_path=f"{models_dir}/", name_prefix="Q-Learning"
    )

    if record_trajectories:
        rec_dir = f"{logdir}/solutions"
        if not os.path.exists(rec_dir):
            os.makedirs(rec_dir)
        callback = CallbackList(
            [
                checkpoint_callback,
                Logger.RecordTrajectories(output_dir=rec_dir),
            ]
        )
    else:
        callback = checkpoint_callback

    if learning_method == "online":
        agent = QLearningAgent(env)
        agent.learn(timesteps, callback)
    else:  # offline
        agent = QLearningAgent(
            env,
            learning_rate=1,
            initial_epsilon=0,
            final_epsilon=0,
        )

        if type(env.env) is BasicMinecraft:
            filename = "agents/scripts/macro_actions_script.txt"
        elif type(env.env) is IntermediateMinecraft:
            filename = "agents/scripts/intermediate_actions_script.txt"
        else:  # AdvancedMinecraft
            filename = "agents/scripts/advanced_actions_script.txt"

        expert = FixedScriptAgent(env.env, filename=filename, human_readable=True)

        agent.learn(timesteps, callback, expert)

    print("Done training")


def evaluate(env, model):
    """Evaluate the trained model"""
    avg = []
    for domain_path in CONFIG.EVALUATION_DOMAINS_PATH:
        env.set_domain(domain_path)
        rewards, _ = evaluate_policy(
            model=model,
            env=env,
            n_eval_episodes=1,
            return_episode_rewards=True,
            deterministic=True,
        )
        avg.append(sum(rewards) / len(rewards))

    print("Average Reward:", avg)
    print("Total Average Reward:", sum(avg) / len(avg))


def main():
    env_index = 0  # 0: BasicMinecraft, 1: IntermediateMinecraft, 2: AdvancedMinecraft, 3: MaskedMinecraft
    minecraft = [
        BasicMinecraft,
        IntermediateMinecraft,
        AdvancedMinecraft,
        MaskedMinecraft,
    ][env_index]

    # only start pal
    # env = minecraft(visually=True, start_pal=True, keep_alive=True)
    # env.reset()
    # env.close()
    # return

    env = minecraft(visually=True, start_pal=True, keep_alive=False)

    learning_index = (
        0  # 0: Planning, 1: BC, 2: Q-Learning, 3: DQN, 4: PPO, 5: GAIL, 6: Masked-PPO
    )
    learning_method = [
        "Planning",
        "BC",
        "Q-Learning",
        "DQN",
        "PPO",
        "GAIL",
        "Masked-PPO",
    ][learning_index]
    timesteps: int = 1024
    save_freq = env.max_steps * 8

    # train the model
    if learning_method == "Q-Learning":
        train_with_qlearning(
            env, "online", timesteps, record_trajectories=True, save_freq=save_freq
        )
    else:
        train_rl_agent(
            env,
            learning_method,
            timesteps,
            record_trajectories=True,
            save_freq=save_freq,
        )

    # load and evaluate the model
    # if learning_method == "BC":
    #     model = bc.reconstruct_policy("models/BC/1/BC_1024_steps.zip")
    # elif learning_method == "Q-Learning":
    #     model = QLearningAgent.load(
    #         "models/qlearning_online/1/Q-Learning_1024_steps.zip", env=env
    #     )
    # elif learning_method == "DQN":
    #     model = DQN.load(f"models/{learning_method}/1/DQN_1024_steps.zip", env=env)
    # elif learning_method == "PPO":
    #     model = PPO.load(f"models/{learning_method}/1/PPO_1024_steps.zip", env=env)
    # elif learning_method == "GAIL":
    #     model = PPO.load(f"models/{learning_method}/1/GAIL_1024_steps.zip", env=env)
    # elif learning_method == "Masked-PPO":
    #     model = MaskablePPO.load(
    #         f"models/{learning_method}/1/Masked-PPO_1024_steps.zip", env=env
    #     )
    # else:  # Planning
    #     if env_index == 0:
    #         domain = f"{os.getcwd()}/planning/basic_minecraft_domain.pddl"
    #         problem = f"{os.getcwd()}/planning/basic_minecraft_problem.pddl"
    #     elif env_index == 1:
    #         domain = f"{os.getcwd()}/planning/intermediate_minecraft_domain.pddl"
    #         problem = f"{os.getcwd()}/planning/intermediate_minecraft_problem.pddl"
    #     else:
    #         domain = f"{os.getcwd()}/planning/advanced_minecraft_domain.pddl"
    #         problem = f"{os.getcwd()}/planning/advanced_minecraft_problem.pddl"

    #     planner = solver()
    #     plan = planner.create_plan(domain, problem)
    #     # print(plan)
    #     model = FixedScriptAgent(env, script=plan)

    # evaluate(env, model)

    env.close()


if __name__ == "__main__":
    # profiling
    # cProfile.run("main()", filename="cProfile.prof")
    # file = open("cProfile.txt", "w")
    # profile = pstats.Stats("cProfile.prof", stream=file)
    # profile.sort_stats("tottime")
    # profile.print_stats(50)
    # file.close()

    # time to run
    # avg = 0
    # repeats = 3
    # for i in range(repeats):
    #     start_time = time.time()
    #     main()
    #     end_time = time.time()
    #     avg += (end_time - start_time) / repeats
    # print(f"Average time to run: {int(end_time - start_time)}s")

    # standard run
    main()
