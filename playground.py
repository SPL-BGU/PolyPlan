import os
import pickle
import shutil

import time
import cProfile, pstats
import config as CONFIG
from stable_baselines3.common.evaluation import evaluate_policy

import numpy as np
import pandas as pd

from envs import BasicMinecraft
from polycraft_policy import PolycraftPPOPolicy, PolycraftDQNPolicy

from planning.enhsp import ENHSP
from agents import FixedScriptAgent

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from imitation.algorithms import bc
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util import logger as imit_logger

from gym.wrappers import RecordEpisodeStatistics


def save_log(env: RecordEpisodeStatistics, filename: str, window_size: int = 256):
    score = np.array(env.return_queue)
    length = np.array(env.length_queue)
    array_score = []
    array_length = []

    i = 0
    while i < len(score):
        j = i + 1
        total = length[i]
        while j < len(score) and (total := total + length[j]) < window_size:
            j += 1
        avg_score = np.average(score[i:j])
        avg_length = np.average(length[i:j])
        array_score.append(avg_score)
        array_length.append(avg_length)
        i = j

    results = np.array([array_score, array_length]).transpose()
    df = pd.DataFrame(results, columns=["score", "length"])
    df.to_csv(filename)


def train_rl_agent(
    env,
    learning_method: str,
    timesteps: int = 1024,
    epoch: int = 256,
    batch_size: int = 64,
):
    """Train RL agent using PPO, BC, or GAIL

    Args:
        learning_method (str): one of "BC", "PPO", "GAIL"
        timesteps (int, optional): number of timesteps to train. Defaults to 1024.
        epoch (int, optional): how much steps to until update the net. Defaults to 256.
        batch_size (int, optional): size of the sub-updated in each epoch. Defaults to 64.
    """

    if learning_method not in ["BC", "DQN", "PPO", "GAIL"]:
        raise ValueError("learning method must be one of BC, DQN, PPO, GAIL")

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

    if learning_method not in ["PPO", "DQN"]:
        # load expert trajectory
        with open("expert_trajectory.pkl", "rb") as fp:
            rollouts = pickle.load(fp)

    models_dir = f"models/{learning_method}/{dir_index}"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    save_freq = 1024

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
        )  # using default policy for better performance

        bc_trainer.train(n_epochs=timesteps)
        bc_trainer.save_policy(f"{models_dir}/BC_{timesteps}_steps.zip")

    elif learning_method == "DQN":
        model = DQN(
            PolycraftDQNPolicy,  # "MlpPolicy"
            env,
            verbose=1,
            learning_rate=3e-4,
            learning_starts=5000,
            train_freq=(1, "episode"),
            target_update_interval=2000,
            batch_size=batch_size,
            tensorboard_log=logdir,
        )

        model.learn(
            total_timesteps=timesteps,
            callback=checkpoint_callback,
        )

    else:  # PPO or GAIL
        # agent
        model = PPO(
            PolycraftPPOPolicy,
            env,
            verbose=1,
            n_steps=epoch,
            batch_size=batch_size,
            tensorboard_log=logdir,
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
                demo_batch_size=11,
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
            # RL using PPO
            model.learn(
                total_timesteps=timesteps,
                callback=checkpoint_callback,
            )

    save_log(env, f"{logdir}/output.csv", epoch)

    print("Done training")


def evaluate(env, model):
    """Evaluate the trained model"""
    avg = []
    for domain_path in CONFIG.EVALUATION_DOMAINS_PATH:
        env.set_domain(domain_path)
        rewards, _ = evaluate_policy(
            model=model,
            env=env,
            n_eval_episodes=5,
            return_episode_rewards=True,
            deterministic=False,
        )
        avg.append(sum(rewards) / len(rewards))

    print("Average Reward:", avg)
    print("Total Average Reward:", sum(avg) / len(avg))


def main():
    # only start pal
    # env = BasicMinecraft(visually=True, start_pal=True, keep_alive=True)
    # env.reset()
    # env.close()
    # return

    env = BasicMinecraft(visually=True, start_pal=True, keep_alive=False)
    learning_method = ["BC", "DQN", "PPO", "GAIL"][0]
    timesteps: int = 1024

    train_rl_agent(env, learning_method, timesteps)

    # load and evaluate the model
    # if learning_method == "BC":
    #     model = bc.reconstruct_policy("models/BC/1/BC_1024_steps.zip")
    # elif learning_method == "DQN":
    #     model = DQN.load(f"models/{learning_method}/1/DQN_102400_steps.zip", env=env)
    # else:  # PPO or GAIL
    #     model = PPO.load(f"models/{learning_method}/1/PPO_1024_steps.zip", env=env)

    # enhsp = ENHSP()
    # plan = enhsp.create_plan()
    # # print(plan)
    # model = FixedScriptAgent(env, script=plan)

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
