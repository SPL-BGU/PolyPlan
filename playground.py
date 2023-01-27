import os
import pickle
import shutil

import time
import cProfile, pstats
import config as CONFIG
from stable_baselines3.common.evaluation import evaluate_policy

from envs import BasicMinecraft
from polycraft_policy import PolycraftPolicy

from planning.enhsp import ENHSP
from agents import FixedScriptAgent

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.algorithms import bc
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm


def train_rl_agent(env, learning_method: str, train_time: int):
    """Train RL agent using PPO, BC, or GAIL"""

    if learning_method not in ["BC", "PPO", "GAIL"]:
        raise ValueError("learning method must be one of BC, PPO, GAIL")

    epoch: int = 256  # 64 256 # how much steps to update the net
    batch_size: int = 64  # 32 64 # size of sub-update from the total update size

    if learning_method == "BC":
        timesteps = 8 * epoch  # 2048 actions takes ~ 0.75 minute
    else:
        timesteps = epoch  # 256 actions takes ~ 1 minute

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

    # load expert trajectory
    with open("expert_trajectory.pkl", "rb") as fp:
        rollouts = pickle.load(fp)

    models_dir = f"models/{learning_method}/{dir_index}"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if learning_method == "BC":
        # basic behavior cloning
        bc_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=rollouts,
        )  # using default policy for better performance

        for i in range(1, train_time + 1):
            bc_trainer.train(n_epochs=timesteps)
            bc_trainer.save_policy(f"{models_dir}/{dir_index}_{i}_{timesteps}.h5f")

            # save log
            shutil.copy(
                f"{bc_trainer.logger.get_dir()}/progress.csv",
                f"{logdir}/progress_{i}.csv",
            )
    else:

        # agent
        model = PPO(
            PolycraftPolicy,
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
                demo_batch_size=batch_size,
                gen_replay_buffer_capacity=buffer_size,
                venv=venv,
                gen_algo=model,
                reward_net=reward_net,
                log_dir=logdir,
                init_tensorboard=True,
                init_tensorboard_graph=True,
                allow_variable_horizon=True,
            )

            for i in range(1, train_time + 1):
                gail_trainer.train(timesteps)
                model.save(f"{models_dir}/{dir_index}_{i}_{timesteps}.h5f")
        else:
            # RL using PPO
            for i in range(1, train_time + 1):
                model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
                model.save(f"{models_dir}/{dir_index}_{i}_{timesteps}.h5f")

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
    # # env.reset()
    # env.close()
    # return

    env = BasicMinecraft(visually=True, start_pal=True, keep_alive=False)
    learning_method = ["BC", "PPO", "GAIL"][0]
    train_time: int = 1  # time in minutes

    train_rl_agent(env, learning_method, train_time)

    # load and evaluate the model
    # if learning_method == "BC":
    #     model = bc.reconstruct_policy("models/BC/1/1_1_2048.h5f")
    # else:
    #     model = PPO.load(f"models/{learning_method}/1/1_1_256.h5f", env=env)

    # enhsp = ENHSP()
    # plan = enhsp.create_plan()
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
