import os
import pickle
import shutil

import time
import cProfile, pstats
from stable_baselines3.common.evaluation import evaluate_policy

from polycraft_gym_env import PolycraftGymEnv
from polycraft_policy import PolycraftPolicy

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.algorithms import bc
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm


def main():

    # only start pal
    # env = PolycraftGymEnv(visually=True, start_pal=True, keep_alive=True)
    # env.close()
    # return

    # basic
    # env = PolycraftGymEnv(visually=True, start_pal=True, keep_alive=True)
    # file = open("my_playground.txt", "r")
    # domain_path = file.readline()
    # # while domain_path != "done\n":
    # for _ in range(1):
    #     env.set_domain(domain_path)
    #     for _ in range(1):
    #         state = env.reset()
    #         done = False
    #         while not done:
    #             action = env.action_space.sample()
    #             state, reward, done, info = env.step(action)
    #             env.render()
    #     domain_path = file.readline()
    # env.close()
    # return

    env = PolycraftGymEnv(visually=True, start_pal=True, keep_alive=False)

    training = True
    learning_method = ["BC", "PPO", "GAIL"][0]
    train_time: int = 1  # time in minutes

    epoch: int = 64
    batch_size: int = 32

    if learning_method == "BC":
        timesteps = 67 * epoch  # 4288 actions takes ~ 1 minute
    else:
        timesteps = 6 * epoch  # 384 actions takes ~ 1 minute

    if training:

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

        models_dir = f"models/{learning_method}"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        if learning_method == "BC":
            # policy
            policy = PolycraftPolicy(
                env.observation_space, env.action_space, lambda _: 3e-4
            )

            # basic behavior cloning
            bc_trainer = bc.BC(
                observation_space=env.observation_space,
                action_space=env.action_space,
                policy=policy,
                demonstrations=rollouts,
            )

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
                    gen_replay_buffer_capacity=epoch,
                    venv=venv,
                    gen_algo=model,
                    reward_net=reward_net,
                    log_dir=logdir,
                    init_tensorboard=True,
                    init_tensorboard_graph=True,
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
    else:
        # load and evaluate the model
        if learning_method == "BC":
            model = bc.reconstruct_policy("models/BC/1_1_4288.h5f")
        else:
            model = PPO.load(f"models/{learning_method}/1_1_384.h5f", env=env)

        rewards, _ = evaluate_policy(
            model=model,
            env=env,
            n_eval_episodes=10,
            return_episode_rewards=True,
            deterministic=False,
        )
        print("Rewards:", rewards)

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
