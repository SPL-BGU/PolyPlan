import os
import pickle
from polycraft_gym_env import PolycraftGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import cProfile, pstats
from stable_baselines3.common.evaluation import evaluate_policy

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
    train_time = 1  # time in minutes

    if training:

        print("Start training")

        # load expert trajectory
        with open("expert_trajectory.pkl", "rb") as fp:
            rollouts = pickle.load(fp)

        models_dir = f"models/{learning_method}"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        if learning_method == "BC":
            # basic behavior cloning
            bc_trainer = bc.BC(
                observation_space=env.observation_space,
                action_space=env.action_space,
                demonstrations=rollouts,
            )

            TIMESTEPS = 4320 * train_time  # 4320 takes 1 minute
            bc_trainer.train(n_epochs=TIMESTEPS)
            bc_trainer.save_policy(f"{models_dir}/{TIMESTEPS}.h5f")
        else:
            logdir = "logs"
            if not os.path.exists(logdir):
                os.makedirs(logdir)

            # agent
            batch_size = 32
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                n_steps=batch_size * 2,
                batch_size=batch_size,
                tensorboard_log=logdir,
            )

            if learning_method == "GAIL":
                # behavior cloning using GAIL
                venv = DummyVecEnv([lambda: RolloutInfoWrapper(env)])
                reward_net = BasicRewardNet(
                    env.observation_space,
                    env.action_space,
                    normalize_input_layer=RunningNorm,
                )
                gail_trainer = GAIL(
                    demonstrations=rollouts,
                    demo_batch_size=batch_size,
                    gen_replay_buffer_capacity=batch_size * 2,
                    n_disc_updates_per_round=4,
                    venv=venv,
                    gen_algo=model,
                    reward_net=reward_net,
                )

                TIMESTEPS = 384 * train_time  # 384 actions takes 1 minute
                gail_trainer.train(TIMESTEPS)
                model.save(f"{models_dir}/{TIMESTEPS}.h5f")
            else:
                # RL using PPO
                epochs = 3 * train_time
                TIMESTEPS = batch_size * 4
                # actions = epochs * TIMESTEPS, 1000 actions takes 2.5 mins
                # example: 3 * (32*4) ~= 384 actions = 1 min
                for i in range(1, epochs + 1):  # save the model every epoch
                    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
                    model.save(f"{models_dir}/{i*TIMESTEPS}.h5f")

        print("Done training")
    else:
        # load and evaluate the model
        if learning_method == "BC":
            model = bc.reconstruct_policy("models/BC/4320.h5f")
        else:
            model = PPO.load(f"models/{learning_method}/384.h5f", env=env)

        rewards, _ = evaluate_policy(
            model=model,
            env=env,
            n_eval_episodes=1,
            return_episode_rewards=True,
            deterministic=True,
        )
        print("Rewards:", rewards)

    env.close()


if __name__ == "__main__":
    # cProfile.run("main()", filename="cProfile.prof")
    # file = open("cProfile.txt", "w")
    # profile = pstats.Stats("cProfile.prof", stream=file)
    # profile.sort_stats("tottime")
    # profile.print_stats(50)
    # file.close()

    main()
