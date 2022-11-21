import os
import pickle
from polycraft_gym_env import PolycraftGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import cProfile, pstats
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
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

    if training:
        # training the model
        models_dir = "models/PPO"
        logdir = "logs"

        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        print("Start training")

        venv = DummyVecEnv([lambda: RolloutInfoWrapper(env)])

        # load expert trajectory
        with open("expert_trajectory.pkl", "rb") as fp:
            rollouts = pickle.load(fp)

        # behavior cloning
        # bc_trainer = bc.BC(
        #     observation_space=env.observation_space,
        #     action_space=env.action_space,
        #     demonstrations=rollouts,
        # )
        # bc_trainer.train(n_epochs=100)
        # reward, _ = evaluate_policy(bc_trainer.policy, env, 1)
        # env.close()
        # return

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

        # trainer
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

        # pre-train with behavior cloning using GAIL
        gail_trainer.train(384 * 1)

        # post-training with RL
        epochs = 3 * 1
        TIMESTEPS = batch_size * 4
        # actions = epochs * TIMESTEPS, 1000 actions takes 2.5 mins
        # example: 3 * (32*4) ~= 384 actions = 1 min
        for i in range(1, epochs + 1):  # save the model every epoch
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
            model.save(f"{models_dir}/{i*TIMESTEPS}.h5f")
        print("Done training")
    else:
        # load and evaluate the model
        model = PPO.load("models/PPO/640.h5f", env=env)
        rewards, _ = evaluate_policy(
            model=model,
            env=env,
            n_eval_episodes=5,
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
