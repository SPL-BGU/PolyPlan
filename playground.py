from polycraft_gym_env import PolycraftGymEnv
from stable_baselines3 import DQN, PPO
import cProfile, pstats
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
import os


def main():

    # Check that an environment follows Gym API
    # check_env(PolycraftGymEnv(), skip_render_check=False)
    # return

    # basic
    # env = PolycraftGymEnv(visually=True)
    # file = open("my_playground.txt", "r")
    # domain_path = file.readline()
    # # while domain_path != "done\n":
    # for _ in range(1):
    #     env.set_domain(domain_path)
    #     state = env.reset()
    #     for _ in range(10):
    #         action = env.action_space.sample()
    #         state, reward, done, info = env.step(action)
    #         env.render()
    #     domain_path = file.readline()
    # env.close()
    # return

    env = PolycraftGymEnv(visually=True, start_pal=True)

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
        batch_size = 32
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            # learning_starts=0,
            n_steps=batch_size * 2,
            batch_size=batch_size,
            # target_update_interval=batch_size,
            tensorboard_log=logdir,
        )

        epochs = 6
        TIMESTEPS = batch_size * 4
        # actions = epochs * TIMESTEPS, 1000 actions takes 2.5 mins
        # example: 3 * (32*4) ~= 1000 actions = 1 min
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

    env.close(end_pal=True)


if __name__ == "__main__":
    # cProfile.run("main()", filename="cProfile.prof")
    # file = open("cProfile.txt", "w")
    # profile = pstats.Stats("cProfile.prof", stream=file)
    # profile.sort_stats("tottime")
    # profile.print_stats(50)
    # file.close()

    main()
