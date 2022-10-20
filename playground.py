from polycraft_gym_env import PolycraftGymEnv
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
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

        epochs = 4
        TIMESTEPS = batch_size * 4
        # actions = epochs * TIMESTEPS, 1000 actions takes 2.5 mins
        for i in range(1, epochs + 1):  # save the model every epoch
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
            model.save(f"{models_dir}/{i*TIMESTEPS}.h5f")
        print("Done training")
    else:
        # loading the model
        model = PPO.load("models/PPO/512.h5f", env=env)

        state = env.reset()
        done = False
        # while not done:
        for _ in range(100):
            action, _state = model.predict(state, deterministic=False)
            state, reward, done, info = env.step(action)
            env.render()

    env.close(end_pal=True)


if __name__ == "__main__":
    main()
