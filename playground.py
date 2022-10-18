from polycraft_gym_env import PolycraftGymEnv
from stable_baselines3 import DQN
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

    env = PolycraftGymEnv(visually=True)

    # training the model
    models_dir = "models/DQN"
    logdir = "logs"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    print("Start training")
    model = DQN(
        "MultiInputPolicy",
        env,
        verbose=1,
        target_update_interval=100,
        tensorboard_log=logdir,
    )
    TIMESTEPS = 1000  # 1000 actions takes 2.5 mins
    for i in range(1, 3):
        model.learn(
            total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN"
        )
        model.save(f"{models_dir}/{TIMESTEPS*i}.h5f")
    print("Done training")
    env.close()
    return

    # loading the model
    # model = DQN.load("model.h5f", env=env)

    state = env.reset()
    done = False
    # while not done:
    for _ in range(1000):
        action, _state = model.predict(
            state, deterministic=True
        )  # work only as deterministic
        state, reward, done, info = env.step(action)
        env.render()

    env.close()


if __name__ == "__main__":
    main()
