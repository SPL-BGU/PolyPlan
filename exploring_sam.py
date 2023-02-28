import os

from envs import BasicMinecraft
from polycraft_policy import PolycraftPolicy

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)


class RecordTrajectories(BaseCallback):
    """
    Record trajectories for planning.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, verbose=0, output_dir="solutions"):
        super().__init__(verbose)
        self.episode = 0
        self.output_dir = output_dir
        self.file = open(f"{output_dir}/pfile0.solution", "a")

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        env = self.locals["env"].envs[0]

        action = self.locals["actions"][0]
        action = env.decoder.decode_to_planning(action)
        self.file.write(f"({action})\n")

        if env.rounds_left == 64:
            self.file.close()
            self.episode += 1
            self.file = open(f"{self.output_dir}/pfile{self.episode}.solution", "a")

        return True


def main():
    # only start pal
    # env = BasicMinecraft(visually=True, start_pal=True, keep_alive=True)
    # env.reset()
    # env.close()
    # return

    # start the polycraft environment
    env = BasicMinecraft(visually=True, start_pal=True, keep_alive=False)

    timesteps = 1024
    epoch = 256
    batch_size = 64

    # make log directory
    logdir = f"logs/E-SAM"
    dir_index = 1
    while os.path.exists(f"{logdir}/{dir_index}") and len(
        os.listdir(f"{logdir}/{dir_index}")
    ):
        dir_index += 1
    logdir = f"{logdir}/{dir_index}"
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    print("Start training")

    models_dir = f"models/E-SAM/{dir_index}"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # agent
    model = PPO(
        PolycraftPolicy,
        env,
        verbose=1,
        n_steps=epoch,
        batch_size=batch_size,
        tensorboard_log=logdir,
    )

    # checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=1024, save_path=f"{models_dir}/", name_prefix="E-SAM"
    )

    callback = CallbackList([checkpoint_callback, RecordTrajectories()])

    # RL using PPO
    model.learn(
        total_timesteps=timesteps,
        callback=callback,
    )

    print("Done training")

    env.close()  # close the environment


if __name__ == "__main__":
    main()
