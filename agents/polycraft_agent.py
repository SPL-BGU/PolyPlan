from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    ConvertCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv
from abc import ABC, abstractmethod
from tqdm import tqdm
import pickle


class PolycraftAgent(ABC):
    """Abstract base class for all Polycraft agents."""

    def __init__(self, env):
        # Create a server controller.
        self.env = env
        self.episodes = 0
        self.num_timesteps = 0
        self.logger = None

    @abstractmethod
    def choose_action(self, state) -> int:
        """Choose an action based on the state."""
        raise NotImplementedError

    def act(self) -> int:
        """Choose an action and send it to the Polycraft server."""
        state = self.env.state
        action = self.do(state)
        return action

    def do(self, state) -> str:
        """Choose an action and send it to the Polycraft server."""
        action = self.choose_action(state)
        self.env.step(action)
        return action

    def predict(self, observations, state, episode_start, deterministic) -> tuple:
        """Wrapper for gym predict function."""
        return [self.choose_action(observations[0])], None

    # init callback from stable-baselines3
    def _init_callback(
        self,
        callback: MaybeCallback,
    ) -> BaseCallback:
        """
        :param callback: Callback(s) called at every step with state of the algorithm.
        :return: A hybrid callback calling `callback` and performing evaluation.
        """
        # Convert a list of callbacks into a callback
        if isinstance(callback, list):
            callback = CallbackList(callback)

        # Convert functional callback to object
        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        callback.init_callback(self)
        return callback

    def learn(self, total_timesteps: int, callback: MaybeCallback = None):
        """Learn the policy for a total of `total_timesteps` steps."""

        env = DummyVecEnv([lambda: self.env])  # callback use

        # start callback
        if callback is not None:
            callback = self._init_callback(callback)
            callback.on_training_start(locals(), globals())

        self.num_timesteps = 0
        pbar = tqdm(total=total_timesteps)

        # play n_episodes episodes
        while self.num_timesteps < total_timesteps:
            state = self.env.reset()
            done = False

            # play one episode
            while not done:
                action = self.choose_action(state)
                actions = [action]  # callback use

                # update if the environment is done and the current obs
                state, _, done, _ = self.env.step(action)

                # update callback
                if callback is not None:
                    callback.update_locals(locals())
                    if callback.on_step() is False:
                        break

                self.num_timesteps += 1

            self.episodes += 1

            # rollout end callback
            if callback is not None:
                callback.on_rollout_end()

            # Update the progress bar
            pbar.n = self.num_timesteps
            pbar.set_postfix({"Episodes": self.episodes})

        # end callback
        if callback is not None:
            callback.on_training_end()

    def save(self, path):
        with open(path, "wb") as f:
            env, self.env = self.env, None
            pickle.dump(self, f)
            self.env = env

    @staticmethod
    def load(path, env):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        obj.env = env
        return obj

    def get_env(self):
        return self.env
