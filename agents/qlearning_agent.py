from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from agents import PolycraftAgent
import numpy as np
import pandas as pd
from tqdm import tqdm


class QLearningAgent(PolycraftAgent):
    """Q-Learning Agent."""

    def __init__(
        self,
        env,
        learning_rate: float = 0.1,
        initial_epsilon: float = 1,
        epsilon_decay: float = 0.02,
        final_epsilon: float = 0.05,
        discount_factor: float = 0.99,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_table), a learning rate and an epsilon.

        Args:
            env: The environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        super().__init__(env)
        self.action_space = self.env.action_space
        self.q_table = pd.DataFrame(
            columns=range(self.action_space.n), dtype=np.float64
        )
        self.eval = False

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    # overriding abstract method
    def choose_action(self, state=None) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """

        state = str(state)
        self.check_state_exist(state)

        if self.eval:
            return int(np.argmax(self.q_table.loc[state]))

        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_table.loc[state]))

    def update(self, trajectory):
        """Updates the Q-table by playing one episode."""

        self.check_state_exist(trajectory[-1][-1])
        trajectory.reverse()
        for state, action, reward, terminated, next_state in trajectory:
            future_q_value = (not terminated) * np.max(self.q_table.loc[next_state])
            temporal_difference = (
                reward
                + self.discount_factor * future_q_value
                - self.q_table.loc[state, action]
            )

            self.q_table.loc[state, action] = (
                self.q_table.loc[state, action] + self.lr * temporal_difference
            )
            self.training_error.append(temporal_difference)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table.loc[state] = [0] * self.action_space.n

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        expert: PolycraftAgent = None,
    ):
        """Learns the Q-table by playing n_episodes episodes.
        if expert is not None do offline learning from expert trajectories"""

        env = DummyVecEnv([lambda: self.env])  # callback use

        # start callback
        if callback is not None:
            callback = self._init_callback(callback)
            callback.on_training_start(locals(), globals())

        self.num_timesteps = 1
        pbar = tqdm(total=total_timesteps)

        # play n_episodes episodes
        while self.num_timesteps <= total_timesteps:
            state = self.env.reset()
            done = False

            # callback on training start
            if callback is not None:
                callback.on_rollout_start()

            trajectory = []

            # play one episode
            while not done:
                if expert:
                    self.check_state_exist(str(state))
                    action = expert.choose_action(state)
                else:
                    action = self.choose_action(state)
                actions = [action]  # callback use

                next_state, reward, done, _ = self.env.step(action)

                # save trajectory
                trajectory.append((str(state), action, reward, done, str(next_state)))

                # update if the environment is done and the current obs
                state = next_state

                # update callback
                if callback is not None:
                    callback.update_locals(locals())
                    if callback.on_step() is False:
                        break

                self.num_timesteps += 1

            # update the agent
            self.update(trajectory)
            self.decay_epsilon()

            self.episodes += 1

            # rollout end callback
            if callback is not None:
                callback.on_rollout_end()

            # Update the progress bar
            pbar.n = self.num_timesteps
            pbar.set_postfix({"Episodes": self.episodes})

    def save_table(self, path):
        """Export the Q-table to a csv file."""
        self.q_table.to_csv(f"{path}/qtable_{self.num_timesteps}_episodes.csv")

    def load_table(self, path):
        """Load the Q-table from a csv file."""
        self.q_table = pd.read_csv(path, index_col=0).rename(columns=lambda x: int(x))
