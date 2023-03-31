from agents.polycraft_agent import PolycraftAgent
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
        save_path: str = "qlearning/",
        save_interval: int = 100,
        output_dir: str = "solutions",
        record_trajectories: bool = False,
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

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.save_path = save_path
        self.save_interval = save_interval
        self.num_timesteps = 0

        self.episode = 0
        if record_trajectories:
            self.output_dir = output_dir
            self.file = open(f"{output_dir}/pfile0.solution", "a")
        else:
            self.file = None

        self.training_error = []

    # overriding abstract method
    def choose_action(self, state=None) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """

        state = str(state)
        self.check_state_exist(state)

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

    def learn(self, n_episodes: int, expert: PolycraftAgent = None):
        """Learns the Q-table by playing n_episodes episodes.
        if expert is not None do offline learning from expert trajectories"""

        for _ in tqdm(range(n_episodes)):
            state = self.env.reset()
            done = False

            trajectory = []

            # play one episode
            while not done:
                if expert:
                    self.check_state_exist(str(state))
                    action = expert.choose_action(state)
                    act = action
                else:
                    action = self.choose_action(state)
                    act = self.env.decoder.decode_to_planning(action)

                # save action
                if self.file:
                    self.file.write(f"({act})\n")

                next_state, reward, done, _ = self.env.step(action)

                # save trajectory
                trajectory.append((str(state), action, reward, done, str(next_state)))

                # update if the environment is done and the current obs
                state = next_state

            # update the agent
            self.update(trajectory)

            self.episode += 1

            if self.file:
                self.file.close()
                self.file = open(f"{self.output_dir}/pfile{self.episode}.solution", "a")

            # add to logger
            self.num_timesteps += 1
            if self.num_timesteps % self.save_interval == 0:
                self.save()
            self.decay_epsilon()

    def save(self):
        """Export the Q-table to a csv file."""
        self.q_table.to_csv(
            f"{self.save_path}/qtable_{self.num_timesteps}_episodes.csv"
        )

    def load(self, path):
        """Load the Q-table from a csv file."""
        self.q_table = pd.read_csv(path, index_col=0).rename(columns=lambda x: int(x))
