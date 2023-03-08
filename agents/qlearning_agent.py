from agents.polycraft_agent import PolycraftAgent
import numpy as np
import pandas as pd
from tqdm import tqdm


class QLearningAgent(PolycraftAgent):
    """Agent that act randomly."""

    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float = 1,
        epsilon_decay: float = 0.001,
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

        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_table.loc[state]))

    def update(
        self,
        state,
        action: int,
        reward: float,
        terminated: bool,
        next_state,
    ):
        """Updates the Q-value of an action."""

        state = str(state)
        next_state = str(next_state)
        self.check_state_exist(next_state)

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

    def learn(self, n_episodes):
        for _ in tqdm(range(n_episodes)):
            state = self.env.reset()
            done = False

            # play one episode
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)

                # update the agent
                self.update(state, action, reward, done, next_state)

                # update if the environment is done and the current obs
                state = next_state

            self.decay_epsilon()
    
    def save(self, path):
        self.q_table.to_csv(path)
