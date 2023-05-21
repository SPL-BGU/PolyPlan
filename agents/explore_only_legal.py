from agents.polycraft_agent import PolycraftAgent
import numpy as np
import pandas as pd
from random import choice


class ExploreOnlyLegal(PolycraftAgent):
    """Agent that learn which action is legal and explore the least used action"""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = self.env.action_space.n
        self.graph = pd.DataFrame(columns=range(self.action_space), dtype=np.float64)

        self.action_count = [1] * self.action_space

        # for update useless actions
        self.last_state = None
        self.last_action = None

        # for update useless states
        self.pre_state = None
        self.pre_action = None

    # overriding abstract method
    def choose_action(self, state=None) -> int:
        """Sample action that least used in the state and learn which action is legal"""
        state = str(state)
        self.check_state_exist(state)

        if self.last_state == state:
            self.graph.loc[state][self.last_action] = 0
        else:
            if self.last_action is not None:
                self.action_count[self.last_action] += 1
            self.pre_state = self.last_state
            self.pre_action = self.last_action

        actions_values = [
            i1 / i2 for i1, i2 in zip(self.graph.loc[state], self.action_count)
        ]

        if max(actions_values) > 0:
            avalible_actions = [
                i for i, j in enumerate(actions_values) if j == max(actions_values)
            ]
            action = choice(avalible_actions)
        else:
            self.graph.loc[self.pre_state][self.pre_action] = -1
            action = 0

        self.last_state = state
        self.last_action = action

        return action

    def check_state_exist(self, state):
        if state not in self.graph.index:
            # append new state to the graph
            self.graph.loc[state] = [1] * self.action_space

    def save(self, path):
        self.graph.to_csv(path)

    def load(self, path):
        self.graph = pd.read_csv(path)
