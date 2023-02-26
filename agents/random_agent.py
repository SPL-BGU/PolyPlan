from agents.polycraft_agent import PolycraftAgent


class RandomAgent(PolycraftAgent):
    """Agent that act randomly."""

    def __init__(self, env):
        super().__init__(env)

    # overriding abstract method
    def choose_action(self, state=None) -> str:
        """Ignore the state and return random action"""
        action = self.env.action_space.sample()
        return action
