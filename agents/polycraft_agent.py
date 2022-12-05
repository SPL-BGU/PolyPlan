from abc import ABC, abstractmethod


class PolycraftAgent(ABC):
    """Abstract base class for all Polycraft agents."""

    def __init__(self, env):
        # Create a server controller.
        self.env = env

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
