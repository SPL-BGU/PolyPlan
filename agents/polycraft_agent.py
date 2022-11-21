from abc import ABC, abstractmethod
from utils.server_controller import ServerController


class PolycraftAgent(ABC):
    """Abstract base class for all Polycraft agents."""

    def __init__(self, env):
        # Create a server controller.
        self.server_controller = env.server_controller

    @abstractmethod
    def choose_action(self, state):
        """Choose an action based on the state."""
        raise NotImplementedError

    def act(self):
        """Get the state and choose an action."""
        state = self.sense_all()
        action = self.do(state)
        return action

    def sense_all(self):
        """Get the state from the Polycraft server."""
        return self.server_controller.send_command("SENSE_ALL NONAV")

    def do(self, state):
        """Choose an action and send it to the Polycraft server."""
        action = self.choose_action(state)
        self.server_controller.send_command(action)
        return action
