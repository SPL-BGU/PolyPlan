from abc import ABC, abstractmethod
import socket
import utils


class PolycraftAgent(ABC):
    """Abstract base class for all Polycraft agents."""

    def __init__(self):
        # Create a socket connection to the Polycraft server.
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    @abstractmethod
    def choose_action(self, state):
        """Choose an action based on the state."""
        pass

    def act(self):
        """Get the state and choose an action."""
        state = self.sense_all()
        action = self.do(state)
        return action

    def sense_all(self):
        """Get the state from the Polycraft server."""
        return utils.send_command(self.sock, "SENSE_ALL NONAV")

    def do(self, state):
        """Choose an action and send it to the Polycraft server."""
        action = self.choose_action(state)
        utils.send_command(self.sock, action)
        return action

    def open_connection(self, host="127.0.0.1", port=9000):
        """Open the connection to the Polycraft server."""
        self.host = host
        self.port = port
        self.sock.connect((host, port))

    def close_connection(self):
        """Close the connection to the Polycraft server."""
        self.sock.close()
