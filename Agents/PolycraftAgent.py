from abc import ABC, abstractmethod
import socket, time, json


class PolycraftAgent(ABC):
    """Abstract base class for all Polycraft agents."""

    def __init__(self, host="127.0.0.1", port=9000):
        # Create a socket connection to the Polycraft server.
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.host = host
        self.port = port
        self.sock.connect((host, port))

    @abstractmethod
    def choose_action(self, state):
        """Choose an action based on the state."""
        pass

    def act(self):
        """Get the state from the Polycraft server and choose an action."""
        state = self.send_command("SENSE_ALL NONAV")
        action = self.do(state)
        return action

    def do(self, state):
        """Choose an action and send it to the Polycraft server."""
        action = self.choose_action(state)
        self.send_command(action)
        return action

    def send_command(self, command):
        """Send a command to the Polycraft server and return the response."""
        self.sock.send(str.encode(command + "\n"))
        # print(command)
        BUFF_SIZE = 4096  # 4 KiB
        data = b""
        while True:
            part = self.sock.recv(BUFF_SIZE)
            data += part
            if len(part) < BUFF_SIZE:
                # either 0 or end of data
                break
        data_dict = json.loads(data.decode("utf-8"))
        # print(data_dict)
        time.sleep(0.5)
        return data_dict

    def close_connection(self):
        """Close the connection to the Polycraft server."""
        self.sock.close()
