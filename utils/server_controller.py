import json
import time
import socket
from typing import Dict


class ServerController:
    """Class to control the Polycraft server."""

    def __init__(self):
        # Create a socket connection to the Polycraft server.
        self.sock = socket.socket()
        self.host = "127.0.0.1"
        self.port = 9000

    def set_timeout(self, timeout: float = 1.0) -> None:
        """Set the timeout for the socket connection."""
        self.sock.settimeout(timeout)

    def open_connection(self, host: str = "127.0.0.1", port: int = 9000) -> None:
        """Open the connection to the Polycraft server."""
        self.host = host
        self.port = port
        self.sock.connect((host, port))
        self.set_timeout()

    def send_command(self, command: str) -> Dict:
        """Send a command to the Polycraft server and return the response."""
        sleep_time = 0.025
        attempt_result = 0
        while attempt_result != 10:
            attempt_result += 1
            if "\n" not in command:
                command += "\n"
            try:
                self.sock.send(str.encode(command))
                time.sleep(sleep_time)
                # sleep_time = min(1, sleep_time * 1.5)
                sleep_time = sleep_time * 1.5
            except BrokenPipeError:
                raise ConnectionError("Not connected to the Polycraft server.")
            # print(command)
            BUFF_SIZE = 4096  # 4 KiB
            data = b""
            while True:  # read the response
                try:
                    part = self.sock.recv(BUFF_SIZE)
                    data += part
                    if len(part) < BUFF_SIZE:
                        # either 0 or end of data
                        break
                except socket.timeout:
                    break
            if data:
                try:
                    data_dict = json.loads(data)
                    # if data_dict["command_result"]["result"] == "SUCCESS":
                    #     break
                    break
                except:
                    pass
        # print(data_dict)
        return data_dict

    def close_connection(self) -> None:
        """Close the connection to the Polycraft server."""
        self.sock.close()


if __name__ == "__main__":
    # Test the server controller.
    server_controller = ServerController()
    server_controller.open_connection()
    while user_input := input("Enter a command: "):
        print(server_controller.send_command(user_input))
    server_controller.close_connection()
