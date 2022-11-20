import json
import socket
from typing import Dict


class ServerController:
    """Class to control the Polycraft server."""

    def __init__(self):
        # Create a socket connection to the Polycraft server.
        self.sock = socket.socket()
        self.host = "127.0.0.1"
        self.port = 9000

    def open_connection(self, host: str = "127.0.0.1", port: int = 9000) -> None:
        """Open the connection to the Polycraft server."""
        self.host = host
        self.port = port
        self.sock.connect((host, port))

    def send_command(self, command: str) -> Dict:
        """Send a command to the Polycraft server and return the response."""

        if "\n" not in command:
            command += "\n"
        try:
            self.sock.send(str.encode(command))
        except BrokenPipeError:
            raise ConnectionError("Not connected to the Polycraft server.")
        # print(command)
        BUFF_SIZE = 4096  # 4 KiB
        data = b""
        while True:  # read the response
            part = self.sock.recv(BUFF_SIZE)
            data += part
            if len(part) < BUFF_SIZE:
                # either 0 or end of data
                break
        if not data:  # RESET command returns no data
            data_dict = {}
        else:
            data_dict = json.loads(data)
        # print(data_dict)
        return data_dict

    def close_connection(self) -> None:
        """Close the connection to the Polycraft server."""
        self.sock.close()
