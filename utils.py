import time, json


def send_command(sock, command):
    """Send a command to the Polycraft server and return the response."""
    try:
        sock.send(str.encode(command + "\n"))
    except BrokenPipeError:
        raise ConnectionError("Not connected to the Polycraft server.")
    # print(command)
    BUFF_SIZE = 4096  # 4 KiB
    data = b""
    while True:
        part = sock.recv(BUFF_SIZE)
        data += part
        if len(part) < BUFF_SIZE:
            # either 0 or end of data
            break
    data_dict = json.loads(data)
    # print(data_dict)
    time.sleep(0.5)
    return data_dict
