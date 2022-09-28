import socket, os
import utils

print("INITIALIZING")

AGENT_HOST = "127.0.0.1"
AGENT_PORT = 9000

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
if "PAL_AGENT_PORT" in os.environ:
    sock.connect((AGENT_HOST, int(os.environ["PAL_AGENT_PORT"])))
    print("Using Port: " + os.environ["PAL_AGENT_PORT"])
else:
    sock.connect((AGENT_HOST, AGENT_PORT))
    print("Using Port: " + str(AGENT_PORT))

command_stream = [
    "TELEPORT 27 4 27 90 0",
    "SENSE_ALL NONAV",
    "TP_TO 18,4,26",
    "BREAK_BLOCK",
    "MOVE w",
    "CRAFT 1 minecraft:log 0 0 0",
    "TP_TO 3,4,17",
    "BREAK_BLOCK",
    "MOVE w",
    "CRAFT 1 minecraft:log 0 0 0",
    "TP_TO 10,4,10",
    "BREAK_BLOCK",
    "MOVE w",
    "CRAFT 1 minecraft:log 0 0 0",
    "TP_TO 8,4,2",
    "BREAK_BLOCK",
    "MOVE w",
    "CRAFT 1 minecraft:log 0 0 0",
    "CRAFT 1 minecraft:planks 0 minecraft:planks 0",
    "CRAFT 1 minecraft:planks 0 minecraft:planks 0",
    "TP_TO 20,4,20",
    "CRAFT 1 minecraft:planks minecraft:stick minecraft:planks minecraft:planks 0 minecraft:planks 0 minecraft:planks 0",
    "TP_TO 23,4,15 2",
    "PLACE_TREE_TAP",
    "COLLECT",
    "TP_TO 20,4,20",
    "CRAFT 1 minecraft:stick minecraft:stick minecraft:stick minecraft:planks minecraft:stick minecraft:planks 0 polycraft:sack_polyisoprene_pellets 0",
    "SENSE_ALL NONAV",
]

for command in command_stream:
    print(command)
    data_dict = utils.send_command(sock, command)
    print(data_dict)

sock.close()

print("Socket closed")
