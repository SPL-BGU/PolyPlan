import json


def send_command(sock, command):  # add time=0 ? and update both of the demo agents
    """Send a command to the Polycraft server and return the response."""

    if "\n" not in command:
        command += "\n"
    try:
        sock.send(str.encode(command))
    except BrokenPipeError:
        raise ConnectionError("Not connected to the Polycraft server.")
    # print(command)
    BUFF_SIZE = 4096  # 4 KiB
    data = b""
    while True:  # read the response
        part = sock.recv(BUFF_SIZE)
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


class Decoder:
    _actions_decoder = {
        0: "NOP",
        1: "MOVE W",
        2: "MOVE A",
        3: "MOVE D",
        4: "MOVE X",
        5: "TURN 90",
        6: "TURN -90",
        7: "BREAK_BLOCK",
        8: "COLLECT",
        9: "PLACE_TREE_TAP",
        10: "CRAFT 1 minecraft:log 0 0 0",
        11: "CRAFT 1 minecraft:planks 0 minecraft:planks 0",
        12: "CRAFT 1 minecraft:planks minecraft:stick minecraft:planks minecraft:planks 0 minecraft:planks 0 minecraft:planks 0",
        13: "CRAFT 1 minecraft:stick minecraft:stick minecraft:stick minecraft:planks minecraft:stick minecraft:planks 0 polycraft:sack_polyisoprene_pellets 0",
    }
    _actions_size = len(_actions_decoder)

    _blocks_decoder = {
        "minecraft:air": 0,
        "minecraft:bedrock": 1,
        "polycraft:block_of_platinum": 2,
        "minecraft:crafting_table": 3,
        "minecraft:diamond_ore": 4,
        "minecraft:log": 5,
        "polycraft:plastic_chest": 6,
        "minecraft:sapling": 7,
        "polycraft:tree_tap": 8,
        "minecraft:wooden_door": 9,
        "polycraft:safe": 10,
    }
    _blocks_size = len(_blocks_decoder)

    _items_decoder = {
        "polycraft:block_of_platinum": 0,
        "polycraft:block_of_titanium": 1,
        "minecraft:crafting_table": 2,
        "minecraft:diamond": 3,
        "minecraft:diamond_block": 4,
        "minecraft:iron_pickaxe": 5,
        "polycraft:key": 6,
        "minecraft:log": 7,
        "minecraft:planks": 8,
        "minecraft:sapling": 9,
        "minecraft:stick": 10,
        "polycraft:sack_polyisoprene_pellets": 11,
        "polycraft:tree_tap": 12,
        "polycraft:wooden_pogo_stick": 13,
    }
    _items_size = len(_items_decoder)

    _entitys_decoder = {
        "EntityPogoist": 0,
        "EntityTrader": 1,
        "EntityItem": 2,
    }
    _entitys_size = len(_entitys_decoder)

    @staticmethod
    def decode_action_type(num):
        return Decoder._actions_decoder[num]

    @staticmethod
    def get_actions_size():
        return Decoder._actions_size

    @staticmethod
    def decode_block_type(name):
        return Decoder._blocks_decoder[name]

    @staticmethod
    def get_blocks_size():
        return Decoder._blocks_size

    @staticmethod
    def decode_item_type(name):
        return Decoder._items_decoder[name]

    @staticmethod
    def get_items_size():
        return Decoder._items_size

    @staticmethod
    def decode_entity_type(name):
        return Decoder._entitys_decoder[name]

    @staticmethod
    def get_entitys_size():
        return Decoder._entitys_size
