from gym.spaces import MultiDiscrete
import numpy as np
import json
import socket
from typing import Dict, List


def send_command(sock: socket.socket, command: str) -> Dict:
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


class ActionType:

    _actions: Dict[int, str]

    def __init__(self):
        self._actions = None

    @property
    def actions(self) -> Dict[int, str]:
        return self._actions

    @property
    def length(self) -> int:
        return len(self._actions)


class Move(ActionType):
    class_actions = {1: "MOVE W", 2: "MOVE A", 3: "MOVE D", 4: "MOVE X"}

    def __init__(self):
        super().__init__()
        self._actions = Move.class_actions


class Turn(ActionType):
    class_actions = {1: "TURN 90", 2: "TURN -90"}

    def __init__(self):
        super().__init__()
        self._actions = Turn.class_actions


class Break(ActionType):
    class_actions = {1: "BREAK_BLOCK"}

    def __init__(self):
        super().__init__()
        self._actions = Break.class_actions


class TP(ActionType):
    class_actions = {1: "NOP", 2: "NOP", 3: "NOP", 4: "NOP", 5: "NOP", 6: "NOP"}

    def __init__(self):
        super().__init__()
        self._actions = TP.class_actions

    @staticmethod
    def update_actions(sense_all: Dict = None) -> None:
        TP.class_actions = {
            1: "NOP",
            2: "NOP",
            3: "NOP",
            4: "NOP",
            5: "NOP",
            6: "NOP",
        }

        if sense_all is None:
            return

        # find all the blocks that can be TP to
        tp_blocks = []
        for location, block in sense_all["map"].items():
            if (
                block["name"] == "minecraft:log"
                or block["name"] == "minecraft:crafting_table"
            ):
                tp_blocks.append(location)

        # update the actions
        for index in range(1, min(len(TP.class_actions), len(tp_blocks)) + 1):
            if index >= len(tp_blocks):
                break
            TP.class_actions[index] = "TP_TO " + tp_blocks[index - 1]


class Craft(ActionType):
    class_actions = {
        1: "CRAFT 1 minecraft:log 0 0 0",
        2: "CRAFT 1 minecraft:planks 0 minecraft:planks 0",
        3: "CRAFT 1 minecraft:planks minecraft:stick minecraft:planks minecraft:planks 0 minecraft:planks 0 minecraft:planks 0",
        4: "CRAFT 1 minecraft:stick minecraft:stick minecraft:stick minecraft:planks minecraft:stick minecraft:planks 0 polycraft:sack_polyisoprene_pellets 0",
    }

    def __init__(self):
        super().__init__()
        self._actions = Craft.class_actions


class Collect(ActionType):
    class_actions = {1: "COLLECT"}

    def __init__(self):
        super().__init__()
        self._actions = Collect.class_actions


class PlaceTreeTap(ActionType):
    class_actions = {1: "PLACE_TREE_TAP"}

    def __init__(self):
        super().__init__()
        self._actions = PlaceTreeTap.class_actions


class Decoder:

    actions_decoder: Dict[int, Dict[int, str]]
    actions_size: List[int]
    blocks_decoder: Dict[str, int]
    blocks_size: int
    items_decoder: Dict[str, int]
    items_size: int
    entitys_decoder: Dict[str, int]
    entitys_size: int

    actions_decoder = {
        0: Move().actions,
        1: Turn().actions,
        2: Break().actions,
        3: TP().actions,
        4: Craft().actions,
        5: Collect().actions,
        6: PlaceTreeTap().actions,
    }
    actions_size = [
        Move().length,
        Turn().length,
        Break().length,
        TP().length,
        Craft().length,
        Collect().length,
        PlaceTreeTap().length,
    ]

    blocks_decoder = {
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
    blocks_size = len(blocks_decoder)

    items_decoder = {
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
    items_size = len(items_decoder)

    entitys_decoder = {
        "EntityPogoist": 0,
        "EntityTrader": 1,
        "EntityItem": 2,
    }
    entitys_size = len(entitys_decoder)

    @staticmethod
    def update_actions(sense_all: Dict = None) -> None:
        TP.update_actions(sense_all)
        Decoder.actions_decoder[3] = TP().actions

    @staticmethod
    def decode_action_type(lst: List[int]) -> str:
        # print(lst)
        action = [(index, value) for index, value in enumerate(lst) if value > 0]

        if not action:
            return "NOP"
        else:
            action = action[0]
            return Decoder.actions_decoder[action[0]][action[1]]

    @staticmethod
    def get_actions_size() -> List[int]:
        return Decoder.actions_size

    @staticmethod
    def decode_block_type(name) -> int:
        return Decoder.blocks_decoder[name]

    @staticmethod
    def get_blocks_size() -> int:
        return Decoder.blocks_size

    @staticmethod
    def decode_item_type(name) -> int:
        return Decoder.items_decoder[name]

    @staticmethod
    def get_items_size() -> int:
        return Decoder.items_size

    @staticmethod
    def decode_entity_type(name) -> int:
        return Decoder.entitys_decoder[name]

    @staticmethod
    def get_entitys_size() -> int:
        return Decoder.entitys_size


class UnblancedDiscrete(MultiDiscrete):
    """
    The unblanced-discrete action space consists of a series of discrete action spaces with different number of actions in each
    where sampling will choose one of the actions and sample it only (the others will be zero).


    Can be initialized as UnblancedDiscrete([ 5, 2, 2 ])
    Example for a sample [ 0, 1, 0 ]

    """

    # override method
    def sample(self) -> List[int]:
        length = len(self.nvec)
        filtering = np.zeros((length,), dtype=int)
        filtering[self.np_random.choice(length)] = 1
        action = super().sample()
        return action * filtering
