from typing import Dict


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


class NOP(ActionType):
    class_actions = {0: "NOP"}

    def __init__(self):
        super().__init__()
        self._actions = NOP.class_actions


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
        for index in range(min(len(TP.class_actions), len(tp_blocks)) + 1):
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
