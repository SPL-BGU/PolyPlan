from abc import ABC
from typing import Dict


class SingleAction(ABC):
    _actions: Dict[int, str]
    _encoder: Dict[str, int]

    def __init__(self):
        self._actions = None
        self._encoder = None

    @property
    def actions(self) -> Dict[int, str]:
        return self._actions

    @property
    def length(self) -> int:
        return len(self._actions)

    @property
    def encoder(self) -> Dict[str, int]:
        return self._encoder

    def meet_requirements(
        self, action: int, state: dict = None, items_decoder: dict = None
    ) -> str:
        """If the action is not available in the this state, return NOP"""
        return self._actions[action]


class NOP(SingleAction):
    class_actions = {0: "NOP"}
    class_encoder = {"NOP": 0}

    def __init__(self):
        super().__init__()
        self._actions = NOP.class_actions
        self._encoder = NOP.class_encoder


class Move(SingleAction):
    class_actions = {0: "MOVE W", 1: "MOVE A", 2: "MOVE D", 3: "MOVE X"}
    class_encoder = {"MOVE_W": 0, "MOVE_A": 1, "MOVE_D": 2, "MOVE_X": 3}

    def __init__(self):
        super().__init__()
        self._actions = Move.class_actions
        self._encoder = Move.class_encoder


class Turn(SingleAction):
    class_actions = {0: "TURN 90", 1: "TURN -90"}
    class_encoder = {"TURN_RIGHT": 0, "TURN_LEFT": 1}

    def __init__(self):
        super().__init__()
        self._actions = Turn.class_actions
        self._encoder = Turn.class_encoder


class Break(SingleAction):
    class_actions = {0: "BREAK_BLOCK"}
    class_encoder = {"BREAK": 0}

    def __init__(self):
        super().__init__()
        self._actions = Break.class_actions
        self._encoder = Break.class_encoder


class TP(SingleAction):
    class_actions = {0: "TP_TO"}
    class_encoder = {"TP_TO": 0}

    def __init__(self):
        super().__init__()
        self._actions = TP.class_actions
        self._encoder = TP.class_encoder


class Craft(SingleAction):
    class_actions = {
        0: "CRAFT 1 minecraft:log 0 0 0",
        1: "CRAFT 1 minecraft:planks 0 minecraft:planks 0",
        2: "CRAFT 1 minecraft:planks minecraft:stick minecraft:planks minecraft:planks 0 minecraft:planks 0 minecraft:planks 0",
        3: "CRAFT 1 minecraft:stick minecraft:stick minecraft:stick minecraft:planks minecraft:stick minecraft:planks 0 polycraft:sack_polyisoprene_pellets 0",
    }
    class_encoder = {
        "CRAFT_PLANK": 0,
        "CRAFT_STICK": 1,
        "CRAFT_TREE_TAP": 2,
        "CRAFT_WOODEN_POGO": 3,
    }

    def __init__(self):
        super().__init__()
        self._actions = Craft.class_actions
        self._encoder = Craft.class_encoder


class Collect(SingleAction):
    class_actions = {0: "COLLECT"}
    class_encoder = {"COLLECT": 0}

    def __init__(self):
        super().__init__()
        self._actions = Collect.class_actions
        self._encoder = Collect.class_encoder


class PlaceTreeTap(SingleAction):
    class_actions = {0: "PLACE_TREE_TAP"}
    class_encoder = {"PLACE_TREE_TAP": 0}

    def __init__(self):
        super().__init__()
        self._actions = PlaceTreeTap.class_actions
        self._encoder = PlaceTreeTap.class_encoder
