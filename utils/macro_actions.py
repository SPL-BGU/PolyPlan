from typing import Dict, List
import random


class MacroAction:

    _actions: Dict[int, List[str]]
    _encoder: Dict[str, int]

    def __init__(self):
        self._actions = None

    @property
    def actions(self) -> Dict[int, str]:
        return self._actions

    @property
    def length(self) -> int:
        return len(self._actions)

    @property
    def encoder(self) -> Dict[str, int]:
        return self._encoder


class TP_Break_And_Collect(MacroAction):
    class_actions = {
        0: [
            "NOP",
            "BREAK_BLOCK",
        ]
    }
    class_encoder = {"GET TREE": 0}

    def __init__(self):
        super().__init__()
        self._actions = TP_Break_And_Collect.class_actions
        self._encoder = TP_Break_And_Collect.class_encoder
        self.tree_locations = []

    def update(self, locations: list) -> None:
        self.tree_locations = locations

    def next_location(self) -> List[str]:
        if self.tree_locations:
            self._actions[0][0] = f"TP_TO {self.tree_locations.pop(0)}"
            return self._actions[0]
        else:
            return ["NOP"]


class Craft(MacroAction):
    class_actions = {
        0: ["CRAFT 1 minecraft:log 0 0 0"],
        1: ["CRAFT 1 minecraft:planks 0 minecraft:planks 0"],
        2: [
            "NOP",
            "CRAFT 1 minecraft:planks minecraft:stick minecraft:planks minecraft:planks 0 minecraft:planks 0 minecraft:planks 0",
        ],
        3: [
            "NOP",
            "CRAFT 1 minecraft:stick minecraft:stick minecraft:stick minecraft:planks minecraft:stick minecraft:planks 0 polycraft:sack_polyisoprene_pellets 0",
        ],
    }
    class_encoder = {
        "CRAFT PLANK": 0,
        "CRAFT STICK": 1,
        "CRAFT TREE_TAP": 2,
        "CRAFT WOODEN_POGO": 3,
    }

    def __init__(self):
        super().__init__()
        self._actions = Craft.class_actions
        self._encoder = Craft.class_encoder

    def update(self, location: str) -> None:
        self._actions[2][0] = f"TP_TO {location}"
        self._actions[3][0] = f"TP_TO {location}"


class PlaceTreeTap(MacroAction):
    class_actions = {0: ["NOP", "MOVE D", "PLACE_TREE_TAP", "COLLECT"]}
    class_encoder = {"PLACE TREE_TAP": 0}

    def __init__(self):
        super().__init__()
        self._actions = PlaceTreeTap.class_actions
        self._encoder = PlaceTreeTap.class_encoder

    def update(self, location: str) -> None:
        self._actions[0][0] = f"TP_TO {location}"


class TP_Update:
    @staticmethod
    def update_actions(
        sense_all: Dict,
        tree_chop: TP_Break_And_Collect,
        craft: Craft,
        place_treetap: PlaceTreeTap,
    ) -> None:
        tree_locations = []
        crafting_table_location = "NOP"

        # find all the blocks that can be TP to
        for location, block in sense_all["map"].items():
            if block["name"] == "minecraft:log":
                tree_locations.append(location)
            elif block["name"] == "minecraft:crafting_table":
                crafting_table_location = location

        random.shuffle(tree_locations)  # add randomness to the order

        # update the actions
        craft.update(crafting_table_location)
        place_treetap.update(tree_locations.pop(0))
        tree_chop.update(tree_locations)
