from typing import Dict, List
from utils.macro_actions import MacroAction, Craft
from utils.advanced_actions import Break, PlaceTreeTap


class TP(MacroAction):
    class_actions = {
        0: ["NOP"],
        1: ["NOP"],
        2: ["NOP"],
        3: ["NOP"],
        4: ["NOP"],
        5: ["NOP"],
    }
    class_encoder = {
        "TP_TO": 0,
    }

    def __init__(self):
        super().__init__()
        self._actions = TP.class_actions
        self._encoder = TP.class_encoder

    def update(self, crafting_table_location: str, trees_location: List[str]) -> None:
        self._actions[0][0] = f"TP_TO {crafting_table_location}"
        self._encoder = {}
        self._encoder[f"TP_TO {crafting_table_location}"] = 0
        for i in range(1, len(trees_location)):
            self._actions[i][0] = f"TP_TO {trees_location[i]}"
            self._encoder[f"TP_TO {trees_location[i]}"] = i


class TP_Update:
    @staticmethod
    def update_actions(
        sense_all: Dict,
        craft: Craft,
        tp: TP,
    ) -> list:
        tree_locations = []
        crafting_table_location = "NOP"

        # find all the blocks that can be TP to
        for location, block in sense_all["map"].items():
            if block["name"] == "minecraft:log":
                tree_locations.append(location)
            elif block["name"] == "minecraft:crafting_table":
                crafting_table_location = location

        # update the actions
        craft.update(crafting_table_location)
        tp.update(crafting_table_location, tree_locations)
