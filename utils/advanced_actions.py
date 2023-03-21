from typing import Dict, List
from utils.macro_actions import MacroAction, Craft


class TP(MacroAction):
    class_actions = {i: ["TP_TO"] for i in range(900)}
    class_encoder = {"TP_TO": 0}

    def __init__(self):
        super().__init__()
        self._actions = TP.class_actions
        self._encoder = TP.class_encoder


class Break(MacroAction):
    class_actions = {0: ["BREAK_BLOCK"]}
    class_encoder = {"BREAK": 0}

    def __init__(self):
        super().__init__()
        self._actions = Break.class_actions
        self._encoder = Break.class_encoder


class PlaceTreeTap(MacroAction):
    class_actions = {
        0: ["MOVE D", "PLACE_TREE_TAP", "COLLECT", "BREAK_BLOCK", "MOVE A"]
    }
    class_encoder = {"PLACE_TREE_TAP": 0}

    def __init__(self):
        super().__init__()
        self._actions = PlaceTreeTap.class_actions
        self._encoder = PlaceTreeTap.class_encoder


class TP_Update:
    @staticmethod
    def update_actions(
        sense_all: Dict,
        craft: Craft,
    ) -> None:
        crafting_table_location = "NOP"

        # find the crafting table location
        for location, block in sense_all["map"].items():
            if block["name"] == "minecraft:crafting_table":
                crafting_table_location = location
                break

        # update the actions
        craft.update(crafting_table_location)
