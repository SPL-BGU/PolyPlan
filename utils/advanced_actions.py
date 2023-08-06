from typing import Dict, List
from utils.macro_actions import MacroAction, Craft
from math import sqrt


class TP(MacroAction):
    class_actions = {0: "TP_TO"}
    class_encoder = {"TP_TO": 0}

    def __init__(self, map_size=30):
        super().__init__()
        self.map_size = map_size
        TP.class_actions = {i: ["TP_TO"] for i in range(map_size**2)}
        self._actions = TP.class_actions
        self._encoder = TP.class_encoder

    def meet_requirements(
        self, action: int, state: dict = None, items_decoder: dict = None
    ) -> str:
        """If the action is not available in the this state, return NOP"""
        to_pos = action
        x_pos = (to_pos % self.map_size) + 1
        z_pos = (to_pos // self.map_size) + 1

        state["position"] = [to_pos]

        return [f"TP_TO {x_pos},4,{z_pos}"]


class Break(MacroAction):
    class_actions = {0: ["BREAK_BLOCK"]}
    class_encoder = {"BREAK": 0}

    def __init__(self):
        super().__init__()
        self._actions = Break.class_actions
        self._encoder = Break.class_encoder

    def meet_requirements(
        self, action: int, state: dict = None, items_decoder: dict = None
    ) -> str:
        """If the action is not available in the this state, return NOP"""
        if action == 0 and state["blockInFront"][0] == 1:
            return self._actions[action]
        else:
            return ["NOP"]


class PlaceTreeTap(MacroAction):
    class_actions = {
        0: ["MOVE D", "PLACE_TREE_TAP", "COLLECT", "BREAK_BLOCK", "MOVE A"]
    }
    class_encoder = {"PLACE_TREE_TAP": 0}

    def __init__(self):
        super().__init__()
        self._actions = PlaceTreeTap.class_actions
        self._encoder = PlaceTreeTap.class_encoder

    def meet_requirements(
        self, action: int, state: dict = None, items_decoder: dict = None
    ) -> str:
        """If the action is not available in the this state, return NOP"""
        if (
            action == 0
            and state["blockInFront"][0] == 1
            and state["inventory"][items_decoder["polycraft:tree_tap"]] > 0
        ):
            return self._actions[action]
        else:
            return ["NOP"]


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
