from typing import Dict, List
from utils.single_action import SingleAction


class MacroAction(SingleAction):

    _actions: Dict[int, List[str]]

    @property
    def actions(self) -> Dict[int, List[str]]:
        return self._actions


class TP_Break_And_Collect(MacroAction):
    class_actions = {
        0: [
            "NOP",
            "BREAK_BLOCK",
        ]
    }
    class_encoder = {"GET_LOG": 0}

    def __init__(self):
        super().__init__()
        self._actions = TP_Break_And_Collect.class_actions
        self._encoder = TP_Break_And_Collect.class_encoder

    def update(self, location: str) -> None:
        self._actions[0][0] = f"TP_TO {location}"


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
        "CRAFT_PLANK": 0,
        "CRAFT_STICK": 1,
        "CRAFT_TREE_TAP": 2,
        "CRAFT_WOODEN_POGO": 3,
    }

    def __init__(self):
        super().__init__()
        self._actions = Craft.class_actions
        self._encoder = Craft.class_encoder

    def update(self, location: str) -> None:
        self._actions[2][0] = f"TP_TO {location}"
        self._actions[3][0] = f"TP_TO {location}"

    def meet_requirements(
        self, action: int, state: dict = None, items_decoder: dict = None
    ) -> str:
        """If the action is not available in the this state, return NOP"""
        if (
            (action == 0)
            or (action == 1)
            or (
                action == 2
                and state["inventory"][items_decoder["minecraft:planks"]] > 4
                and state["inventory"][items_decoder["minecraft:stick"]] > 0
            )
            or (
                action == 3
                and state["inventory"][items_decoder["minecraft:planks"]] > 1
                and state["inventory"][items_decoder["minecraft:stick"]] > 3
                and state["inventory"][
                    items_decoder["polycraft:sack_polyisoprene_pellets"]
                ]
                > 0
            )
        ):
            return self._actions[action]
        else:
            return ["NOP"]


class PlaceTreeTap(MacroAction):
    class_actions = {0: ["NOP", "MOVE D", "PLACE_TREE_TAP", "COLLECT"]}
    class_encoder = {"PLACE_TREE_TAP": 0}

    def __init__(self):
        super().__init__()
        self._actions = PlaceTreeTap.class_actions
        self._encoder = PlaceTreeTap.class_encoder

    def update(self, location: str) -> None:
        self._actions[0][0] = f"TP_TO {location}"

    def meet_requirements(
        self, action: int, state: dict = None, items_decoder: dict = None
    ) -> str:
        """If the action is not available in the this state, return NOP"""
        if action == 0 and state["inventory"][items_decoder["polycraft:tree_tap"]] > 0:
            return self._actions[action]
        else:
            return ["NOP"]


class TP_Update:
    @staticmethod
    def update_actions(
        sense_all: Dict,
        craft: Craft,
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
        return tree_locations
