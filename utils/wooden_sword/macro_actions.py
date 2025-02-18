from typing import Dict, List
from utils.macro_actions import MacroAction, TP_Break_And_Collect, TP_Update


class Craft(MacroAction):
    class_actions = {
        0: ["CRAFT 1 minecraft:log 0 0 0"],
        1: ["CRAFT 1 minecraft:planks 0 minecraft:planks 0"],
        2: [
            "NOP",
            "CRAFT 1 0 minecraft:planks 0 0 minecraft:planks 0 0 minecraft:stick 0",
        ],
    }
    class_encoder = {
        "CRAFT_PLANK": 0,
        "CRAFT_STICK": 1,
        "CRAFT_WOODEN_SWORD": 2,
    }

    def __init__(self):
        super().__init__()
        self._actions = Craft.class_actions
        self._encoder = Craft.class_encoder

    def update(self, location: str) -> None:
        self._actions[2][0] = f"TP_TO {location}"

    def meet_requirements(
        self, action: int, state: dict = None, items_decoder: dict = None
    ) -> str:
        """If the action is not available in the this state, return NOP"""
        if (
            (action == 0)
            or (action == 1)
            or (
                action == 2
                and state["inventory"][items_decoder["minecraft:planks"]] > 1
                and state["inventory"][items_decoder["minecraft:stick"]] > 0
            )
        ):
            return self._actions[action]
        else:
            return ["NOP"]
