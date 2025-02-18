from typing import Dict, List
from utils.macro_actions import MacroAction
from utils.advanced_actions import TP, Break, TP_Update
from utils.wooden_sword.macro_actions import Craft as CraftMacro
from math import sqrt


class Craft(CraftMacro):
    def __init__(self):
        super().__init__()
        self.crafting_table_location = None

    def meet_requirements(
        self, action: int, state: dict = None, items_decoder: dict = None
    ) -> str:
        """If the action is not available in the this state, return NOP"""
        if (action in [0, 1]) or (
            action == 2
            and state["inventory"][items_decoder["minecraft:planks"]] >= 2
            and state["inventory"][items_decoder["minecraft:stick"]] >= 1
            and state["position"] != [self.crafting_table_location]
        ):
            if action in [2]:
                state["position"] = [self.crafting_table_location]
            return self._actions[action]
        else:
            return ["NOP"]
