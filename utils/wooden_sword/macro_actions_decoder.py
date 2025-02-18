from utils import MacroActionsDecoder as ActionsDecoder
from utils.wooden_sword.macro_actions import *
from typing import Dict, List


class MacroActionsDecoder(ActionsDecoder):
    def __init__(self):
        super().__init__()
        self.tree_locations: List[str]
        self.macro_actions: Dict[int, MacroAction]
        self.actions_encoder: Dict[int, Dict[str, int]]
        self.actions_size: Dict[int, int]

        self.tree_locations = []

        self.items_decoder = {
            "minecraft:log": 0,
            "minecraft:planks": 1,
            "minecraft:stick": 2,
            "minecraft:wooden_sword": 3,
        }
        self.items_size = len(self.items_decoder)

        self.macro_actions = {
            0: TP_Break_And_Collect(),
            1: Craft(),
        }

        self.actions_encoder = {
            0: self.macro_actions[0].encoder,
            1: self.macro_actions[1].encoder,
        }

        self.actions_size = {
            0: self.macro_actions[0].length,  # 1 - index 0 (TP BREAK AND COLLECT)
            1: self.macro_actions[1].length,  # 3 - index 1-3 (CRAFT)
        }

    # overriding abstract method
    def decode_action_type(self, action: int, state: Dict) -> List[str]:
        """Decode the gym action to polycraft server action"""
        if action >= self.get_actions_size():
            raise ValueError(f"decode not found action '{action}'")

        for index, size in self.actions_size.items():
            if action < size:
                break
            action -= size

        if index != 1 and not self.tree_locations:  # if no place to teleport
            return ["NOP"]
        elif index == 0:  # if break tree
            self.macro_actions[0].update(self.tree_locations.pop(0))

        return self.macro_actions[index].meet_requirements(
            action, state, self.items_decoder
        )
