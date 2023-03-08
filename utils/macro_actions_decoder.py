from utils import ActionsDecoder
from utils.macro_actions import *
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
            "polycraft:sack_polyisoprene_pellets": 3,
            "polycraft:tree_tap": 4,
            "polycraft:wooden_pogo_stick": 5,
        }
        self.items_size = len(self.items_decoder)

        self.macro_actions = {
            0: TP_Break_And_Collect(),
            1: Craft(),
            2: PlaceTreeTap(),
        }

        self.actions_encoder = {
            0: self.macro_actions[0].encoder,
            1: self.macro_actions[1].encoder,
            2: self.macro_actions[2].encoder,
        }

        self.actions_size = {
            0: self.macro_actions[0].length,  # 1 - index 0 (TP BREAK AND COLLECT)
            1: self.macro_actions[1].length,  # 4 - index 1-4 (CRAFT)
            2: self.macro_actions[2].length,  # 1 - index 5 (PLACE TREE TAP)
        }

    # overriding super method
    def update_tp(self, sense_all: Dict) -> None:
        self.tree_locations = TP_Update.update_actions(sense_all, self.macro_actions[1])

    # overriding abstract method
    def decode_action_type(self, action: int) -> List[str]:
        """Decode the action type from int to polycraft action string"""
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
        elif index == 2:  # if place tree tap
            self.macro_actions[2].update(self.tree_locations[0])

        return self.macro_actions[index].actions[action]

    # overriding abstract method
    def encode_action_type(self, action: str) -> int:
        """Encode the action type from planning level string to int"""
        for i, dic in self.actions_encoder.items():
            for act, j in dic.items():
                if action == act:
                    return j + sum(list(self.actions_size.values())[:i])

        raise ValueError(f"encode not found action '{action}'")

    # overriding abstract method
    def decode_to_planning(self, action: int) -> str:
        """Decode the action type from int to planning level string"""
        for i, dic in self.actions_encoder.items():
            for act, j in dic.items():
                if j + sum(list(self.actions_size.values())[:i]) == action:
                    return act

        raise ValueError(f"decode to planning not found action '{action}'")
