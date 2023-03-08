from utils import ActionsDecoder
from utils.single_action import *
from typing import Dict, List


class SingleActionDecoder(ActionsDecoder):
    def __init__(self):
        super().__init__()
        self.actions_list: Dict[int, SingleAction]
        self.actions_encoder: Dict[int, Dict[str, int]]
        self.actions_size: Dict[int, int]

        self.actions_list = {
            0: NOP(),
            1: Move(),
            2: Turn(),
            3: Break(),
            4: TP(),
            5: Craft(),
            6: Collect(),
            7: PlaceTreeTap(),
        }

        self.actions_encoder = {
            0: self.actions_list[0].encoder,
            1: self.actions_list[1].encoder,
            2: self.actions_list[2].encoder,
            3: self.actions_list[3].encoder,
            4: self.actions_list[4].encoder,
            5: self.actions_list[5].encoder,
            6: self.actions_list[6].encoder,
            7: self.actions_list[7].encoder,
        }

        self.actions_size = {
            0: self.actions_list[0].length,  # 1 - index 0 (NOP)
            1: self.actions_list[1].length,  # 4 - index 1-4 (MOVE)
            2: self.actions_list[2].length,  # 2 - index 5-6 (TURN)
            3: self.actions_list[3].length,  # 1 - index 7 (BREAK)
            4: self.actions_list[4].length,  # 1 - index 8 (TP)
            5: self.actions_list[5].length,  # 4 - index 9-12 (CRAFT)
            6: self.actions_list[6].length,  # 1 - index 13 (COLLECT)
            7: self.actions_list[7].length,  # 1 - index 14 (PLACE TREE TAP)
        }

    # overriding abstract method
    def decode_action_type(self, action: List[int]) -> List[str]:
        """Decode the action type from list of int to polycraft action string"""
        single_action = action[0]
        if single_action >= self.get_actions_size():
            raise ValueError(f"decode not found action '{single_action}'")

        if single_action == 8:  # if teleport
            return [f"TP_TO {action[1]},4,{action[2]}"]

        for index, size in self.actions_size.items():
            if single_action < size:
                break
            single_action -= size

        return [self.actions_list[index].actions[single_action]]

    # overriding abstract method
    def encode_action_type(self, action: str) -> List[int]:
        """Encode the action type from planning level string to list of int"""
        if action.startswith("TP_TO"):
            action = action.split(" ")[1].split(",")
            return [8, int(action[0]), int(action[1])]

        for i, dic in self.actions_encoder.items():
            for act, j in dic.items():
                if action == act:
                    return [j + sum(list(self.actions_size.values())[:i]), 0, 0]

        raise ValueError(f"encode not found action '{action}'")

    # overriding abstract method
    def decode_to_planning(self, action: List[int]) -> str:
        """Decode the action type from list of int to planning level string"""
        single_action = action[0]
        for i, dic in self.actions_encoder.items():
            for act, j in dic.items():
                if j + sum(list(self.actions_size.values())[:i]) == single_action:
                    if single_action == 8:
                        return f"TP_TO {action[1]},4,{action[2]}"
                    return act

        raise ValueError(f"decode to planning not found action '{action}'")
