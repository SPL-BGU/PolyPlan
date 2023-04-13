from utils import ActionsDecoder
from utils.advanced_actions import *
from typing import Dict, List


class AdvancedActionsDecoder(ActionsDecoder):
    def __init__(self):
        super().__init__()
        self.advanced_actions: Dict[int, MacroAction]
        self.actions_encoder: Dict[int, Dict[str, int]]
        self.actions_size: Dict[int, int]

        self.blocks_decoder = {
            "minecraft:air": 0,
            "minecraft:log": 1,
            "minecraft:crafting_table": 2,
            "minecraft:bedrock": 3,
        }
        self.blocks_size = len(self.blocks_decoder)

        self.items_decoder = {
            "minecraft:log": 0,
            "minecraft:planks": 1,
            "minecraft:stick": 2,
            "polycraft:sack_polyisoprene_pellets": 3,
            "polycraft:tree_tap": 4,
            "polycraft:wooden_pogo_stick": 5,
        }
        self.items_size = len(self.items_decoder)

        self.advanced_actions = {
            0: TP(),
            1: Break(),
            2: Craft(),
            3: PlaceTreeTap(),
        }

        self.actions_encoder = {
            0: self.advanced_actions[0].encoder,
            1: self.advanced_actions[1].encoder,
            2: self.advanced_actions[2].encoder,
            3: self.advanced_actions[3].encoder,
        }

        self.actions_size = {
            0: self.advanced_actions[0].length,  # 900 - index 0-899 (TP_TO)
            1: self.advanced_actions[1].length,  # 1 - index 900 (BREAK)
            2: self.advanced_actions[2].length,  # 4 - index 901-904 (CRAFT)
            3: self.advanced_actions[3].length,  # 1 - index 905 (PLACE TREE TAP)
        }

        self.agent_state = None

    # overriding super method
    def update_tp(self, sense_all: Dict) -> None:
        TP_Update.update_actions(sense_all, self.advanced_actions[2])

    # overriding abstract method
    def decode_action_type(self, action: int, state: Dict) -> List[str]:
        """Decode the action type from list of int to polycraft action string"""
        if action >= self.get_actions_size():
            raise ValueError(f"decode not found action '{action}'")

        if action < 900:  # if teleport
            pos = action
            x_pos = pos % 30 + 1
            z_pos = pos // 30 + 1

            return [f"TP_TO {x_pos},4,{z_pos}"]

        for index, size in self.actions_size.items():
            if action < size:
                break
            action -= size

        return self.advanced_actions[index].meet_requirements(
            action, state, self.items_decoder
        )

    # overriding abstract method
    def encode_action_type(self, action: str) -> int:
        """Encode the action type from planning level string to list of int"""
        if action.startswith("TP_TO"):
            action = action.split(" ")[1].split(",")
            position = (int(action[0]) - 1) + ((int(action[1]) - 1) * 30)
            return position

        for i, dic in self.actions_encoder.items():
            for act, j in dic.items():
                if action == act:
                    return j + sum(list(self.actions_size.values())[:i])

        raise ValueError(f"encode not found action '{action}'")

    # overriding abstract method
    def decode_to_planning(self, action: int) -> str:
        """Decode the action type from list of int to planning level string"""
        location = self.agent_state["position"][0]
        if action < 900:
            return f"TP_TO cell{location} cell{action}"

        for i, dic in self.actions_encoder.items():
            for act, j in dic.items():
                if j + sum(list(self.actions_size.values())[:i]) == action:
                    if action == 901 or action == 902:
                        return act
                    return f"{act} cell{location}"

        raise ValueError(f"decode to planning not found action '{action}'")
