from utils import ActionsDecoder
from utils.advanced_actions import *
from typing import Dict, List
from math import sqrt


class AdvancedActionsDecoder(ActionsDecoder):
    def __init__(self, map_size: int):
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
            0: TP(int(sqrt(map_size))),
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
            0: self.advanced_actions[
                0
            ].length,  # map_size - index 0-(map_size-1) (TP_TO)
            1: self.advanced_actions[1].length,  # 1 - index map_size (BREAK)
            2: self.advanced_actions[
                2
            ].length,  # 4 - index (map_size+1)-(map_size+4) (CRAFT)
            3: self.advanced_actions[3].length,  # 1 - index map_size+5 (PLACE TREE TAP)
        }

        self.agent_state = None
        self.map_size = map_size
        self.crafting_table_cell = 589

    # overriding super method
    def update_tp(self, sense_all: Dict) -> None:
        TP_Update.update_actions(sense_all, self.advanced_actions[2])
        crafting_table_cell = (
            self.advanced_actions[2].actions[2][0].split(" ")[1].split(",")
        )
        map_size = int(sqrt(self.map_size))
        self.crafting_table_cell = (int(crafting_table_cell[0]) - 1) + (
            (int(crafting_table_cell[2]) - 1) * map_size
        )
        self.advanced_actions[2].crafting_table_location = self.crafting_table_cell

    # overriding abstract method
    def decode_action_type(self, action: int, state: Dict) -> List[str]:
        """Decode the gym action to polycraft server action"""
        if action >= self.get_actions_size():
            raise ValueError(f"decode not found action '{action}'")

        if action < self.map_size:  # if teleport
            return self.advanced_actions[0].meet_requirements(
                action, state, self.items_decoder
            )

        for index, size in self.actions_size.items():
            if action < size:
                break
            action -= size

        return self.advanced_actions[index].meet_requirements(
            action, state, self.items_decoder
        )

    # overriding abstract method
    def encode_human_action_type(self, action: str) -> int:
        """Encode the human readable action to gym action"""
        if action.startswith("TP_TO"):
            if "crafting_table" in action:
                position = self.crafting_table_cell
            else:
                action = action.split(" ")[1].split(",")
                map_size = int(sqrt(self.map_size))
                position = (int(action[0]) - 1) + ((int(action[1]) - 1) * map_size)
            return position

        for i, dic in self.actions_encoder.items():
            for act, j in dic.items():
                if action == act:
                    return j + sum(list(self.actions_size.values())[:i])

        raise ValueError(f"encode not found action '{action}'")

    def encode_planning_action_type(self, action):
        """Encode the planning action to gym action"""
        action = action[1:-1].split(" ")
        action = action[0].upper() + " " + " ".join(action[1:])
        if action.startswith("TP_TO"):
            position = action.split(" ")[2]
            if position == "crafting_table":
                position = self.crafting_table_cell
            else:
                position = int(position.replace("cell", ""))
            return position

        for i, dic in self.actions_encoder.items():
            for act, j in dic.items():
                if action.startswith(act):
                    return j + sum(list(self.actions_size.values())[:i])

        raise ValueError(f"encode not found action '{action}'")

    # overriding abstract method
    def decode_to_planning(self, action: int, location: int = -1) -> str:
        """Decode the gym action to planning action"""
        if location == -1:
            location = self.agent_state["position"][0]
        if action < self.map_size:
            if location == self.crafting_table_cell:
                from_location = "crafting_table"
            else:
                from_location = f"cell{location}"
            if action == self.crafting_table_cell:
                to_location = "crafting_table"
            else:
                to_location = f"cell{action}"
            return f"TP_TO {from_location} {to_location}"

        for i, dic in self.actions_encoder.items():
            for act, j in dic.items():
                if j + sum(list(self.actions_size.values())[:i]) == action:
                    if action == (self.map_size + 1) or action == (self.map_size + 2):
                        return act
                    if location == self.crafting_table_cell:
                        return f"{act} crafting_table"
                    else:
                        return f"{act} cell{location}"

        raise ValueError(f"decode to planning not found action '{action}'")
