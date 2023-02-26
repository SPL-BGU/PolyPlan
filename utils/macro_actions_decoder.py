from utils.macro_actions import *
from typing import Dict


class MacroActionsDecoder:
    def __init__(self):
        self.tree_locations: List[str]
        self.macro_actions: Dict[int, MacroAction]
        self.actions_encoder: Dict[int, Dict[str, int]]
        self.actions_size: Dict[int, int]
        self.blocks_decoder: Dict[str, int]
        self.blocks_size: int
        self.items_decoder: Dict[str, int]
        self.items_size: int
        self.entitys_decoder: Dict[str, int]
        self.entitys_size: int
        self.directions_decoder: Dict[str, int]

        self.tree_locations = []

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
            0: self.macro_actions[0].length,
            1: self.macro_actions[1].length,
            2: self.macro_actions[2].length,
        }

        self.blocks_decoder = {
            "minecraft:air": 0,
            "minecraft:bedrock": 1,
            "polycraft:block_of_platinum": 2,
            "minecraft:crafting_table": 3,
            "minecraft:diamond_ore": 4,
            "minecraft:log": 5,
            "polycraft:plastic_chest": 6,
            "minecraft:sapling": 7,
            "polycraft:tree_tap": 8,
            "minecraft:wooden_door": 9,
            "polycraft:safe": 10,
        }
        self.blocks_size = len(self.blocks_decoder)

        self.items_decoder = {
            "polycraft:block_of_platinum": 0,
            "polycraft:block_of_titanium": 1,
            "minecraft:crafting_table": 2,
            "minecraft:diamond": 3,
            "minecraft:diamond_block": 4,
            "minecraft:iron_pickaxe": 5,
            "polycraft:key": 6,
            "minecraft:log": 7,
            "minecraft:planks": 8,
            "minecraft:sapling": 9,
            "minecraft:stick": 10,
            "polycraft:sack_polyisoprene_pellets": 11,
            "polycraft:tree_tap": 12,
            "polycraft:wooden_pogo_stick": 13,
        }
        self.items_size = len(self.items_decoder)

        self.entitys_decoder = {
            "EntityPogoist": 0,
            "EntityTrader": 1,
            "EntityItem": 2,
        }
        self.entitys_size = len(self.entitys_decoder)

        self.directions_decoder = {"NORTH": 0, "EAST": 1, "SOUTH": 2, "WEST": 3}

    def update_tp(self, sense_all: Dict) -> None:
        self.tree_locations = TP_Update.update_actions(sense_all, self.macro_actions[1])

    def decode_action_type(self, action: int) -> str:
        """Decode the action type from int to polycraft action string"""
        if action >= self.get_actions_size():
            raise ValueError(f"decode not found action '{action}'")

        for index, size in self.actions_size.items():
            if action < size:
                break
            action -= size

        if index != 1 and not self.tree_locations:
            return ["NOP"]
        elif index == 0:
            self.macro_actions[0].update(self.tree_locations.pop(0))
        elif index == 2:
            self.macro_actions[2].update(self.tree_locations[0])

        return self.macro_actions[index].actions[action]

    def encode_action_type(self, action: str) -> int:
        """Encode the action type from planning level string to int"""
        for i, dic in self.actions_encoder.items():
            for act, j in dic.items():
                if action == act:
                    return j + sum(list(self.actions_size.values())[:i])

        raise ValueError(f"encode not found action '{action}'")

    def decode_to_planning(self, action: int) -> str:
        """Decode the action type from int to planning level string"""
        for i, dic in self.actions_encoder.items():
            for act, j in dic.items():
                if j + sum(list(self.actions_size.values())[:i]) == action:
                    return act

        raise ValueError(f"decode to planning not found action '{action}'")

    def get_actions_size(self) -> int:
        return sum(self.actions_size.values())

    def decode_block_type(self, name) -> int:
        return self.blocks_decoder[name]

    def get_blocks_size(self) -> int:
        return self.blocks_size

    def decode_item_type(self, name) -> int:
        return self.items_decoder[name]

    def get_items_size(self) -> int:
        return self.items_size

    def decode_entity_type(self, name) -> int:
        return self.entitys_decoder[name]

    def get_entitys_size(self) -> int:
        return self.entitys_size

    def decode_direction(self, direction) -> int:
        return self.directions_decoder[direction]
