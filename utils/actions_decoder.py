from abc import ABC, abstractmethod
from typing import Dict


class ActionsDecoder(ABC):
    def __init__(self) -> None:
        self.blocks_decoder: Dict[str, int]
        self.blocks_size: int
        self.items_decoder: Dict[str, int]
        self.items_size: int
        self.entitys_decoder: Dict[str, int]
        self.entitys_size: int
        self.directions_decoder: Dict[str, int]

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

        self.actions_size = {}

    def update_tp(self, sense_all: Dict) -> None:
        return

    @abstractmethod
    def decode_action_type(self):
        """Decode the action type to polycraft action string"""
        raise NotImplementedError

    @abstractmethod
    def encode_human_action_type(self, action):
        """Encode the action type from planning level"""
        raise NotImplementedError

    def encode_planning_action_type(self, action):
        """Encode the planning action to gym action"""
        action = action[1:-1].split(" ")
        if len(action) == 1:
            action = action[0].upper()
        else:
            action = action[0].upper() + " " + " ".join(action[1:])

        return self.encode_human_action_type(action)

    @abstractmethod
    def decode_to_planning(self):
        """Decode the action type to planning"""
        raise NotImplementedError

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
