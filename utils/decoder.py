from utils.actions_type import *
from typing import Dict, List


class Decoder:

    actions_decoder: Dict[int, Dict[int, str]]
    actions_size: Dict[int, int]
    blocks_decoder: Dict[str, int]
    blocks_size: int
    items_decoder: Dict[str, int]
    items_size: int
    entitys_decoder: Dict[str, int]
    entitys_size: int

    actions_decoder = {
        0: NOP().actions,
        1: Move().actions,
        2: Turn().actions,
        3: Break().actions,
        4: TP().actions,
        5: Craft().actions,
        6: Collect().actions,
        7: PlaceTreeTap().actions,
    }
    actions_size = {
        0: 0,  # NOP is no action and start at 0
        1: Move().length,
        2: Turn().length,
        3: Break().length,
        4: TP().length,
        5: Craft().length,
        6: Collect().length,
        7: PlaceTreeTap().length,
    }

    blocks_decoder = {
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
    blocks_size = len(blocks_decoder)

    items_decoder = {
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
    items_size = len(items_decoder)

    entitys_decoder = {
        "EntityPogoist": 0,
        "EntityTrader": 1,
        "EntityItem": 2,
    }
    entitys_size = len(entitys_decoder)

    @staticmethod
    def update_actions(sense_all: Dict = None) -> None:
        TP.update_actions(sense_all)
        Decoder.actions_decoder[4] = TP().actions

    @staticmethod
    def decode_action_type(action: int) -> str:
        for index, size in Decoder.actions_size.items():
            if action <= size:
                return Decoder.actions_decoder[index][action]
            action -= size

        raise ValueError(f"action '{action}' is out of range")

    @staticmethod
    def encode_action_type(action: str) -> int:
        if action == "NOP":
            return 0
        for index, act in Decoder.actions_decoder.items():
            if action in act.values():
                return (
                    list(act.values()).index(action)
                    + sum(list(Decoder.actions_size.values())[:index])
                    + 1
                )

        raise ValueError(f"action '{action}' not found")

    @staticmethod
    def get_actions_size() -> List[int]:
        return sum(Decoder.actions_size.values()) + 1  # +1 for the NOP action

    @staticmethod
    def decode_block_type(name) -> int:
        return Decoder.blocks_decoder[name]

    @staticmethod
    def get_blocks_size() -> int:
        return Decoder.blocks_size

    @staticmethod
    def decode_item_type(name) -> int:
        return Decoder.items_decoder[name]

    @staticmethod
    def get_items_size() -> int:
        return Decoder.items_size

    @staticmethod
    def decode_entity_type(name) -> int:
        return Decoder.entitys_decoder[name]

    @staticmethod
    def get_entitys_size() -> int:
        return Decoder.entitys_size
