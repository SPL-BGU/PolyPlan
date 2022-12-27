from utils.macro_actions import *
from typing import Dict


class Decoder:
    macro_actions: Dict[int, MacroAction]
    actions_encoder: Dict[int, Dict[str, int]]
    actions_size: Dict[int, int]
    blocks_decoder: Dict[str, int]
    blocks_size: int
    items_decoder: Dict[str, int]
    items_size: int
    entitys_decoder: Dict[str, int]
    entitys_size: int

    macro_actions = {
        0: TP_Break_And_Collect(),
        1: Craft(),
        2: PlaceTreeTap(),
    }

    actions_encoder = {
        0: macro_actions[0].encoder,
        1: macro_actions[1].encoder,
        2: macro_actions[2].encoder,
    }

    actions_size = {
        0: macro_actions[0].length,
        1: macro_actions[1].length,
        2: macro_actions[2].length,
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
    def update_tp(sense_all: Dict) -> None:
        TP_Update.update_actions(
            sense_all,
            Decoder.macro_actions[0],
            Decoder.macro_actions[1],
            Decoder.macro_actions[2],
        )

    @staticmethod
    def decode_action_type(action: int) -> str:
        if action >= Decoder.get_actions_size():
            raise ValueError(f"decode not found action '{action}'")

        for index, size in Decoder.actions_size.items():
            if action < size:
                break
            action -= size

        if index == 0:
            return Decoder.macro_actions[0].next_location()

        return Decoder.macro_actions[index].actions[action]

    @staticmethod
    def encode_action_type(action: str) -> int:
        for i, dic in Decoder.actions_encoder.items():
            for act, j in dic.items():
                if action == act:
                    return j + sum(list(Decoder.actions_size.values())[:i])

        raise ValueError(f"encode not found action '{action}'")

    @staticmethod
    def get_actions_size() -> int:
        return sum(Decoder.actions_size.values())

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
