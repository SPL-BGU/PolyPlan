from utils import AdvancedActionsDecoder as ActionsDecoder
from utils.wooden_sword.advanced_actions import *
from typing import Dict, List
from math import sqrt


class AdvancedActionsDecoder(ActionsDecoder):
    def __init__(self, map_size: int):
        super().__init__(map_size)
        self.advanced_actions: Dict[int, MacroAction]
        self.actions_encoder: Dict[int, Dict[str, int]]
        self.actions_size: Dict[int, int]

        self.items_decoder = {
            "minecraft:log": 0,
            "minecraft:planks": 1,
            "minecraft:stick": 2,
            "minecraft:wooden_sword": 3,
        }
        self.items_size = len(self.items_decoder)

        self.advanced_actions = {
            0: TP(int(sqrt(map_size))),
            1: Break(),
            2: Craft(),
        }

        self.actions_encoder = {
            0: self.advanced_actions[0].encoder,
            1: self.advanced_actions[1].encoder,
            2: self.advanced_actions[2].encoder,
        }

        self.actions_size = {
            0: self.advanced_actions[
                0
            ].length,  # map_size - index 0-(map_size-1) (TP_TO)
            1: self.advanced_actions[1].length,  # 1 - index map_size (BREAK)
            2: self.advanced_actions[
                2
            ].length,  # 3 - index (map_size+1)-(map_size+3) (CRAFT)
        }

        self.agent_state = None
        self.map_size = map_size
        self.crafting_table_cell = 589
