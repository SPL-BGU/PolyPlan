from envs import PolycraftGymEnv
from gym.spaces import Box, Discrete
from gym.spaces import Dict as GymDict
from gym.spaces import flatten_space, flatten
import numpy as np
from math import sqrt
from collections import OrderedDict
from utils import AdvancedActionsDecoder
from typing import Union, List


class AdvancedMinecraft(PolycraftGymEnv):
    """
    Create the advanced minecraft environment
    Where pal_path must be updated in the config.py file to work

    args:
        visually: if True, the environment will be displayed in the screen
        start_pal: if True, the pal will be started
        keep_alive: if True, the pal will be kept alive after the environment is closed
        rounds: actions in the environment until reset
    """

    def __init__(self, map_size=30, rounds: int = 32, **kwargs):
        map_size_square = map_size**2

        # PolycraftGymEnv
        super().__init__(
            rounds=rounds, **kwargs, decoder=AdvancedActionsDecoder(map_size_square)
        )

        self.map_size = map_size

        # basic minecraft environment observation space
        self._observation_space = GymDict(
            {
                "blockInFront": Box(
                    low=0,
                    high=self.decoder.get_blocks_size(),
                    shape=(1,),
                    dtype=np.uint8,
                ),  # 4
                "gameMap": Box(
                    low=0,
                    high=self.decoder.get_blocks_size(),
                    shape=(map_size_square,),
                    dtype=np.uint8,
                ),
                "inventory": Box(
                    low=0,
                    high=64,  # 64 is the max stack size
                    shape=(self.decoder.get_items_size(),),  # 6
                    dtype=np.uint8,
                ),  # count of each item in the inventory
                "position": Box(
                    low=0,
                    high=map_size_square,
                    shape=(1,),
                    dtype=np.int16,
                ),
            }
        )
        self.observation_space = flatten_space(self._observation_space)

        self.action_space = Discrete(
            self.decoder.get_actions_size()
        )  # 6 + actual map size

        # current state start with all zeros
        self._state = OrderedDict(
            {
                "blockInFront": np.zeros(
                    (1,),
                    dtype=np.uint8,
                ),
                "gameMap": np.zeros(
                    (map_size_square,),
                    dtype=np.uint8,
                ),
                "inventory": np.zeros(
                    (self.decoder.get_items_size(),),
                    dtype=np.uint8,
                ),
                "position": np.zeros(
                    (1,),
                    dtype=np.int16,
                ),
            }
        )
        self.state = flatten(self._observation_space, self._state)

        self.last_pos = {
            "position": np.zeros(
                (1,),
                dtype=np.int16,
            )
        }

        self.decoder.agent_state = self.last_pos

    def move_to_start(self) -> None:
        sense_all = self._senses()
        pos_x = sense_all["player"]["pos"][0]
        pos_z = sense_all["player"]["pos"][2]
        position = (pos_x - 1) + ((pos_z - 1) * self.map_size)
        self._state["position"][0] = position
        self.last_pos["position"][0] = position

    def step(self, action: int) -> tuple:
        self.last_pos["position"][0] = self._state["position"][0]
        return super().step(action)

    def _sense_all(self) -> None:
        """Sense the environment - update the state and get reward"""
        self.reward = 0

        # get the state from the Polycraft server
        sense_all = self._senses()

        inventory_before = self._state["inventory"].copy()

        self._state["blockInFront"][0] = self.decoder.decode_block_type(
            sense_all["blockInFront"]["name"]
        )

        # update the gameMap
        gameMap = np.zeros(
            (self.map_size, self.map_size, 1),
            dtype=np.uint8,
        )
        for pos_x in range(1, self.map_size + 1):
            for pos_z in range(1, self.map_size + 1):
                gameMap[pos_x - 1][pos_z - 1][0] = self.decoder.decode_block_type(
                    sense_all["map"][f"{pos_x},{4},{pos_z}"]["name"]
                )
        self._state["gameMap"] = gameMap.ravel()  # flatten the map to 1D vector

        # update the inventory
        inventory = np.zeros(
            (self.decoder.get_items_size(),),
            dtype=np.uint8,
        )
        for location, item in sense_all["inventory"].items():
            if location == "selectedItem":
                continue
            location = int(location)
            inventory[self.decoder.decode_item_type(item["item"])] = item["count"]
        self._state["inventory"] = inventory

        inventory_after = self._state["inventory"].copy()

        # update the reward
        reward = 0
        change = [int(inventory_after[i]) - int(inventory_before[i]) for i in range(6)]
        # if change[0] > 0:  # log
        #     reward += 0.002
        # if change[1] > 0:  # planks
        #     reward += 0.004
        # if change[2] > 0:  # sticks
        #     reward += 0.004
        # if change[3] > 0:  # get rubber
        #     reward += 0.1
        # if change[4] > 0:  # tree tap
        #     reward += 0.1
        if change[5] > 0:  # wooden pogo
            reward += 1

        # update the reward
        self.reward = reward
        self.collected_reward += self.reward

        self.state = flatten(self._observation_space, self._state)

        return self.reward

    def is_game_over(self) -> bool:
        done = (self.reward == 1) or (self.rounds_left == 0)
        self.done = done
        return done
