from envs import PolycraftGymEnv
from gym.spaces import Box, Discrete
from gym.spaces import Dict as GymDict
from gym.spaces import flatten_space, flatten
import numpy as np
from collections import OrderedDict
from utils import IntermediateActionsDecoder
from typing import Union, List
import time


class IntermediateMinecraft(PolycraftGymEnv):
    """
    Create the basic intermediate environment
    Where pal_path must be updated in the config.py file to work

    args:
        visually: if True, the environment will be displayed in the screen
        start_pal: if True, the pal will be started
        keep_alive: if True, the pal will be kept alive after the environment is closed
        max_steps: actions in the environment until reset
    """

    def __init__(self, max_steps: int = 128, **kwargs):
        # PolycraftGymEnv
        super().__init__(
            max_steps=max_steps, **kwargs, decoder=IntermediateActionsDecoder()
        )

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
                    shape=(6,),
                    dtype=np.uint8,
                ),  # 5 trees, 1 crafting table
                "inventory": Box(
                    low=0,
                    high=64,  # 64 is the max stack size
                    shape=(self.decoder.get_items_size(),),  # 6
                    dtype=np.uint8,
                ),  # count of each item in the inventory
                "position": Box(
                    low=0,
                    high=5,
                    shape=(1,),
                    dtype=np.uint8,
                ),  # point in map
            }
        )
        self.observation_space = flatten_space(self._observation_space)

        self.action_space = Discrete(self.decoder.get_actions_size())  # 12

        # current state start with all zeros
        self._state = OrderedDict(
            {
                "blockInFront": np.array(
                    [2],
                    dtype=np.uint8,
                ),
                "gameMap": np.array(
                    [2, 1, 1, 1, 1, 1],
                    dtype=np.uint8,
                ),
                "inventory": np.zeros(
                    (self.decoder.get_items_size(),),
                    dtype=np.uint8,
                ),
                "position": np.array(
                    [0],
                    dtype=np.uint8,
                ),
            }
        )
        self.state = flatten(self._observation_space, self._state)

        self.last_pos = {
            "position": np.zeros(
                (1,),
                dtype=np.uint8,
            )
        }

        self.decoder.agent_state = self.last_pos

    def move_to_start(self) -> None:
        self.step(0)
        self._state["gameMap"][:] = [2, 1, 1, 1, 1, 1]

    def step(self, action: int) -> tuple:
        self.last_pos["position"][0] = self._state["position"][0]
        return super().step(action)

    def _sense_all(self) -> None:
        """Sense the environment - update the state and get reward"""
        self.reward = 0

        # get the state from the Polycraft server
        sense_all = self._senses()

        self._state["blockInFront"][0] = self.decoder.decode_block_type(
            sense_all["blockInFront"]["name"]
        )

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

        # update the reward, binary reward - achieved the goal or not
        self.reward = (
            int(sense_all["goal"]["goalAchieved"]) if "goal" in sense_all else 0
        )
        self.collected_reward += self.reward

        self.state = flatten(self._observation_space, self._state)

        return self.reward
