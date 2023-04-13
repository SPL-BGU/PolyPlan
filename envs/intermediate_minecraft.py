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
        rounds: actions in the environment until reset
    """

    def __init__(self, **kwargs):
        # PolycraftGymEnv
        super().__init__(**kwargs, decoder=IntermediateActionsDecoder())

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
                "blockInFront": np.zeros(
                    (1,),
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

        self.max_rounds = 128
        self.decoder.agent_state = self._state

    def reset(self) -> np.ndarray:

        # reset the environment
        self.server_controller.send_command(f"RESET domain {self._domain_path}")
        if self.pal_owner:
            while "game initialization completed" not in str(self._next_line):
                self._next_line = self._check_queues()
        time.sleep(2)

        # reset the teleport according to the new domain
        sense_all = self.server_controller.send_command("SENSE_ALL NONAV")
        self.decoder.update_tp(sense_all)

        # reset the state
        self.collected_reward = 0
        self.action = None
        self.done = False
        self.step(0)

        self._state["position"][0] = 0
        self._state["gameMap"] = np.array(
            [2, 1, 1, 1, 1, 1],
            dtype=np.uint8,
        )
        self.rounds_left = self.max_rounds
        return self.state

    def step(self, action: Union[List[int], int]) -> np.ndarray:
        last_pos = self._state["position"][0]
        state, reward, done, info = super().step(action)
        info["last_pos"] = last_pos
        return state, reward, done, info

    def _sense_all(self) -> None:
        """Sense the environment - update the state and get reward"""
        self.reward = 0

        # get the state from the Polycraft server
        sense_all = self.server_controller.send_command("SENSE_ALL NONAV")

        inventory_before = self._state["inventory"].copy()

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

        inventory_after = self._state["inventory"].copy()

        # update the reward
        reward = 0
        change = [int(inventory_after[i]) - int(inventory_before[i]) for i in range(6)]
        if change[0] > 0:  # log
            reward += 1
        if change[1] > 0:  # planks
            reward += 2
        if change[2] > 0:  # sticks
            reward += 2
        if change[3] == 1 and inventory_after[3] == 1:  # first time get rubber
            reward = 50
        if change[4] > 0:  # tree tap
            reward += 50
        if change[5] > 0:  # wooden pogo
            reward += 500

        # update the reward
        self.reward = reward
        self.collected_reward += self.reward

        self.state = flatten(self._observation_space, self._state)

        return self.reward

    def is_game_over(self) -> bool:
        done = (self.reward == 500) or (self.rounds_left == 0)
        self.done = done
        return done
