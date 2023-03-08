from envs import PolycraftGymEnv
from gym.spaces import Box, MultiDiscrete
from gym.spaces import Dict as GymDict
from gym.spaces import flatten_space, flatten
import numpy as np
from collections import OrderedDict
from utils import AdvancedActionsDecoder


class AdvancedMinecraft(PolycraftGymEnv):
    """
    Create the basic minecraft environment
    Where pal_path must be updated in the config.py file to work

    args:
        visually: if True, the environment will be displayed in the screen
        start_pal: if True, the pal will be started
        keep_alive: if True, the pal will be kept alive after the environment is closed
        rounds: actions in the environment until reset
    """

    def __init__(self, **kwargs):
        # PolycraftGymEnv
        super().__init__(**kwargs, decoder=AdvancedActionsDecoder())

        # basic minecraft environment observation space
        self._observation_space = GymDict(
            {
                "blockInFront": Box(
                    low=0,
                    high=self.decoder.get_blocks_size(),
                    shape=(1,),
                    dtype=np.uint8,
                ),  # 3
                "gameMap": Box(
                    low=0,
                    high=self.decoder.get_blocks_size(),
                    shape=(30 * 30,),
                    dtype=np.uint8,
                ),  # map (30*30)
                "inventory": Box(
                    low=0,
                    high=64,  # 64 is the max stack size
                    shape=(self.decoder.get_items_size(),),  # 6
                    dtype=np.uint8,
                ),  # count of each item in the inventory
                "position": Box(
                    low=0,
                    high=900,
                    shape=(1,),
                    dtype=np.uint8,
                ),  # point in map size (30*30)
            }
        )
        self.observation_space = flatten_space(self._observation_space)

        self.action_space = MultiDiscrete(
            [self.decoder.get_actions_size(), 30 * 30]
        )  # 6, 30*30 -> action, tp_pos

        # current state start with all zeros
        self._state = OrderedDict(
            {
                "blockInFront": np.zeros(
                    (1,),
                    dtype=np.uint8,
                ),
                "gameMap": np.zeros(
                    (30 * 30,),
                    dtype=np.uint8,
                ),
                "inventory": np.zeros(
                    (self.decoder.get_items_size(),),
                    dtype=np.uint8,
                ),
                "position": np.zeros(
                    (1,),
                    dtype=np.uint8,
                ),
            }
        )
        self.state = flatten(self._observation_space, self._state)

    def _sense_all(self) -> None:
        """Sense the environment - update the state and get reward"""
        self.reward = 0

        # get the state from the Polycraft server
        sense_all = self.server_controller.send_command("SENSE_ALL NONAV")

        self._state["blockInFront"][0] = self.decoder.decode_block_type(
            sense_all["blockInFront"]["name"]
        )

        # update the gameMap
        gameMap = np.zeros(
            (30, 30, 1),
            dtype=np.uint8,
        )
        for location, game_block in sense_all["map"].items():
            location = [int(i) for i in location.split(",")]
            pos_x = location[0]
            pos_z = location[2]
            if pos_x == 0 or pos_x == 31 or pos_z == 0 or pos_z == 31:
                continue
            gameMap[pos_x - 1][pos_z - 1][0] = self.decoder.decode_block_type(
                game_block["name"]
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

        # location in map
        pos_x = sense_all["player"]["pos"][0]
        pos_z = sense_all["player"]["pos"][2]
        position = (pos_x - 1) + ((pos_z - 1) * 30)
        self._state["position"][0] = position

        # update the reward, binary reward - achieved the goal or not
        self.reward = (
            int(sense_all["goal"]["goalAchieved"]) if "goal" in sense_all else 0
        )
        self.collected_reward += self.reward

        self.state = flatten(self._observation_space, self._state)

        return self.reward
