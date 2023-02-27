from envs import PolycraftGymEnv
from gym.spaces import Box, Discrete
from gym.spaces import Dict as GymDict
from gym.spaces import flatten_space, flatten
import numpy as np
from collections import OrderedDict
from utils import MacroActionsDecoder


class BasicMinecraft(PolycraftGymEnv):
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
        super().__init__(**kwargs, decoder=MacroActionsDecoder())

        # basic minecraft environment observation space
        self._observation_space = GymDict(
            {
                "treeCount": Box(
                    low=0,
                    high=5,
                    shape=(1,),
                    dtype=np.uint8,
                ),  # count of trees in the map
                "inventory": Box(
                    low=0,
                    high=self.decoder.get_items_size(),  # 18
                    shape=(9 * 2,),
                    dtype=np.uint8,
                ),  # 1 line of inventory (9) and for each item show name and count (*2)
            }
        )
        self.observation_space = flatten_space(self._observation_space)

        self.action_space = Discrete(self.decoder.get_actions_size())  # 6

        # current state start with all zeros
        self._state = OrderedDict(
            {
                "treeCount": np.zeros(
                    (1,),
                    dtype=np.uint8,
                ),
                "inventory": np.zeros(
                    (9 * 2,),
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

        # update the treeCount
        count: int = 0
        for _, game_block in sense_all["map"].items():
            count += 1 if game_block["name"] == "minecraft:log" else 0
        self._state["treeCount"][0] = count

        # update the inventory
        inventory = np.zeros(
            (2, 9),
            dtype=np.uint8,
        )
        for location, item in sense_all["inventory"].items():
            if location == "selectedItem":
                continue
            location = int(location)
            inventory[0][location] = self.decoder.decode_item_type(item["item"])
            inventory[1][location] = item["count"]
        self._state[
            "inventory"
        ] = inventory.ravel()  # flatten the inventory to 1D vector

        # update the reward, binary reward - achieved the goal or not
        self.reward = (
            int(sense_all["goal"]["goalAchieved"]) if "goal" in sense_all else 0
        )
        self.collected_reward += self.reward

        self.state = flatten(self._observation_space, self._state)

        return self.reward
