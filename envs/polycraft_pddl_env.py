from gym import Env
from gym.spaces import Box, MultiDiscrete
from gym.spaces import Dict as GymDict
from gym.spaces import flatten_space, flatten
import numpy as np
from collections import OrderedDict
from utils import ActionsDecoder, SingleActionDecoder
import config as CONFIG
from typing import Union, List
from pathlib import Path
import copy
import os

from math import sqrt

from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser
from pddl_plus_parser.models import State

from pddl_plus_parser.exporters.numeric_trajectory_exporter import TrajectoryExporter

from pddl_plus_parser.lisp_parsers.pddl_tokenizer import PDDLTokenizer


class PolycraftGymEnv(Env):
    """
    Create gym environment for polycraft
    Where pal_path must be updated in the config.py file to work

    args:
        visually: if True, the environment will be displayed in the screen
        start_pal: if True, the pal will be started
        keep_alive: if True, the pal will be kept alive after the environment is closed
        max_steps: actions in the environment until reset
    """

    def __init__(
        self,
        visually: bool = False,
        start_pal: bool = True,
        keep_alive: bool = False,
        max_steps: int = 32,
        decoder: ActionsDecoder = SingleActionDecoder(),
        debug_pal: bool = False,
    ):

        self._pal_path = CONFIG.PAL_PATH
        self._visually = visually
        self._next_line = ""
        self.pal_owner = start_pal
        self._keep_alive = keep_alive

        self._domain_path = CONFIG.DEFUALT_DOMAIN_PATH

        # openai gym environment
        super().__init__()

        self.decoder = decoder

        self._observation_space = GymDict(
            {
                "blockInFront": Box(
                    low=0,
                    high=self.decoder.get_blocks_size(),
                    shape=(1,),
                    dtype=np.uint8,
                ),  # 11
                "gameMap": Box(
                    low=0,
                    high=self.decoder.get_blocks_size(),  # 11
                    shape=(32 * 32 * 2,),
                    dtype=np.uint8,
                ),  # map (32*32) and for each point (block) show name and isAccessible (*2)
                "inventory": Box(
                    low=0,
                    high=self.decoder.get_items_size(),  # 18
                    shape=(9 * 2,),
                    dtype=np.uint8,
                ),  # 1 line of inventory (9) and for each item show name and count (*2)
                "pos": Box(
                    low=0,
                    high=32,
                    shape=(2,),
                    dtype=np.uint8,
                ),  # map size (32*32), without y (up down movement)
                "facing": Box(
                    low=0,
                    high=4,
                    shape=(1,),
                    dtype=np.uint8,
                ),  # 0: north, 1: east, 2: south, 3: west
            }
        )
        self.observation_space = flatten_space(self._observation_space)

        self.action_space = MultiDiscrete(
            [self.decoder.get_actions_size(), 32, 32]
        )  # 15, 32, 32 -> action, x_pos, z_pos

        # current state start with all zeros
        self._state = OrderedDict(
            {
                "blockInFront": np.zeros(
                    (1,),
                    dtype=np.uint8,
                ),
                "gameMap": np.zeros(
                    (32 * 32 * 2,),
                    dtype=np.uint8,
                ),
                "inventory": np.zeros(
                    (9 * 2,),
                    dtype=np.uint8,
                ),
                "pos": np.zeros(
                    (2,),
                    dtype=np.uint8,
                ),
                "facing": np.zeros(
                    (1,),
                    dtype=np.uint8,
                ),
            }
        )
        self.state = flatten(self._observation_space, self._state)
        self.last_state = None

        self.collected_reward = 0

        self.action = None
        self.done = False
        self.reward = 0

        # no. of steps
        self.max_steps = max_steps
        self.steps_left = max_steps

    def decode_action_type(self, action: int) -> List[str]:
        return self.decoder.decode_to_planning(action)

    def step(self, action: int) -> tuple:
        info = {}
        self.steps_left -= 1

        if self.decoder.decode_action_type(action, self._state)[0] != "NOP":

            self.action = self.decode_action_type(action)

            triplet = self.exp.create_single_triplet(
                self.previous_state,
                f"({self.action})",
                self.problem.objects,
            )
            self.previous_state = triplet.next_state

            self._sense_all()

        # uncomment to check if action is valid
        # info["success"] = any(
        #     not np.array_equal(self._state[key], self.last_state[key])
        #     for key in self._state
        # )

        done = self.is_game_over()

        return self.state, float(self.reward), done, info

    def is_game_over(self) -> bool:
        done = (self.reward == 1) or (self.steps_left == 0)
        self.done = done
        return done

    def move_to_start(self) -> None:
        pass

    def reset(self) -> np.ndarray:
        # save the last state
        self.last_state = copy.deepcopy(self._state)
        # self.state = self.state.fill(0)

        # reset the environment
        self.previous_state = State(
            predicates=self.problem.initial_state_predicates,
            fluents=self.problem.initial_state_fluents,
            is_init=True,
        )

        # reset the teleport according to the new domain
        self.decoder.update_tp(self._senses())

        # reset the state
        self.move_to_start()
        self.collected_reward = 0
        self.action = None
        self.done = False
        self._sense_all()
        self.steps_left = self.max_steps
        return self.state

    def set_max_steps(self, steps: int) -> None:
        self.max_steps = steps
        self.steps_left = steps

    def render(self) -> None:
        print(f"Steps Left: {self.steps_left}")
        print(f"Action: {self.action}")
        # print(f"State: {self.state}")
        print(f"Reward: {self.reward}")
        print(f"Total Reward : {self.collected_reward}")
        print(
            "============================================================================="
        )

    def set_domain(self, path: str) -> None:
        # path must be absolute
        path = str(Path(path).absolute())

        # path must be json file and have json2 with the same name
        if not os.path.exists(path) or not os.path.exists(path + "2"):
            raise Exception(f"Domain file not found (path: {path})")

        self._domain_path = path

        # Split the original string
        parts = path.split("/")
        filename = parts[-1].split("_")  # Split by underscore
        filename[0] = "advanced_map"  # Replace 'map' with 'advanced_map'
        lst = filename[-1].split(".")  # Replace 'json' with 'pddl'
        filename[-1] = lst[0] + ".pddl"
        new_filename = "_".join(filename)

        domain_file_path = str(
            Path("planning/advanced_minecraft_domain.pddl").absolute()
        )
        domain = DomainParser(domain_file_path).parse_domain()
        problem_file_path = "/".join(parts[:-1]) + "/" + new_filename
        self.problem = ProblemParser(
            problem_path=problem_file_path, domain=domain
        ).parse_problem()

        self.exp = TrajectoryExporter(domain)
        self.parser = TrajectoryParser(domain)

    def _senses(self) -> dict:
        """Sense the environment - return the state"""

        def serialize_numeric_fluents(previous_state) -> str:
            """Serialize the numeric fluents of the state.

            :return: the string representing the assigned grounded fluents.
            """
            return "\n".join(
                fluent.state_representation
                for fluent in previous_state.state_fluents.values()
            )

        def serialize_predicates(previous_state) -> str:
            """Serialize the predicates the constitute the state's facts.

            :return: the string representation of the state's facts.
            """
            predicates_str = ""
            for grounded_predicates in previous_state.state_predicates.values():
                predicates_str += "\n"
                predicates_str += " ".join(
                    predicate.untyped_representation
                    for predicate in grounded_predicates
                )

            return predicates_str

        serialize = serialize_numeric_fluents(
            self.previous_state
        ) + serialize_predicates(self.previous_state)
        return self.reverse_translate(serialize.split("\n"))
        # return self.reverse_translate(self.previous_state.serialize())

    def _sense_all(self) -> None:
        """Sense the environment - update the state and get reward"""
        self.reward = 0

        # get the state from the Polycraft server
        sense_all = self._senses()

        inventory_before = self._state["inventory"].copy()

        self._state["blockInFront"][0] = self.decoder.decode_block_type(
            sense_all["blockInFront"]["name"]
        )

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
        self._state["inventory"] = (
            inventory.ravel()
        )  # flatten the inventory to 1D vector

        # location in map without y (up down movement)
        self._state["pos"][0] = sense_all["player"]["pos"][0]
        self._state["pos"][1] = sense_all["player"]["pos"][2]

        # facing
        self._state["facing"][0] = self.decoder.directions_decoder[
            sense_all["player"]["facing"]
        ]

        # update the gameMap
        gameMap = np.zeros(
            (32, 32, 2),
            dtype=np.uint8,
        )
        for location, game_block in sense_all["map"].items():
            location = [int(i) for i in location.split(",")]
            gameMap[location[0]][location[2]][0] = self.decoder.decode_block_type(
                game_block["name"]
            )
            gameMap[location[0]][location[2]][1] = int(game_block["isAccessible"])
        self._state["gameMap"] = gameMap.ravel()  # flatten the map to 1D vector

        inventory_after = self._state["inventory"].copy()

        # update the reward - reward bigger when item is more rare
        reward = 0
        change = [int(inventory_after[i]) - int(inventory_before[i]) for i in range(6)]
        if change[0] > 0:  # log
            reward += 0.002
        if change[1] > 0:  # planks
            reward += 0.004
        if change[2] > 0:  # sticks
            reward += 0.004
        if change[3] == 1 and inventory_after[3] == 1:  # first time get rubber
            reward += 0.1
        if change[4] > 0:  # tree tap
            reward += 0.1
        if change[5] > 0:  # wooden pogo
            reward += 1

        self.reward = reward
        self.collected_reward += self.reward

        self.state = flatten(self._observation_space, self._state)

        return self.reward

    def reverse_translate(self, pddl_list):
        state = {}
        game_map = []
        inventory = {5: {"item": "polycraft:wooden_pogo_stick", "count": 0}}
        position = None
        tree_count = 0
        goalAchieved = False

        crafting_table_cell = None
        get_int = lambda x: int(float(x.split()[-1][:-1]))
        for line in pddl_list:
            line = line.strip()
            if line.startswith("(= (count_log_in_inventory"):
                inventory[0] = {"item": "minecraft:log", "count": get_int(line)}
            elif line.startswith("(= (count_planks_in_inventory"):
                inventory[1] = {"item": "minecraft:planks", "count": get_int(line)}
            elif line.startswith("(= (count_stick_in_inventory"):
                inventory[2] = {"item": "minecraft:stick", "count": get_int(line)}
            elif line.startswith("(= (count_sack_polyisoprene_pellets_in_inventory"):
                inventory[3] = {
                    "item": "polycraft:sack_polyisoprene_pellets",
                    "count": get_int(line),
                }
            elif line.startswith("(= (count_tree_tap_in_inventory"):
                inventory[4] = {"item": "polycraft:tree_tap", "count": get_int(line)}
            elif "have_pogo_stick" in line:
                inventory[5]["count"] = 1
                goalAchieved = True
            elif line.startswith("(position"):
                position_string = line[len("(position") :].strip()
                if position_string.startswith("cell"):
                    position = int(position_string[len("cell") :].strip()[:-1])
                else:
                    position = "crafting_table"
            elif (
                line.startswith("(air_cell")
                or line.startswith("(tree_cell")
                or line.startswith("(crafting_table_cell")
            ):
                cells = line.split(") ")
                for cell in cells:
                    if cell[-1] == ")":
                        cell = cell[:-1]
                    if cell:
                        if cell.startswith("(air_cell cell"):
                            cell_num = int(cell[len("(air_cell cell") :].strip())
                            game_map.append((cell_num, 0))
                        elif cell.startswith("(tree_cell cell"):
                            cell_num = int(cell[len("(tree_cell cell") :].strip())
                            game_map.append((cell_num, 1))

        # Compute tree_count from the number of tree cells in game_map
        tree_count = sum(1 for _, value in game_map if value == 1)

        state["map"] = dict(game_map)
        cal_max = max(state["map"].keys())
        grid_size = sqrt(cal_max + 1)
        crafting_table_cell = cal_max * (cal_max + 1) / 2 - sum(state["map"].keys())
        if position == "crafting_table":
            position = crafting_table_cell
        game_map.append((crafting_table_cell, 2))

        state["map"] = dict(
            [
                (f"{int(number % grid_size)+1},4,{int(number // grid_size)+1}", value)
                for number, value in game_map
            ]
        )

        for key, value in state["map"].items():
            if value == 0:
                state["map"][key] = {"name": "minecraft:air"}
            elif value == 1:
                state["map"][key] = {"name": "minecraft:log"}
            elif value == 2:
                state["map"][key] = {"name": "minecraft:crafting_table"}
            elif value == 3:
                state["map"][key] = {"name": "minecraft:bedrock"}

        state["inventory"] = inventory
        state["player"] = {
            "pos": [int(position % grid_size) + 1, 4, int(position // grid_size) + 1]
        }  # [position] if position is not None else [0]
        state["treeCount"] = [tree_count]
        state["goal"] = {"goalAchieved": goalAchieved}
        state["blockInFront"] = {
            "name": state["map"][",".join(map(str, state["player"]["pos"]))]["name"]
        }

        return state
