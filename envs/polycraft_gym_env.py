from gym import Env
from gym.spaces import Box, MultiDiscrete
from gym.spaces import Dict as GymDict
from gym.spaces import flatten_space, flatten
import sys, time, queue, subprocess, threading
import numpy as np
from collections import OrderedDict
from utils import ServerController
from utils import ActionsDecoder, SingleActionDecoder
import config as CONFIG
from typing import Union, List
from pathlib import Path
import copy
import os


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
        # start polycraft server
        self._q = queue.Queue()

        port = 9000
        self.pal_client_process = None
        self.pal_client_thread = None

        self._pal_path = CONFIG.PAL_PATH
        self._visually = visually

        error_messages = [
            "FAILURE: Build failed with an exception",
            "xvfb-run: error: Xvfb failed to start",
            "API FAILED TO START WITH PORT: 9000",
        ]

        self._next_line = ""
        if start_pal:
            self._start_pal(port=port)  # time to start pal is 35 seconds
            while "Minecraft finished loading" not in str(self._next_line):
                self._next_line = self._check_queues(debug_pal)
                if any(msg in str(self._next_line) for msg in error_messages):
                    raise Exception

        self.pal_owner = start_pal
        self._keep_alive = keep_alive

        # init socket connection to polycraft
        self.server_controller = ServerController()
        self.server_controller.open_connection(port=port)

        if start_pal:
            self.server_controller.set_timeout(60.0)
            self.server_controller.send_command(
                "START"
            )  # time to reset the environment is 10 seconds
            self.server_controller.set_timeout()

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
        return self.decoder.decode_action_type(action, self._state)

    def step(self, action: int) -> tuple:
        info = {}
        self.steps_left -= 1

        command_list = self.decode_action_type(action)
        self.action = command_list

        for command in command_list:
            self.server_controller.send_command(command)

        self._sense_all()  # update the state and get reward

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

        # reset the environment
        self.server_controller.set_timeout(60.0)
        self.server_controller.send_command(f"RESET domain {self._domain_path}")
        self.server_controller.set_timeout()
        if self.pal_owner:
            while "game initialization completed" not in str(self._next_line):
                self._next_line = self._check_queues()
        self._next_line = ""

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

    def close(self) -> None:
        """Close the environment"""
        if not self._keep_alive:
            self.server_controller.send_command("RESET")
        self.server_controller.close_connection()
        return super().close()

    def set_domain(self, path: str) -> None:
        # path must be absolute
        path = str(Path(path).absolute())

        # path must be json file and have json2 with the same name
        if not os.path.exists(path) or not os.path.exists(path + "2"):
            raise Exception(f"Domain file not found (path: {path})")

        self._domain_path = path

    def _senses(self) -> dict:
        """Sense the environment - return the state"""

        while True:
            sense_all = self.server_controller.send_command("SENSE_ALL NONAV")

            # sanity check - check sense_all have all the keys
            if not all(
                item in sense_all
                for item in [
                    "blockInFront",
                    "inventory",
                    "map",
                    "player",
                ]
            ):
                continue

            if "minecraft:crafting_table" in [
                tup["name"] for tup in sense_all["map"].values()
            ]:
                break
            else:
                continue

        return sense_all

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

    def _start_pal(self, port: int = 9000):
        """Launch Minecraft Client"""
        if self._visually:
            pal_process_cmd = f"PAL_AGENT_PORT={port} ./gradlew runclient"
        else:
            pal_process_cmd = f"PAL_AGENT_PORT={port} xvfb-run -s '-screen 0 1280x1024x24' ./gradlew --no-daemon --stacktrace runclient"
        print(("PAL command: " + pal_process_cmd))
        self.pal_client_process = subprocess.Popen(
            pal_process_cmd,
            shell=True,
            cwd=self._pal_path,
            stdout=subprocess.PIPE,
            # stdin=subprocess.PIPE,  # DN: 0606 Removed for perforamnce
            stderr=subprocess.STDOUT,  # DN: 0606 - pipe stderr to STDOUT. added for performance
            bufsize=1,  # DN: 0606 Added for buffer issues
            universal_newlines=True,  # DN: 0606 Added for performance - needed for bufsize=1 based on docs?
        )
        self.pal_client_thread = threading.Thread(
            target=PolycraftGymEnv._read_output, args=(self.pal_client_process, self._q)
        )
        self.pal_client_thread.daemon = True
        self.pal_client_thread.start()  # Kickoff the PAL Minecraft Client
        print("PAL Client Initiated")

    def _read_output(pipe, q):
        """
        This is run on a separate daemon thread for both PAL and the AI Agent.

        This takes the STDOUT (and STDERR that gets piped to STDOUT from the Subprocess.POpen() command)
        and places it into a Queue object accessible by the main thread
        """
        # read both stdout and stderr

        flag_continue = True
        while flag_continue and not pipe.stdout.closed:
            try:
                l = pipe.stdout.readline()
                q.put(l)
                sys.stdout.flush()
                pipe.stdout.flush()
            except UnicodeDecodeError as e:
                print(f"Err: UnicodeDecodeError: {e}")
                try:
                    l = pipe.stdout.read().decode("utf-8")
                    q.put(l)
                    sys.stdout.flush()
                    pipe.stdout.flush()
                except Exception as e2:
                    print(f"ERROR - CANT HANDLE OUTPUT ENCODING! {e2}")
                    sys.stdout.flush()
                    pipe.stdout.flush()
            except Exception as e3:
                print(f"ERROR - UNKNOWN EXCEPTION! {e3}")
                sys.stdout.flush()
                pipe.stdout.flush()

    def _check_queues(self, debug_pal: bool = False):
        """
        Check the STDOUT queues in both the PAL and Agent threads, logging the responses appropriately
        :return: next_line containing the STDOUT of the PAL process only:
                    used to determine game ending conditions and update the score_dict{}
        """
        next_line = ""

        # # write output from procedure A (if there is any)
        # DN: Remove "blockInFront" data from PAL, as it just gunks up our PAL logs for no good reason.
        try:
            next_line = self._q.get(False, timeout=0.025)
            if debug_pal:
                print(next_line)
            sys.stdout.flush()
            sys.stderr.flush()
        except queue.Empty:
            pass

        return next_line
