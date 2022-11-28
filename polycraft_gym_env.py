from gym import Env
from gym.spaces import Box, Discrete, MultiDiscrete
from gym.spaces import Dict as GymDict
from gym.spaces import flatten_space, flatten
import sys, time, queue, subprocess, threading
import numpy as np
from collections import OrderedDict
from utils.server_controller import ServerController
from utils.decoder import Decoder
import config as CONFIG


class PolycraftGymEnv(Env):
    """
    Create gym environment for polycraft
    Where pal_path and domain_path must be updated in the config.py file to work

    args:
        visually: if True, the environment will be displayed in the screen
        start_pal: if True, the pal will be started
        keep_alive: if True, the pal will be kept alive after the environment is closed
    """

    def __init__(
        self,
        visually=False,
        start_pal=True,
        keep_alive=False,
        rounds=64,
    ):

        # start polycraft server
        self._q = queue.Queue()

        self._pal_path = CONFIG.PAL_PATH
        self._visually = visually
        self._next_line = ""
        if start_pal:
            self._start_pal()  # time to start pal is 35 seconds
            while "Minecraft finished loading" not in str(self._next_line):
                self._next_line = self._check_queues()
        self.pal_owner = start_pal
        self._keep_alive = keep_alive

        # init socket connection to polycraft
        self.server_controller = ServerController()
        self.server_controller.open_connection()

        if start_pal:
            self.server_controller.send_command(
                "START"
            )  # time to reset the environment is 10 seconds

        self._domain_path = CONFIG.DEFUALT_DOMAIN_PATH

        # openai gym environment
        super().__init__()

        self._observation_space = GymDict(
            {
                "blockInFront": Discrete(Decoder.get_blocks_size()),
                "gameMap": Box(
                    low=0,
                    high=Decoder.get_blocks_size(),
                    shape=(32 * 32 * 2,),
                    dtype=np.uint8,
                ),  # map (32*32*2)and for each point (block) show name and isAccessible (2) as 1D vector
                "goalAchieved": Discrete(2),
                "inventory": Box(
                    low=0,
                    high=Decoder.get_items_size(),
                    shape=(2 * 9,),
                    dtype=np.uint8,
                ),  # 1 line of inventory (1*9) and for each item show name and count (2) as 1D vector
                "pos": MultiDiscrete(
                    [32, 32]
                ),  # map size + 1 to all, without y (up down movement)
            }
        )
        self.observation_space = flatten_space(self._observation_space)

        self.action_space = Discrete(Decoder.get_actions_size())

        # current state start with all zeros
        self._state = OrderedDict(
            {
                "blockInFront": 0,
                "gameMap": np.zeros(
                    (32 * 32 * 2,),
                    dtype=np.uint8,
                ),
                "goalAchieved": 0,
                "inventory": np.zeros(
                    (2 * 9,),
                    dtype=np.uint8,
                ),
                "pos": np.array([0, 0]),
            }
        )
        self.state = flatten(self._observation_space, self._state)

        self.collected_reward = 0

        self.action = None
        self.done = False
        self.reward = 0

        # no. of rounds
        self.max_rounds = rounds
        self.rounds = rounds

    def step(self, action):
        info = {}
        self.rounds -= 1

        command = Decoder.decode_action_type(action)
        self.action = command

        self.server_controller.send_command(command)

        self._sense_all()  # update the state and get reward

        done = self.is_game_over()

        return self.state, float(self.reward), done, info

    def is_game_over(self):
        done = False

        if self.rounds == 0:
            done = True
        else:
            done = bool(self._state["goalAchieved"])

        self.done = done
        return done

    def reset(self):

        # reset the environment
        self.server_controller.send_command(f"RESET domain {self._domain_path}")
        if self.pal_owner:
            while "game initialization completed" not in str(self._next_line):
                self._next_line = self._check_queues()
        time.sleep(2)

        # reset the teleport according to the new domain
        sense_all = self.server_controller.send_command("SENSE_ALL NONAV")
        Decoder.update_actions(sense_all)

        # reset the state
        self.collected_reward = 0
        self.action = None
        self.done = False
        self._sense_all()
        self.rounds = self.max_rounds
        return self.state

    def render(self):
        print(f"Rounds Left: {self.rounds}")
        print(f"Action: {self.action}")
        # print(f"State: {self.state}")
        print(f"Reward: {self.reward}")
        print(f"Total Reward : {self.collected_reward}")
        print(
            "============================================================================="
        )

    def close(self):
        """Close the environment"""
        if not self._keep_alive:
            self.server_controller.send_command("RESET")
        self.server_controller.close_connection()
        return super().close()

    def set_domain(self, path):
        self._domain_path = path

    def _sense_all(self):
        """Sense the environment - update the state and get reward"""
        self.reward = 0

        # get the state from the Polycraft server
        sense_all = self.server_controller.send_command("SENSE_ALL NONAV")

        self._state["blockInFront"] = Decoder.decode_block_type(
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
            # inventory[0][location] = location
            inventory[0][location] = Decoder.decode_item_type(item["item"])
            inventory[1][location] = item["count"]
        self._state[
            "inventory"
        ] = inventory.ravel()  # flatten the inventory to 1D vector

        # location in map without y (up down movement)
        self._state["pos"][0] = sense_all["player"]["pos"][0]
        self._state["pos"][1] = sense_all["player"]["pos"][2]

        # update the gameMap
        gameMap = np.zeros(
            (32, 32, 2),
            dtype=np.uint8,
        )
        for location, game_block in sense_all["map"].items():
            location = [int(i) for i in location.split(",")]
            gameMap[location[0]][location[2]][0] = Decoder.decode_block_type(
                game_block["name"]
            )
            gameMap[location[0]][location[2]][1] = int(game_block["isAccessible"])
        self._state["gameMap"] = gameMap.ravel()  # flatten the map to 1D vector
        self._state["goalAchieved"] = (
            int(sense_all["goal"]["goalAchieved"]) if "goal" in sense_all else 0
        )

        # update the reward
        self.reward = int(
            self._state["goalAchieved"]
        )  # binary reward - achieved the goal or not
        self.collected_reward += self.reward

        self.state = flatten(self._observation_space, self._state)

        return self.reward

    def _start_pal(self):
        """Launch Minecraft Client"""
        if self._visually:
            pal_process_cmd = "./gradlew runclient"
        else:
            pal_process_cmd = "xvfb-run -s '-screen 0 1280x1024x24' ./gradlew --no-daemon --stacktrace runclient"
        print(("PAL command: " + pal_process_cmd))
        pal_client_process = subprocess.Popen(
            pal_process_cmd,
            shell=True,
            cwd=self._pal_path,
            stdout=subprocess.PIPE,
            # stdin=subprocess.PIPE,  # DN: 0606 Removed for perforamnce
            stderr=subprocess.STDOUT,  # DN: 0606 - pipe stderr to STDOUT. added for performance
            bufsize=1,  # DN: 0606 Added for buffer issues
            universal_newlines=True,  # DN: 0606 Added for performance - needed for bufsize=1 based on docs?
        )
        pal_client_thread = threading.Thread(
            target=PolycraftGymEnv._read_output, args=(pal_client_process, self._q)
        )
        pal_client_thread.daemon = True
        pal_client_thread.start()  # Kickoff the PAL Minecraft Client
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

    def _check_queues(self, check_all=False):
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
            sys.stdout.flush()
            sys.stderr.flush()
        except queue.Empty:
            pass

        return next_line
