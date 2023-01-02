from polycraft_gym_env import PolycraftGymEnv
from agents.fixed_script_agent import FixedScriptAgent


class PolycraftGymKLA(PolycraftGymEnv):
    """
    Create wrapper to PolycraftGymEnv environment that will allow to train a agent that learn to play the last k actions
    Where pal_path must be updated in the config.py file to work

    args:
        expert: expert agent that solve the game
        k: k last actions to learn
    """

    def __init__(self, k: int = 1, rounds: int = 13, **kwargs):
        # PolycraftGymEnv
        super().__init__(**kwargs)
        self.max_rounds = rounds
        self.rounds_left = rounds

        self.expert = FixedScriptAgent(self, "my_script.txt")
        self.k = rounds - k

    def reset(self):
        # print(f"reward {self.reward}")

        # reset the environment
        super().reset()

        # do k actions
        for _ in range(self.k):
            self.expert.act()
        self.expert.reset_script()

        return self.state
