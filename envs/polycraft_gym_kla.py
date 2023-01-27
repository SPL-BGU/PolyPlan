import gym
from envs import PolycraftGymEnv
from agents.fixed_script_agent import FixedScriptAgent


class PolycraftGymKLA(gym.Wrapper):
    """
    Create wrapper for an PolycraftGymEnv environment that will allow to train a agent that learn to play the last k actions
    Where pal_path must be updated in the config.py file to work

    args:
        expert: expert agent that solve the game
        k: k last actions to learn
    """

    def __init__(
        self, env: PolycraftGymEnv, k: int = 1, expert_actions: int = 11, **kwargs
    ):
        # PolycraftGymEnv
        self.env = env(**kwargs)
        super().__init__(self.env)
        self.env.max_rounds = expert_actions
        self.env.rounds_left = expert_actions

        self.expert = FixedScriptAgent(self, "my_script.txt")
        self.k = expert_actions - k

    def reset(self):
        # print(f"reward {self.reward}")

        # reset the environment
        super().reset()

        # do k actions
        for _ in range(self.k):
            self.expert.act()
        self.expert.reset_script()

        return self.env.state
