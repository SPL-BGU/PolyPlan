from envs import AdvancedMinecraft
from agents.explore_only_legal import ExploreOnlyLegal


class MaskedMinecraft(AdvancedMinecraft):
    """
    Wrapper to any PolycraftGymEnv that add invalid action masking to the environment

    args:
        visually: if True, the environment will be displayed in the screen
        start_pal: if True, the pal will be started
        keep_alive: if True, the pal will be kept alive after the environment is closed
        max_steps: actions in the environment until reset
    """

    def __init__(self, **kwargs):
        # PolycraftGymEnv
        super().__init__(**kwargs)

        self.mask = ExploreOnlyLegal(self)

    def step(self, action):
        self.mask.choose_action(str(self.state))
        self.mask.last_action = action
        return super().step(action)

    def action_masks(self):
        """Return a mask of valid actions for the current state"""
        self.mask.check_state_exist(str(self.state))
        mask = [bool(x) for x in list(self.mask.graph.loc[str(self.state)])]
        return mask
