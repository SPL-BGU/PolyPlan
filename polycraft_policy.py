from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.dqn.policies import DQNPolicy


class PolycraftPPOPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        # pi is the policy network
        # vf is the value network
        net_arch = [dict(pi=[512, 256, 256], vf=[512, 256, 256])]
        super().__init__(
            observation_space, action_space, lr_schedule, net_arch=net_arch, **kwargs
        )


class PolycraftDQNPolicy(DQNPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        net_arch = [512, 256, 256]
        super().__init__(
            observation_space, action_space, lr_schedule, net_arch=net_arch, **kwargs
        )
