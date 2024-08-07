from stable_baselines3.common.policies import ActorCriticPolicy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.dqn.policies import DQNPolicy

import torch as th
import numpy as np
from typing import Optional, Tuple


class PolycraftPPOPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        # pi is the policy network
        # vf is the value network
        net_arch = dict(pi=[512, 256, 256], vf=[512, 256, 256])
        super().__init__(
            observation_space, action_space, lr_schedule, net_arch=net_arch, **kwargs
        )


class PolycraftMaskedPPOPolicy(MaskableActorCriticPolicy):
    # def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
    #     # pi is the policy network
    #     # vf is the value network
    #     net_arch = dict(pi=[512, 256, 256], vf=[512, 256, 256])
    #     super().__init__(
    #         observation_space, action_space, lr_schedule, net_arch=net_arch, **kwargs
    #     )

    def forward_planner(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param actions: Which action to take
        :param action_masks: Action masks to apply to the action distribution
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob


class PolycraftDQNPolicy(DQNPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        net_arch = [512, 256, 256]
        super().__init__(
            observation_space, action_space, lr_schedule, net_arch=net_arch, **kwargs
        )
