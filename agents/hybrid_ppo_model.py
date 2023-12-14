import torch as th
from gym import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from sb3_contrib.common.maskable.buffers import (
    MaskableDictRolloutBuffer,
    MaskableRolloutBuffer,
)
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported

from sb3_contrib.ppo_mask import MaskablePPO
import numpy as np


class HybridPPO(MaskablePPO):
    """
    Hybrid PPO and NSAM model with action masking support.
    The agent first attempts to construct a plan based on its current learned action model.
    If successful, it executes the generated plan in the environment.
    Else PPO will engage with the environment.
    """

    def __init__(self, exploring_sam=None, use_fluents_map=False, **kwargs):
        # MaskablePPO
        super().__init__(**kwargs)

        self.exploring_sam = exploring_sam
        self.plan_exist = False
        self.use_fluents_map = use_fluents_map

    def update_explorer_problem(self, problem):
        self.exploring_sam.update_problem(problem)
        self.plan_exist = False

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        This method is largely identical to the implementation found in the parent class.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :param use_masking: Whether or not to use invalid action masks during training
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """

        assert isinstance(
            rollout_buffer, (MaskableRolloutBuffer, MaskableDictRolloutBuffer)
        ), "RolloutBuffer doesn't support action masking"
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)
        n_steps = 0
        action_masks = None
        rollout_buffer.reset()

        if use_masking and not is_masking_supported(env):
            raise ValueError(
                "Environment does not support action masking. Consider using ActionMasker wrapper"
            )

        callback.on_rollout_start()
        dones = np.array([False])
        while n_steps < n_rollout_steps:
            # Learn action model and find a plan
            if not self.plan_exist and (dones[0] or n_steps == 0):
                self.plan_exist = self.exploring_sam.update_fixed_explorer(
                    use_fluents_map=self.use_fluents_map, env_is_reset=True
                )

                plan_log = f"{self.exploring_sam.output_dir}/did_plan.txt"
                with open(plan_log, "a") as f:
                    if self.exploring_sam.error_flag == -1:
                        f.write(f"error\n")
                    elif self.exploring_sam.error_flag == 0:
                        f.write(f"plan found\n")
                    elif self.exploring_sam.error_flag == 1:
                        f.write(f"no solution\n")
                    elif self.exploring_sam.error_flag == 2:
                        f.write(f"timeout\n")

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)

                # If we find a plan, we execute it else we use PPO for next action
                if self.plan_exist:
                    action = self.exploring_sam.choose_action()  # next action
                    action_masks = [False] * env.action_space.n
                    action_masks[action] = True
                    action_masks = np.array(action_masks)
                elif use_masking:
                    # This is the only change related to invalid action masking
                    action_masks = get_action_masks(env)

                actions, values, log_probs = self.policy(
                    obs_tensor, action_masks=action_masks
                )

            actions = actions.cpu().numpy()

            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                action_masks=action_masks,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            # Masking is not needed here, the choice of action doesn't matter.
            # We only want the value of the current observation.
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True
