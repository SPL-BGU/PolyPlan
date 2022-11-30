from agents.polycraft_agent import PolycraftAgent
from utils.decoder import Decoder
import pickle
from imitation.data import rollout
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data.wrappers import RolloutInfoWrapper


class LearningAgent(PolycraftAgent):
    """Agent that record the state and action the delegate agent takes."""

    def __init__(self, env, agent):
        super().__init__(env)
        self.env = env
        self._agent = agent
        self._rollouts = None

    # overriding abstract method
    def choose_action(self, state) -> str:
        """Return the action the delegate agent takes."""
        return self._agent.choose_action(state)

    def expert(self, state) -> str:
        action = self._agent.choose_action(state)
        return [Decoder.encode_action_type(action)]

    def record_trajectory(self, steps: int = 64) -> None:
        """Record the trajectory."""
        venv = DummyVecEnv([lambda: RolloutInfoWrapper(self.env)])
        steps = max(steps, 64)

        # create expert policy
        self._rollouts = rollout.rollout(
            self.expert,
            venv,
            rollout.make_sample_until(min_timesteps=64, min_episodes=None),
        )

    def export_trajectory(self, filename: str = "expert_trajectory.pkl") -> None:
        """Export the trajectory to a file."""
        with open(filename, "wb") as fp:
            pickle.dump(self._rollouts, fp)
