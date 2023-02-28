from agents.polycraft_agent import PolycraftAgent
import pickle, json
from imitation.data import rollout
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data.wrappers import RolloutInfoWrapper


class LearningAgent(PolycraftAgent):
    """Agent that record the state and action the delegate agent takes."""

    def __init__(self, env, agent, for_planning: bool = False):
        super().__init__(env)
        self.env = env
        self._agent = agent
        self.for_planning = for_planning

        self._rollouts = None
        self._record = []

    # overriding abstract method
    def choose_action(self, state) -> int:
        """Return the action the delegate agent takes."""
        return self._agent.choose_action(state)

    def record_trajectory(self, steps: int = 64) -> None:
        """Record the trajectory."""
        if not self.for_planning:  # record trajectory for behavioral cloning
            venv = DummyVecEnv([lambda: RolloutInfoWrapper(self.env)])
            steps = max(steps, 64)

            # create expert policy
            self._rollouts = rollout.rollout(
                lambda state: [self.choose_action(state)],
                venv,
                rollout.make_sample_until(min_timesteps=steps, min_episodes=None),
            )
        else:  # record trajectory for planning algorithms
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                self._record.append(self.env.decoder.decode_to_planning(action))
                state, _, done, _ = self.env.step(action)

    def export_trajectory(self, filename: str = "expert_trajectory.pkl") -> None:
        """Export the trajectory to a file."""
        if not self.for_planning:
            with open(filename, "wb") as fp:
                pickle.dump(self._rollouts, fp)
        else:
            with open(filename, "w") as fp:
                for action in self._record:
                    fp.write(f"({action})\n")
