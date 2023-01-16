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
        self._record = {}
        self._no_action = 0

    # overriding abstract method
    def choose_action(self, state) -> str:
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
            self._record[self._no_action] = {"state": state.tolist()}
            self._no_action += 1
            while not done:
                action = self.choose_action(state)
                self._store_action(action)
                state, _, done, _ = self.env.step(action)
                self._store_state(state)

    def _store_state(self, state):
        """Store the agent state."""
        self._record[self._no_action]["state"] = state.tolist()
        self._no_action += 1

    def _store_action(self, action):
        """Store the agent's action corresponds to his state."""
        self._record[self._no_action] = {"action": action}

    def export_trajectory(self, filename: str = "expert_trajectory.pkl") -> None:
        """Export the trajectory to a file."""
        if not self.for_planning:
            with open(filename, "wb") as fp:
                pickle.dump(self._rollouts, fp)
        else:
            with open(filename, "w") as fp:
                json.dump(self._record, fp)
