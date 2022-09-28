from Agents.PolycraftAgent import PolycraftAgent
import json


class LearningAgent(PolycraftAgent):
    """Agent that record the state and action the delegate agent takes."""

    def __init__(self, agent):
        super().__init__()
        self._agent = agent
        self._record = {}
        self._no_action = 0

    # overriding abstract method
    def choose_action(self, state):
        """Return the action the delegate agent takes."""
        return self._agent.choose_action(state)

    def do(self, state):
        """Choose an action, save it and then send it to the Polycraft server."""
        self._store_state(state)
        action = super().do(state)
        self._store_action(action)
        return action

    def _store_state(self, state):
        """Store the agent state."""
        self._record[self._no_action] = {"state": state}

    def _store_action(self, action):
        """Store the agent's action corresponds to his state."""
        self._record[self._no_action]["action"] = action
        self._no_action += 1

    def export_trajectory(self, filename="expert_trajectory.json"):
        """Export the trajectory to a JSON file."""
        with open(filename, "w") as fp:
            json.dump(self._record, fp)
