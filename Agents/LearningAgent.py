from Agents.PolycraftAgent import PolycraftAgent
import json


class LearningAgent(PolycraftAgent):
    """Agent that record the state and action the delegate agent takes."""

    def __init__(self, agent):
        self._agent = agent
        agent.close_connection()
        super().__init__(agent.host, agent.port)
        self._record = {}
        self._no_action = 0

    # overriding abstract method
    def choose_action(self, state):
        """Return the action the delegate agent takes."""
        return self._agent.choose_action(state)

    def do(self, state):
        """Choose an action, save it and then send it to the Polycraft server."""
        action = super().do(state)
        self._record[self._no_action] = {"state": state, "action": action}
        # TODO: state = before ?
        # TODO: add "after" state
        self._no_action += 1
        return action

    def _export_trajectory(self, filename="learning_agent.json"):
        """Export the trajectory to a JSON file."""
        with open(filename, "w") as fp:
            json.dump(self._record, fp)
