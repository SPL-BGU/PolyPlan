from agents.polycraft_agent import PolycraftAgent


class FixedScriptAgent(PolycraftAgent):
    """Agent that follows a fixed script."""

    def __init__(self, env, filename: str = None, script: list = []):
        super().__init__(env)
        if filename:
            file = open(filename, "r")
            self._actions_list = file.read().split("\n")
        else:
            self._actions_list = script

        self._current_action = 0

    # overriding abstract method
    def choose_action(self, state=None) -> int:
        """Ignore the state and return the next action in the script"""
        action = self._next_action()
        return self.env.decoder.encode_action_type(action)

    def _next_action(self) -> str:
        """Return the next action in the script."""

        if self._current_action >= len(self._actions_list):
            self.reset_script()

        command = self._actions_list[self._current_action]
        self._current_action += 1

        return command

    def reset_script(self):
        """Reset the script to the beginning"""
        self._current_action = 0
