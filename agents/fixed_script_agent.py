from agents.polycraft_agent import PolycraftAgent


class FixedScriptAgent(PolycraftAgent):
    """Agent that follows a fixed script"""

    def __init__(
        self,
        env,
        filename: str = None,
        script: list = [],
        human_readable: bool = False,
        env_is_reset: bool = False,
    ):
        super().__init__(env)

        # you must reset the env before init this agent
        if not env_is_reset:
            env.reset()

        if filename:
            file = open(filename, "r")
            self._actions_list = file.read().split("\n")
        else:
            self._actions_list = script

        if human_readable:
            self._actions_list = [
                self.env.decoder.encode_human_action_type(action)
                for action in self._actions_list
                if action != ""
            ]
        else:
            self._actions_list = [
                self.env.decoder.encode_planning_action_type(action)
                for action in self._actions_list
                if action != ""
            ]

        if len(self._actions_list) == 0:
            raise Exception("Script is empty")

        self._current_action = 0

    # overriding abstract method
    def choose_action(self, state=None) -> int:
        """Ignore the state and return the next action in the script"""
        action = self._next_action()
        return action

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

    @property
    def length(self) -> int:
        return len(self._actions_list)
