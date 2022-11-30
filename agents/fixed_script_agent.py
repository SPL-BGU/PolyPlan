from agents.polycraft_agent import PolycraftAgent


class FixedScriptAgent(PolycraftAgent):
    """Agent that follows a fixed script."""

    def __init__(self, env, filename: str):
        super().__init__(env)
        self._file = open(filename, "r")

    # overriding abstract method
    def choose_action(self, state) -> str:
        """Ignore the state and return the next action in the script"""
        return self._read_script()

    def _read_script(self) -> str:
        """Return the next action in the script."""
        command = self._file.readline()

        # if finished read the file again
        if not command:
            self._file.seek(0)
            command = self._file.readline()

        return command[:-1]  # remove the newline character
