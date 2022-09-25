from Agents.PolycraftAgent import PolycraftAgent


class FixedScriptAgent(PolycraftAgent):
    """Agent that follows a fixed script."""

    def __init__(self, filename, host="127.0.0.1", port=9000):
        super().__init__(host, port)
        self._file = open(filename, "r")

    # overriding abstract method
    def choose_action(self, state):
        """Ignore the state and return the next action in the script"""
        return self._read_script()

    def _read_script(self):
        """Return the next action in the script."""
        command = self._file.readline()

        # if finished read the file then return NOP command
        if not command:
            return "NOP"

        return command[:-1]  # remove the newline character
