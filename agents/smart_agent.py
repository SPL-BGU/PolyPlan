from agents.polycraft_agent import PolycraftAgent


class SmartAgent(PolycraftAgent):
    """Wrapper of pre trained model."""

    def __init__(self, env, model):
        super().__init__(env)
        self.model = model

    # overriding abstract method
    def choose_action(self, state) -> int:
        """Predict the next action"""
        return self.model.predict(state)[0]
