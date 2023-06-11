from agents.polycraft_agent import PolycraftAgent
from numpy import ndarray
from numpy.random import choice as random_choice


class SmartAgent(PolycraftAgent):
    """Wrapper of pre trained model."""

    def __init__(self, env, model):
        super().__init__(env)
        self.model = model

    # overriding abstract method
    def choose_action(self, state) -> int:
        """Predict the next action"""
        predictions, _ = self.model.predict(state)
        if type(predictions) is ndarray:
            return random_choice(predictions)
        return predictions

    def predict(self, observations, state, episode_start, deterministic) -> tuple:
        """Wrapper for gym predict function."""
        return self.model.predict(observations, state, episode_start, deterministic)
