from abc import ABC, abstractmethod
from typing import Any, Union

from pokergym.env.enums import Action


class Agent(ABC):
    def __init__(self, idx, action_space=None, observation_space=None):
        self.idx = idx
        self.action_space = action_space
        self.observation_space = observation_space

    @abstractmethod
    def act(self, observation) -> Union[Action, Any]:
        """
        Implement the agent's action logic here.
        This is a placeholder method and should be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
