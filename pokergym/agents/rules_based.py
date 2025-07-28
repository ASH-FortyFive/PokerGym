from typing import Any, Union
from pokergym.agents.base import Agent
from pokergym.env.enums import Action

from random import Random

class RulesBased(Agent):
    def __init__(self, idx, action_space=None, observation_space=None):
        super().__init__(idx, action_space, observation_space)
        raise NotImplementedError("RulesBased agent is not implemented yet.")

    def act(self, observation) -> Union[Action, Any]:
        """
        Randomly selects an action from the available action space.
        """
        raise NotImplementedError("RulesBased agent is not implemented yet.")