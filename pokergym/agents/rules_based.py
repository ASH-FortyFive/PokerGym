from typing import Any, Union
from pokergym.agents.base import Agent
from pokergym.env.enums import Action

from random import Random

class FishAgent(Agent):
    """
    Never raises when possible, folds when it cannot call.
    """
    def __init__(self, idx, action_space=None, observation_space=None):
        super().__init__(idx, action_space, observation_space)

    def act(self, observation) -> Union[Action, Any]:
        """
        Randomly selects an action from the available action space.
        """
        action = {
            "action": None,
            "raise_amount": None,
        }
        action_mask = observation["action_mask"]["action"]
        # Check if call is available
        if action_mask[Action.CALL.value]:
            action["action"] = Action.CALL
        elif action_mask[Action.CHECK.value]:
            action["action"] = Action.CHECK
        elif action_mask[Action.FOLD.value]:
            action["action"] = Action.FOLD
        else:
            raise ValueError("No legal action available for FishAgent.")

        return action
    
class FoldAgent(Agent):
    """
    Always folds when it cannot check.
    """
    def __init__(self, idx, action_space=None, observation_space=None):
        super().__init__(idx, action_space, observation_space)

    def act(self, observation) -> Union[Action, Any]:
        """
        Randomly selects an action from the available action space.
        """
        action = {
            "action": None,
            "raise_amount": None,
        }
        action_mask = observation["action_mask"]["action"]
        # Check if call is available
        if action_mask[Action.CHECK.value]:
            action["action"] = Action.CHECK
        elif action_mask[Action.FOLD.value]:
            action["action"] = Action.FOLD
        else:
            raise ValueError("No legal action available for FoldAgent.")

        return action