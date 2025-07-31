from random import Random
from typing import Any, Union

from pokergym.agents.base import Agent
from pokergym.env.enums import Action
import numpy as np

class FoldAgent(Agent):
    """
    Always folds if able.
    """
    def __init__(self, idx, action_space=None, observation_space=None):
        super().__init__(
            idx,
            action_space,
            observation_space,
        )

    def act(self, observation: Any, action_mask: dict) -> Action:
        if np.sum(action_mask["action"]) == 0:
            return None
        
        mask = action_mask["action"]
        action = {
            "action": Action.PASS.value,
            "total_bet": 0.0 
        }
        if mask[Action.PASS.value]: # Must pass
            pass
        elif mask[Action.FOLD.value]: # Must fold
            action["action"] = Action.FOLD.value
        elif mask[Action.CHECK.value]: # Must check
            action["action"] = Action.CHECK.value
        elif mask[Action.CALL.value]: # Must call (if all others folded and you were blind)
            action["action"] = Action.CALL.value
        else:
            raise ValueError(f"No valid action available for {self.__class__.__name__}.")
        return action
        
class CheckAgent(Agent):
    """
    Folds when it cannot check.
    """

    def __init__(self, idx, action_space=None, observation_space=None):
        super().__init__(
            idx,
            action_space,
            observation_space,
        )

    def act(self, observation: Any, action_mask: dict) -> Action:
        if np.sum(action_mask["action"]) == 0:
            return None

        mask = action_mask["action"]
        action = {
            "action": Action.PASS.value,
            "total_bet": 0.0 
        }
        if mask[Action.PASS.value]: # Must pass
            pass
        elif mask[Action.CHECK.value]: # Must check
            action["action"] = Action.CHECK.value
        elif mask[Action.FOLD.value]: # Must fold
            action["action"] = Action.FOLD.value
        elif mask[Action.CALL.value]: # Must call (if all others folded and you were blind)
            action["action"] = Action.CALL.value
        else:
            raise ValueError(f"No valid action available for {self.__class__.__name__}.")
        return action


class CallAgent(Agent):
    """
    Folds when it cannot call or check.
    """

    def __init__(self, idx, action_space=None, observation_space=None):
        super().__init__(
            idx,
            action_space,
            observation_space,
        )

    def act(self, observation: Any, action_mask: dict) -> Action:
        if np.sum(action_mask["action"]) == 0:
            return None
        
        mask = action_mask["action"]
        action = {
            "action": Action.PASS.value,
            "total_bet": 0.0 
        }
        if mask[Action.PASS.value]: # Must pass
            pass
        elif mask[Action.CALL.value]: # Must call
            action["action"] = Action.CALL.value
        elif mask[Action.CHECK.value]: # Must check
            action["action"] = Action.CHECK.value
        elif mask[Action.FOLD.value]: # Must fold
            action["action"] = Action.FOLD.value
        else:
            raise ValueError(f"No valid action available for {self.__class__.__name__}.")
        return action


class RaiseAgent(Agent):
    """
    Folds when it cannot raise, call, or check.
    If given will raise by "fixed_raise" or less.
    """

    def __init__(self, idx, action_space=None, observation_space=None, fixed_raise: float = None):
        assert action_space is not None, "Action space must be provided for RaiseAgent."
        super().__init__(
            idx,
            action_space,
            observation_space,
        )
        self.fixed_raise = fixed_raise
    
    def act(self, observation: Any, action_mask: dict) -> Action:
        if np.sum(action_mask["action"]) == 0:
            return None
        
        mask = action_mask["action"]
        action = {
            "action": Action.PASS.value,
            "total_bet": 0.0 
        }
        if mask[Action.PASS.value]: # Must pass
            pass
        elif mask[Action.RAISE.value]: # Must raise
            action["action"] = Action.RAISE.value
            raise_mask = action_mask["total_bet"]
            if self.fixed_raise is not None:
                raise_mask[1] = min(raise_mask[1], self.fixed_raise)
            action["total_bet"] = self.action_space["total_bet"].sample(mask=raise_mask)
        elif mask[Action.CALL.value]: # Must call
            action["action"] = Action.CALL.value
        elif mask[Action.CHECK.value]: # Must check
            action["action"] = Action.CHECK.value
        elif mask[Action.FOLD.value]: # Must fold
            action["action"] = Action.FOLD.value
        else:
            raise ValueError(f"No valid action available for {self.__class__.__name__}.")
        return action
