from random import Random
from typing import Any, Union

from pokergym.agents.base import Agent
from pokergym.env.enums import Action


class FoldAgent(Agent):
    """
    Always folds.
    """
    def __init__(self, idx, action_space=None, observation_space=None):
        super().__init__(
            idx,
            action_space,
            observation_space,
        )

    def act(self, observation: Any, action_mask: dict) -> Action:
        mask = action_mask["action"]
        action = {
            "action": Action.PASS,
            "raise_amount": [0, 0]  # No raise amount since we are folding
        }
        if mask[Action.PASS.value]: # Must pass
            pass
        elif mask[Action.FOLD.value]: # Must fold
            action["action"] = Action.FOLD
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
        mask = action_mask["action"]
        action = {
            "action": Action.PASS,
            "raise_amount": [0, 0]  # No raise amount since we are folding
        }
        if mask[Action.PASS.value]: # Must pass
            pass
        elif mask[Action.CHECK.value]: # Must check
            action["action"] = Action.CHECK
        elif mask[Action.FOLD.value]: # Must fold
            action["action"] = Action.FOLD
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
        mask = action_mask["action"]
        action = {
            "action": Action.PASS,
            "raise_amount": [0, 0]  # No raise amount since we are folding
        }
        if mask[Action.PASS.value]: # Must pass
            pass
        elif mask[Action.CALL.value]: # Must call
            action["action"] = Action.CALL
        elif mask[Action.CHECK.value]: # Must check
            action["action"] = Action.CHECK
        elif mask[Action.FOLD.value]: # Must fold
            action["action"] = Action.FOLD
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
        mask = action_mask["action"]
        action = {
            "action": Action.PASS,
            "raise_amount": [0, 0]  # No raise amount since we are folding
        }
        if mask[Action.PASS.value]: # Must pass
            pass
        elif mask[Action.RAISE.value]: # Must raise
            action["action"] = Action.RAISE
            raise_mask = action_mask["raise_amount"]
            if self.fixed_raise is not None:
                raise_mask[1] = min(raise_mask[1], self.fixed_raise)
            action["raise_amount"] = self.action_space["raise_amount"].sample(mask=raise_mask)
        elif mask[Action.CALL.value]: # Must call
            action["action"] = Action.CALL
        elif mask[Action.CHECK.value]: # Must check
            action["action"] = Action.CHECK
        elif mask[Action.FOLD.value]: # Must fold
            action["action"] = Action.FOLD
        else:
            raise ValueError(f"No valid action available for {self.__class__.__name__}.")
        return action
