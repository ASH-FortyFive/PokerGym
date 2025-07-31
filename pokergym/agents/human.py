from typing import Any, Union

import numpy as np

from pokergym.agents.base import Agent
from pokergym.env.enums import Action


class HumanAgent(Agent):
    """
    Check for human input, does auto-pass.
    """

    def __init__(self, idx, action_space=None, observation_space=None, max_chips=None):
        super().__init__(
            idx,
            action_space,
            observation_space,
        )
        self.max_chips = max_chips

    def get_valid_action(self, mask):
        options = [Action(i).name for i in range(len(mask)) if mask[i]]
        question = f"Enter action ({', '.join(options)}): "
        while True:
            user_input = input(question).strip().lower()
            if user_input == "fold" and mask[Action.FOLD.value]:
                return Action.FOLD.value
            elif user_input == "check" and mask[Action.CHECK.value]:
                return Action.CHECK.value
            elif user_input == "raise" and mask[Action.RAISE.value]:
                return Action.RAISE.value
            elif user_input == "call" and mask[Action.CALL.value]:
                return Action.CALL.value
            print(
                f"Invalid action. Must be one of: {', '.join(options)}. Please try again."
            )

    def get_valid_raise(self, mask):
        min_raise = (
            mask[0] if self.max_chips is None else round(mask[0] * self.max_chips)
        )
        max_raise = (
            mask[1] if self.max_chips is None else round(mask[1] * self.max_chips)
        )
        while True:
            try:
                if self.max_chips is not None:
                    total_bet = round(
                        input(
                            f"Enter raise amount (min: {min_raise}, max: {max_raise}): "
                        )
                    )
                else:
                    total_bet = float(
                        input(
                            f"Enter raise amount (min: {min_raise}, max: {max_raise}): "
                        )
                    )
                if min_raise <= total_bet <= max_raise:
                    if self.max_chips is not None:
                        return np.array(total_bet / self.max_chips)
                    else:
                        return np.array(total_bet)
                else:
                    print(
                        f"Invalid raise amount. Must be between {min_raise} and {max_raise}."
                    )
            except ValueError:
                print("Please enter a valid integer for the raise amount.")

    def act(self, observation: Any, action_mask: dict) -> dict:
        mask = action_mask["action"]
        raise_mask = action_mask["total_bet"]
        action = {
            "action": Action.PASS.value,
            "total_bet": 0.0,  # No raise amount since we are folding
        }
        if mask[Action.PASS.value]:  # Must pass
            pass
        else:
            action["action"] = np.array(self.get_valid_action(mask))
            if action["action"] == Action.RAISE.value:
                action["total_bet"] = self.get_valid_raise(raise_mask)
        return action
