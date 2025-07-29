from typing import Any, Union
from pokergym.agents.base import Agent
from pokergym.env.enums import Action

import numpy as np

class RandomAgent(Agent):
    def __init__(self, idx, action_space=None, observation_space=None, reasonable_raises=False):
        super().__init__(idx, action_space, observation_space)
        self.reasonable_raises = reasonable_raises

    def act(self, observation) -> Union[Action, Any]:
        """
        Randomly selects an action from the available action space.
        """
        legal_actions = observation["action_mask"]["action"]
        action_indices = np.where(legal_actions)[0]
        action_idx = np.random.choice(action_indices)
        action = Action(action_idx)
        raise_amount = None
        if action == Action.RAISE:
            min_raise = observation["action_mask"]["raise_amount"][0]
            max_raise = observation["action_mask"]["raise_amount"][1]
            if not self.reasonable_raises:
                raise_amount = np.random.uniform(min_raise, max_raise)
            else:
                # Sample with-in a gaussian distribution around the min raise + 10% of remaining chips
                remaining_chips = observation["chip_counts"][0] # Observations are always relative to the agent's perspective
                mean_raise = min_raise + 0.1 * remaining_chips
                std_dev = (max_raise - min_raise) / 4  # Adjust the standard deviation as needed
                raise_amount = np.clip(np.random.normal(mean_raise, std_dev), min_raise, max_raise)


        return {
            "action": action,
            "raise_amount": raise_amount
        }
