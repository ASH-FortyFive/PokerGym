from typing import Any, Union
from pokergym.agents.base import Agent
from pokergym.env.enums import Action
import numpy as np

class RandomAgent(Agent):
    """
    Randomly selects an action from the available action space.
    If reasonable_raises is True, raises are sample with-in a gaussian distribution around the min raise + 10% of remaining chips
    """
    def __init__(self, idx, action_space=None, observation_space=None, reasonable_raises=False):
        super().__init__(idx, action_space, observation_space)
        self.reasonable_raises = reasonable_raises

    def act(self, observation, action_mask) -> Union[Action, Any]:
        """
        Randomly selects an action from the available action space.
        """
        if np.sum(action_mask["action"]) == 0:
            return None
        
        action = self.action_space.sample(mask=action_mask) 
        if self.reasonable_raises and action == Action.RAISE:
            min_raise = observation["action_mask"]["total_bet"][0]
            max_raise = observation["action_mask"]["total_bet"][1]
            remaining_chips = observation["chip_counts"][0]
            mean_raise = observation["action_mask"]["total_bet"][0] + 0.1 * remaining_chips
            std_dev = (max_raise - min_raise) / 4  # Adjust the standard deviation as needed
            action["total_bet"] = np.clip(np.random.normal(mean_raise, std_dev), min_raise, max_raise)
        return action
