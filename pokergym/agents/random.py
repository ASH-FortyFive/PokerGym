from typing import Any, Union
from pokergym.agents.base import Agent
from pokergym.env.enums import Action

from random import Random

class RandomAgent(Agent):
    def __init__(self, idx, action_space=None, observation_space=None, seed=None):
        super().__init__(idx, action_space, observation_space)
        self.shuffle_random = Random(seed)

    def act(self, observation) -> Union[Action, Any]:
        """
        Randomly selects an action from the available action space.
        """
        legal_actions = observation["legal_actions"]
        return self.shuffle_random.choice(legal_actions) if legal_actions else Action.FOLD