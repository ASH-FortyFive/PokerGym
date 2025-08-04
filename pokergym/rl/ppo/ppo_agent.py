from pokergym.agents.base import Agent
from pokergym.env.conversion import obs_to_tensor, action_to_tensor, tensor_to_action

class PPOAgent(Agent):
    """
    Base class for PPO agents.
    This class can be extended to implement specific PPO agent behaviors.
    """

    def __init__(self, idx, action_space=None, observation_space=None):
        super().__init__(idx, action_space, observation_space)

    def act(self, observation, action_mask):
        """
        Override this method to implement the agent's action selection logic.
        """
        obs_tensor = obs_to_tensor(observation, self.observation_space)
        action, log_prob, entropy = self.policy.act(obs_tensor, action_mask)
        return tensor_to_action(action, self.action_space), log_prob, entropy