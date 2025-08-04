import functools
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from deuces import Card, Evaluator
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
from gymnasium.utils import EzPickle
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers

from pokergym.env.cards import SeededDeck, card_to_int
from pokergym.env.config import PokerConfig
from pokergym.env.custom_spaces import MaskableBox
from pokergym.env.enums import Action, BettingRound
from pokergym.env.states import PlayerState
from pokergym.env.poker_logic import ActionDict, Poker, PokerGameState
from pokergym.visualise.terminal_vis import terminal_render


def env(**kwargs):
    """Wrapper function to create a Poker environment.
    Args:
        **kwargs: Additional keyword arguments to pass to the Poker environment.
    Returns:
        An instance of the Poker environment, wrapped with various utility wrappers.
    """
    env = raw_env(**kwargs)
    # env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    # env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv, EzPickle):
    """Poker environment for Texas Hold'em.
    This class wraps the Poker logic to follow the AEC (Agent Environment Cycle) interface.
    It supports multiple players, betting rounds, and various poker actions.
    """

    metadata = {
        "name": "PokerEnv_v1",
        "render_modes": ["terminal"],
    }

    def __init__(
        self,
        config: PokerConfig = PokerConfig(),
        seed: Optional[int] = None,
        autorender: bool = False,
    ):
        EzPickle.__init__(self, config, seed)
        super().__init__()
        self.seed = seed
        self.poker = Poker(config=config, seed=self.seed, autorender=autorender)

        # AECEnv Attributes
        ## Following https://pettingzoo.farama.org/api/aec/
        self.possible_agents = [f"player_{i}" for i in range(config.num_players)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, range(config.num_players))
        )
        self.inv_agent_name_mapping = {v: k for k, v in self.agent_name_mapping.items()}

        # Players have identical observation and action spaces
        self.observation_spaces = {
            agent: self.observation_space(agent) for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: self.action_space(agent) for agent in self.possible_agents
        }

        self._hand_over = {agent: False for agent in self.possible_agents}

    # Gymnasium Environment Methods
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to the initial state.
        Args:
            seed: Optional seed for reproducibility.
            options: Additional options for resetting the environment.
        Returns:
            A tuple containing the initial observations for all agents.
        """
        self.seed = seed if seed is not None else self.seed
        self.poker.reset(seed=self.seed, options=options)

        self.agents = self.possible_agents.copy()
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()


    # Spaces
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent=None):
        """Get the action space for a specific agent.
        Args:
            agent: The agent for which to get the action space. If None, returns the default action space.
            Passing the agent sets the seed used to sample actions.
        Returns:
            A gymnasium Discrete space representing the action space for the agent.
        """
        agent_id = self.agent_name_mapping[agent] if agent is not None else None
        seed = self.seed + agent_id if self.seed is not None else None
        MAX_BB = float(self.poker.MAX_CHIPS / self.poker.config.big_blind)
        return Dict(
            {
                # Action space for the agent
                "action": Discrete(len(Action), seed=seed),
                # Raise amount space for the agent, normalized to be in terms of big blinds
                "total_bet": MaskableBox(low=0.0, high=MAX_BB, shape=(), seed=seed),
            }
        )

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent=None):
        """Get the observation space for a specific agent.
        Currently, all agents share the same observation space.
        Args:
            agent: The agent for which to get the observation space. If None, returns the default observation space.
        Returns:
            A gymnasium Dict space representing the observation space for the agent.
        """
        N = self.poker.game_state.num_players
        MAX_BB = float(self.poker.MAX_CHIPS / self.poker.config.big_blind)
        return Dict(
            {
                # Per-agent observations
                ## Players hands, 53 to account for no card (0  )
                "hand": MultiDiscrete([53] * self.poker.config.max_hand_cards),
                # Global observations
                ## Table state
                ### Community cards, 53 to account for no card (0)
                "community_cards": MultiDiscrete(
                    [53] * self.poker.config.max_community_cards
                ),
                ### Normalized to represent fractions of the maximum chips
                "pot": Box(low=0.0, high=MAX_BB, shape=()),
                "current_bet": Box(low=0.0, high=MAX_BB, shape=()),
                ## Relative to player locations
                "chip_counts": Box(low=0.0, high=MAX_BB, shape=(N,)),
                "bets": Box(low=0.0, high=MAX_BB, shape=(N,)),
                "total_contribution": Box(low=0.0, high=MAX_BB, shape=(N,)),
                "folded": MultiBinary(N),
                "all_in": MultiBinary(N),
                "active": MultiBinary(N),
                "relative_dealer_position": Box(low=0.0, high=1.0, shape=()),
                ## Meta Observations
                "betting_round": Discrete(len(BettingRound)),
                "hand_number": Box(low=0.0, high=1.0, shape=()),
                # Action Masks
                "action_mask": Dict(
                    {
                        "action": MultiBinary(len(Action)),  # 0 or 1 for each action
                        "total_bet": Box(low=0.0, high=MAX_BB, shape=(2,)),
                    }
                ),
            }
        )

    # Reading Game State
    def observe(self, agent):
        """Get the observation space for a specific agent.
        The observation includes the agent's hand and the global observation.
        Args:
            agent: The agent for which to get the observation.
        Returns:
            A dictionary containing the agent's hand and the global observation.
        """
        agent_id = self.agent_name_mapping[agent]
        obs = self.state()
        obs.update(self._get_agent_obs(agent_id))
        obs.update(self._get_action_mask(agent_id))

        relative_obs = [
            "chip_counts",
            "bets",
            "total_contribution",
            "folded",
            "all_in",
            "active",
        ]
        for key in relative_obs:
            obs[key] = self._rotate_to_idx(obs[key], agent_id)
        obs["relative_dealer_position"] = np.array(
            (self.poker.game_state.dealer_idx - agent_id)
            % self.poker.config.num_players
            / self.poker.config.num_players,
            dtype=np.float32,
        )
        return obs

    def state(self):
        """Get the global state of the game.
        This includes the community cards, pot, current bet, betting round, and round number.
        Everything is normalized to represent fractions of the maximum chips or rounds.
        Returns:
            A dictionary containing the global state of the game.
        """
        # Per-agent observations
        folded = np.zeros(self.poker.config.num_players, dtype=np.int8)
        all_in = np.zeros(self.poker.config.num_players, dtype=np.int8)
        active = np.zeros(self.poker.config.num_players, dtype=np.int8)
        bets = np.zeros(self.poker.config.num_players, dtype=np.float32)
        total_contribution = np.zeros(self.poker.config.num_players, dtype=np.float32)
        chip_counts = np.zeros(self.poker.config.num_players, dtype=np.float32)
        for p in self.poker.game_state.players:
            folded[p.idx] = p.folded
            all_in[p.idx] = p.all_in
            active[p.idx] = p.active
            bets[p.idx] = self._norm_chip(p.bet)
            total_contribution[p.idx] = self._norm_chip(p.total_contribution)
            chip_counts[p.idx] = self._norm_chip(p.chips)
        # Table observations
        cards = np.array(
            [
                (
                    card_to_int(self.poker.game_state.community_cards[i])
                    if i < len(self.poker.game_state.community_cards)
                    else 0
                )
                for i in range(self.poker.config.max_community_cards)
            ],
            dtype=np.int8,
        )
        hand_number = np.array(
            (
                self.poker.game_state.hand_number / self.poker.config.max_hands
                if self.poker.config.max_hands > 0
                else 0
            ),
            dtype=np.float32,
        )
        return {
            "community_cards": cards,
            "pot": np.array(
                self._norm_chip(self.poker.game_state.pot), dtype=np.float32
            ),
            "current_bet": np.array(
                self._norm_chip(self.poker.game_state.current_bet), dtype=np.float32
            ),
            "betting_round": self.poker.game_state.betting_round.value,
            "chip_counts": chip_counts,
            "bets": bets,
            "total_contribution": total_contribution,
            "folded": folded,
            "all_in": all_in,
            "active": active,
            "hand_number": hand_number,
        }

    def _get_agent_obs(self, agent_id):
        """Get the observation for a specific agent.
        Args:
            agent_id (int): The agent for which to get the observation.
        Returns:
            A dictionary containing the agent's hand and its index.
        """
        player = self.poker.game_state.players[agent_id]
        hand = np.array(
            [
                card_to_int(player.hand[i]) if i < len(player.hand) else 0
                for i in range(self.poker.config.max_hand_cards)
            ],
            dtype=np.int8,
        )
        return {
            "hand": hand,
        }

    def _get_action_mask(self, agent_id):
        """Get the action mask for a specific agent.
        Args:
            agent_id (int): The agent for which to get the action mask.
        Returns:
            A dictionary containing the action mask for the agent.
        """
        action_dict = self.poker.legal_actions(idx=agent_id)
        action_mask = np.zeros(len(Action), dtype=np.int8)
        for action in action_dict["action"]:
            action_mask[action.value] = 1
        total_bet = np.vectorize(lambda x: self._norm_chip(x))(
            action_dict["total_bet"]
        ).astype(np.float32)
        return {
            "action_mask": {
                "action": action_mask,
                "total_bet": total_bet,
            }
        }

    # Overriding AECEnv Methods
    def last(self, observe: bool = True
    ) -> tuple["ObsType", float, bool, bool, dict[str, Any]]:
        """Returns observation, cumulative reward, terminated, truncated, info for the current agent (specified by self.agent_selection)."""
        obs, _, term, trunc, info = super().last(observe=observe)
        reward = self._cumulative_rewards[self.agent_selection]
        return (
            obs,
            reward,
            term,
            trunc,
            info
        )

    
     # Stepping the Environment
    def step(self, action):
        """Perform a step in the environment.
        This method processes the action taken by the current agent, updates the game state,
        and manages the turn order.
        Will cycle through agents, only updating the game logic if the current agent is the one taking action.
        Args:
            action: The action to perform, which includes the action type and total bet amount.
        Raises:
            ValueError: If the action is not valid for the current agent.
        """
        self._cumulative_rewards[self.agent_selection] = 0.0  # Clear rewards at the start of each step
        if self._hand_over[self.agent_selection]:
            self.infos[self.agent_selection]["hand_over"] = True
            self._hand_over[self.agent_selection] = False
        else:
            self.infos[self.agent_selection]["hand_over"] = False

        self._clear_rewards()
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            if len(self.agents) != 0:
                self.agent_selection = self._agent_selector.next()
            return True

        agent_id = self.agent_name_mapping[self.agent_selection]
        player_deltas = None  # None if the hand is not ended
        if agent_id == self.poker.game_state.current_idx:
            action_dict = self._convert_action(action)
            player_deltas = self.poker.step(idx=agent_id, action_dict=action_dict)
        hand_over = player_deltas is not None
        # Deltas being provided means the hand has ended
        if hand_over:
            # Calculate rewards and update game state
            for player_idx, chip_delta in player_deltas.items():
                agent = self.inv_agent_name_mapping[player_idx]
                norm_delta = self._norm_chip(chip_delta)
                if agent in self.rewards:
                    self.rewards[agent] += norm_delta
                self._hand_over[agent] = True
            self._accumulate_rewards()

            # Update terminations and truncations
            self.terminations = {
                agent: (
                    not self.poker.game_state.players[
                        self.agent_name_mapping[agent]
                    ].active
                )
                for agent in self.agents
            }
            if self.poker.game_over():
                self.truncations = {agent: True for agent in self.agents}

        self.agent_selection = self._agent_selector.next()
        # self._clear_rewards()
        return True

    def _convert_action(self, action) -> ActionDict:
        """Convert the action dictionary to a format suitable for the Poker logic.
        Args:
            action: The action dictionary containing the action type and total bet amount.
        Returns:
            An ActionDict object representing the action to be taken.
        """
        bet = np.vectorize(lambda x: self._denorm_chip(x))(action["total_bet"])
        action_enum = Action(action["action"])
        return ActionDict(
            action=action_enum,
            total_bet=bet,
        )

    # Utility Methods
    def _norm_chip(self, value: float) -> float:
        """Normalize a chip value to be in terms of big blinds."""
        return float(value / self.poker.config.big_blind)

    def _denorm_chip(self, value: float) -> float:
        """Denormalize a chip value from being in terms of big blinds to actual chip value."""
        return round(value * self.poker.config.big_blind)

    def _rotate_to_idx(self, observation: np.ndarray, idx: int) -> np.ndarray:
        """Rotate the observation array to start from the specified agent index.
        Args:
            observation: The observation array to rotate.
            idx: The index of the agent to start from.
        Returns:
            The rotated observation array.
        """
        if idx == 0:
            return observation
        return np.roll(observation, -idx, axis=0)
    
    def render(self):
        """Render the environment.
        Mode is given by the config, default is "terminal".
        Returns:
            If the mode is "terminal", prints a representation of the game state.
        """
        self.poker.render()

    def close(self):
        """Close the environment.
        This method is called to clean up resources when the environment is no longer needed.
        """
        pass


if __name__ == "__main__":
    import tyro
    from pettingzoo.test import api_test

    # Define a wrapper to include optional seed
    @dataclass
    class ExtraArgs:
        config: PokerConfig
        seed: Optional[int] = 0  # Default seed value

    args = tyro.cli(ExtraArgs)
    ENV = env(config=args.config, seed=args.seed, autorender=False)
    ENV.reset(seed=args.seed)
    import pdb

    try:
        api_test(ENV, num_cycles=args.config.max_hands * 10_000, verbose_progress=True)
    except Exception as e:
        print(f"An error occurred: {e}")
        pdb.post_mortem()
    api_test(ENV, num_cycles=args.config.max_hands * 10_000, verbose_progress=True)
    # obs = ENV.observe("player_0")
    # obs_space = ENV.observation_space("player_0")
    # obs_space.contains(obs)
    # print("Observation space is valid.")
    # for key, value in obs_space.items():
    #     if isinstance(value, Dict):
    #         for sub_key, sub_value in value.items():
    #             print(f"{key}.{sub_key}: {sub_value.contains(obs[key][sub_key])}")
    #     else:
    #         print(f"{key}: {value.contains(obs[key])}")

    # print("All observations are valid according to the observation space.")
