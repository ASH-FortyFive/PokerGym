import functools
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from deuces import Card, Deck, Evaluator
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
from pettingzoo import AECEnv

from pokergym.env.cards import WORST_RANK, SeededDeck, card_to_int
from pokergym.env.config import PokerConfig
from pokergym.env.custom_spaces import MaskableBox
from pokergym.env.enums import Action, BettingRound
from pokergym.env.player import Player
from pokergym.env.utils import cards_pretty_str, join_player_ids
from pokergym.visualise.terminal_vis import terminal_render


@dataclass
class PokerGameState:
    players: List[Player]
    active_players: int = None
    community_cards: List[Card] = field(default_factory=list)
    pot: int = 0
    dealer_idx: int = 0
    bb_idx: int = 0
    sb_idx: int = 0
    current_player_idx: int = 0
    current_bet: int = 0
    betting_round: BettingRound = BettingRound.START
    round_number: int = 0
    deck: SeededDeck = field(default_factory=SeededDeck)

    def reset_for_new_hand(self):
        """
        Reset the game state for a new hand.
        This includes resetting player states, community cards, pot, and betting round.
        """
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.betting_round = BettingRound.START
        self.deck.shuffle()
        for player in self.players:
            player.reset_for_new_hand()

    def reset(self, seed: Optional[int] = None):
        """
        Fully reset the game state, including the deck.
        If a seed is provided, it will create a seeded deck for reproducibility.
        """
        self.round_number = 0
        self.dealer_idx = 0
        self.current_player_idx = 0
        self.deck = SeededDeck(seed) if seed is not None else SeededDeck()
        self.reset_for_new_hand()


class PokerEnv(AECEnv):
    metadata = {
        "name": "PokerEnv_v0",
        "render_modes": ["terminal"],
    }

    def __init__(self, config: PokerConfig = PokerConfig(), seed: Optional[int] = None):
        super(PokerEnv, self).__init__()

        self.config = config
        self.seed = seed

        # Initialize the game state
        self.state = PokerGameState(
            players=[
                Player(idx=i, _chips=config.starting_stack)
                for i in range(config.num_players)
            ],
            active_players=config.num_players,
            dealer_idx=0,
            current_player_idx=0,
            deck=SeededDeck(self.seed) if self.seed is not None else SeededDeck(),
        )
        self.MAX_CHIPS = self.config.starting_stack * self.config.num_players

        self.evaluator = Evaluator()

        # AECEnv Attributes
        ## Following https://pettingzoo.farama.org/api/aec/
        self.possible_agents = list(range(config.num_players))
        self.observation_spaces = {
            agent: self.observation_space(agent) for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: self.action_space(agent) for agent in self.possible_agents
        }

        self.agents = self.possible_agents.copy()
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}

    def _remove_agent(self, agent):
        """
        Remove an agent from the environment.
        This is used when a player is no longer active (e.g., folded or all-in).
        """
        del self.terminations[agent]
        del self.truncations[agent]
        del self.rewards[agent]
        del self._cumulative_rewards[agent]
        del self.infos[agent]
        self.agents.remove(agent)


    # Dynamic AECEnv Attributes
    # @property
    # def terminations(self):
    #     # Returns a dictionary of NEWLY terminated players
    #     terminations = {player.idx: not player.active for player in self.state.players if not self.terminated[player.idx]}
    #     self.terminated.update(terminations)
    #     if all(terminations.values()):
    #         print()
    #     return terminations

    # @terminations.setter
    # def terminations(self, value):
    #     raise NotImplementedError("Terminations cannot be set directly in PokerEnv.")

    # @property
    # def truncations(self):
    #     # Returns a dictionary of NEWLY truncated players
    #     return {player.idx: False for player in self.state.players}

    # @truncations.setter
    # def truncations(self, value):
    #     raise NotImplementedError("Truncations cannot be set directly in PokerEnv.")

    # @property
    # def rewards(self):
    #     # TODO: Implement a proper reward system
    #     return {player.idx: 0.0 for player in self.state.players if player.active}

    # @rewards.setter
    # def rewards(self, value):
    #     raise NotImplementedError("Rewards cannot be set directly in PokerEnv.")

    # @property
    # def _cumulative_rewards(self):
    #     return {player.idx: 0.0 for player in self.state.players}

    # @_cumulative_rewards.setter
    # def _cumulative_rewards(self, value):
    #     raise NotImplementedError(
    #         "Cumulative rewards cannot be set directly in PokerEnv."
    #     )

    # @property
    # def infos(self):
    #     return {player.idx: {} for player in self.state.players}

    # @infos.setter
    # def infos(self, value):
    #     raise NotImplementedError("Infos cannot be set directly in PokerEnv.")

    @property
    def agent_selection(self):
        """Return the index of the current agent."""
        return self.state.current_player_idx

    @agent_selection.setter
    def agent_selection(self, value):
        raise NotImplementedError("Agent selection cannot be set directly in PokerEnv.")

    # Gymnasium Environment Methods
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment to the initial state.
        """
        self.seed = seed
        self.state.reset(self.seed)
        self.start_round(cards=options.get("cards") if options else None)
        return self.state, {}

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent=None):
        """
        Get the space of all possible actions for the agent.
        """
        seed = self.seed + agent if self.seed is not None else None
        # print(f"Creating action space for agent {agent} with seed {seed}")
        return Dict(
            {
                "action": Discrete(
                    len(Action), seed=seed
                ),  # Action space for the agent
                "raise_amount": MaskableBox(
                    low=0.0, high=1.0, shape=(), seed=seed
                ),  # Amount to raise
            }
        )

    def _get_action_mask(self, agent):
        """
        Mask actions based on the current game state.
        """
        player = self.state.players[agent]
        mask = [False] * len(Action)  # Initialize all actions as invalid
        min_raise = 0.0
        max_raise = 0.0

        # Check if player can "pass", an action that allows "skipping" their turn
        players_in_hand = [
            p for p in self.state.players if p.active and not p.folded and not p.all_in
        ]
        if (
            player.folded
            or not player.active
            or player.all_in
            or (len(players_in_hand) == 1 and player.bet == self.state.current_bet)
        ):
            mask[Action.PASS.value] = True  # Allow PASS action
            # If the player is folded, inactive, or all-in we only allow PASS
            return {
                "action_mask": {
                    "action": np.array(mask, dtype=np.int8),
                    "raise_amount": np.array([min_raise, max_raise], dtype=np.float32),
                }
            }

        mask[Action.FOLD.value] = True  # Always allow folding

        # Player's bet matches the current bet. They can CHECK.
        if player.bet == self.state.current_bet:
            mask[Action.CHECK.value] = True
        # Player's bet is less than the current bet. They can CALL or .
        elif player.bet < self.state.current_bet:
            # Player can call (or go all-in) if they have any chips left.
            if player.chips > 0:
                mask[Action.CALL.value] = True
        else:
            # This case should not happen in a valid game state.
            raise ValueError("Player bet exceeds current bet, inconsistent state.")

        # To raise, they must be able to at least match the call amount plus a minimum raise.
        ## An all-in raise for less than a min-raise is also possible
        to_raise = self.state.current_bet - player.bet + self.config.min_raise
        if player.chips >= to_raise:
            other_max = max(
                [
                    p.bet + p.chips
                    for p in self.state.players
                    if p.active and not p.folded and p.idx != player.idx
                ]
            )
            if other_max != 0:
                # If all other players are all-in or have no chips, there is no raise possible
                mask[Action.RAISE.value] = True
                min_raise = to_raise
                max_raise = player.chips

                min_raise = min(
                    min_raise, other_max
                )  # Limit raise to the maximum of all other players' chips
                max_raise = min(
                    max_raise, other_max
                )  # Limit raise to the minimum of all other players' chips

        # if np.allclose([min_raise],[max_raise]):
        #     print(f"Warning: min_raise ({min_raise}) and max_raise ({max_raise}) are equal for player {agent}. This may indicate an issue with the action mask logic.")

        min_raise /= self.MAX_CHIPS
        max_raise /= self.MAX_CHIPS

        return {
            "action_mask": {
                "action": np.array(mask, dtype=np.int8),
                "raise_amount": np.array([min_raise, max_raise], dtype=np.float32),
            }
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent=None):
        N = self.config.num_players
        return Dict(
            {
                # Per-agent observations
                "hand": MultiDiscrete(
                    [53] * self.config.max_hand_cards
                ),  # 53 to account for no card
                # Global observations
                ## Table state
                "community_cards": MultiDiscrete(
                    [53] * self.config.max_community_cards
                ),  # 53 to account for no card
                "pot": Box(low=0.0, high=1.0, shape=()),
                "current_bet": Box(low=0.0, high=1.0, shape=()),
                ## Relative player locations
                "chip_counts": Box(low=0.0, high=1.0, shape=(N,)),
                "bets": Box(low=0.0, high=1.0, shape=(N,)),
                "total_contribution": Box(low=0.0, high=1.0, shape=(N,)),
                "folded": MultiBinary(N),
                "all_in": MultiBinary(N),
                "active": MultiBinary(N),
                "relative_dealer_position": Box(low=0.0, high=1.0, shape=()),
                ## Meta Observations
                "betting_round": Discrete(len(BettingRound)),
                "round_number": Box(low=0.0, high=1.0, shape=()),
                # Action Masks
                "action_mask": Dict(
                    {
                        "action": MultiBinary(len(Action)),  # 0 or 1 for each action
                        "raise_amount": Box(low=0.0, high=1.0, shape=(2,)),
                    }
                ),
            }
        )

    def observe(self, agent):
        """
        Get the observation space for a specific agent.
        The observation includes the agent's hand and the global observation.
        """
        obs = self._get_global_obs()
        obs.update(self._get_agent_obs(agent))
        obs.update(self._get_action_mask(agent))

        relative_obs = [
            "chip_counts",
            "bets",
            "total_contribution",
            "folded",
            "all_in",
            "active",
        ]
        for key in relative_obs:
            obs[key] = self._rotate_to_idx(obs[key], agent)
        # obs["relative_dealer_position"] = np.array(
        #     [
        #         (self.state.dealer_idx - agent)
        #         % self.config.num_players
        #         / self.config.num_players
        #     ]
        # )
        obs["relative_dealer_position"] = float((self.state.dealer_idx - agent) % self.config.num_players / self.config.num_players)
        return obs

    def _get_global_obs(self):
        """
        Get the observation visible to all agents.
        This includes the community cards, pot, current bet, and betting round.
        """
        obs = {
            "community_cards": [
                (
                    card_to_int(self.state.community_cards[i])
                    if i < len(self.state.community_cards)
                    else 0
                )
                for i in range(self.config.max_community_cards)
            ],
            "pot": self.state.pot / self.MAX_CHIPS,
            "current_bet": self.state.current_bet / self.MAX_CHIPS,
            "betting_round": self.state.betting_round.value,
            "chip_counts": [
                player.chips / self.MAX_CHIPS for player in self.state.players
            ],
            "bets": [player.bet / self.MAX_CHIPS for player in self.state.players],
            "total_contribution": [
                player.total_contribution / self.MAX_CHIPS
                for player in self.state.players
            ],
            "folded": [player.folded for player in self.state.players],
            "all_in": [player.all_in for player in self.state.players],
            "active": [player.active for player in self.state.players],
            "round_number": (
                self.state.round_number / self.config.max_rounds
                if self.config.max_rounds > 0
                else 0
            ),
        }

        return obs

    def _get_agent_obs(self, agent):
        """
        Get the observation specific to the agent.
        This includes the agent's hand and the global observation.
        """
        player = self.state.players[agent]
        return {
            "hand": [card_to_int(card) for card in player.hand],
        }

    def _rotate_to_idx(self, observation, idx):
        """
        Rotate an observation to be start at given idx.
        """
        return observation[idx:] + observation[:idx]

    def step(self, action: Dict):
        """
        Execute one time step within the environment.
        Input:
        - action: A dictionary containing the "action" and optional "raise_amount".
        """
        for agent in self.agents:
            if self.terminations[agent]:
                self._remove_agent(agent)
                if agent == self.state.current_player_idx:
                    return

        # Take the Action
        player = self.state.players[self.state.current_player_idx]

        # Convert normalized raise amount to absolute chips if provided
        extra_bet_chips = None
        if action["action"] == Action.RAISE:
            assert (
                "raise_amount" in action is not None
            ), "Raise requires an raise_amount amount."

        # Apply the action
        self._handle_action(player.idx, action)
        player.last_action = Action(action["action"])

        # Check if betting round is finished:
        ## All active, non-folded players must have matched current_bet or be all-in
        all_acted = True
        for p in self.state.players:
            if not p.active or p.folded:
                continue
            if p.last_action is None:
                all_acted = False
                break
            # End the round if everyone but the next player has no raised
            if p.last_action is Action.RAISE and p.idx != self.next_active_player_idx(
                self.state.current_player_idx
            ):
                all_acted = False
                break

        # Check if only one player remains active (others folded)
        player_in_hand = [p for p in self.state.players if p.active and not p.folded]
        if len(player_in_hand) == 1:
            # Only one player remains, so skip to showdown
            while self.state.betting_round != BettingRound.SHOWDOWN:
                self.step_round()
        elif all_acted:
            self.step_round()
        else:
            # If not all players have acted, move to the next player
            ## Redundant if we enter showdown, but cleaner this way
            self.state.current_player_idx = self.next_active_player_idx(
                self.state.current_player_idx
            )

        if self.state.betting_round == BettingRound.SHOWDOWN:
            self.end_round()
            if self.state.betting_round == BettingRound.END:
                for player in self.state.players:
                    if player.active:
                        player.active = False  # Mark all players as inactive
                        self.terminations[player.idx] = True
            else:
                self.start_round()

        return

    def _handle_action(self, agent, action: dict[str, Union[Action, float]]):
        """
        Apply the action taken by the agent to the game state.
        Allows for all actions defined in the Action enum.
        Includes error handling for invalid actions.
        """
        assert (
            agent == self.state.current_player_idx
        ), f"Player {agent} attempted to take an action when it was not their turn."
        player = self.state.players[agent]
        action_enum = Action(action["action"])
        extra_bet = action.get("raise_amount", 0.0)

        if action_enum is None:
            return True

        if action_enum is Action.FOLD:
            player.folded = True
        elif action_enum is Action.CHECK:
            assert (
                player.bet == self.state.current_bet
            ), "Player can only check if their bet matches the current bet."
        elif action_enum is Action.CALL:
            assert player.chips > 0, "Player cannot call with no chips."
            to_call = min(self.state.current_bet - player.bet, player.chips)
            player.make_bet(to_call)
        elif action_enum is Action.RAISE:
            # Convert normalized raise amount to absolute chips
            extra_chips = round(
                extra_bet * self.MAX_CHIPS
            )  # Shold we enforce rounding?
            other_max = max(
                [
                    p.bet + p.chips
                    for p in self.state.players
                    if p.active and not p.folded and p.idx != player.idx
                ]
            )
            min_bet = self.state.current_bet + self.config.min_raise
            min_bet = min(min_bet, other_max)
            assert (
                player.bet + extra_chips >= min_bet
            ), "Player must raise at least the minimum raise amount."
            player.make_bet(extra_chips)
            self.state.current_bet = player.bet
        elif action_enum is Action.PASS:
            pass
        else:
            raise ValueError(f"Invalid action: {action} for player {agent}.")
        return True

    # Poker Logic
    def start_round(self, cards: Optional[dict[int, List[Card]]] = None):
        """
        Start a new round of poker, dealing cards and setting blinds.
        - If `cards` is provided (dict of player index and list of cards), it will use those cards instead of drawing from the deck.
        """
        assert (
            self.state.betting_round == BettingRound.START
        ), "Cannot start a new round before ending the current one."
        # Reset the game state for a new round
        self.state.reset_for_new_hand()
        for player in self.state.players:
            player.reset_for_new_hand()

        # Deal initial cards
        if cards:
            for i, _cards in cards.items():
                if len(_cards) != self.config.max_hand_cards:
                    raise ValueError(
                        f"Player {i} must receive exactly {self.config.max_hand_cards} cards."
                    )
                player = self.state.players[i]
                if not player.active:
                    continue
                player.hand = _cards
                for card in _cards:
                    self.state.deck.cards.remove(card)
        for i in range(self.config.max_hand_cards * self.config.num_players):
            player = self.state.players[i % self.config.num_players]
            if len(player.hand) >= self.config.max_hand_cards:
                continue
            if not player.active:
                continue
            card = self.state.deck.draw(1)
            player.add_card(card)

        # Set blinds
        self.state.sb_idx = self.next_active_player_idx(self.state.dealer_idx)
        sb_player = self.state.players[self.state.sb_idx]
        sb_bet = self.config.small_blind
        if sb_player.chips < sb_bet:
            sb_bet = sb_player.chips
            sb_player.all_in = True
        sb_player.make_bet(sb_bet)

        self.state.bb_idx = self.next_active_player_idx(self.state.sb_idx)
        bb_player = self.state.players[self.state.bb_idx]
        bb_bet = self.config.big_blind
        if bb_player.chips < bb_bet:
            bb_bet = bb_player.chips
            bb_player.all_in = True
        bb_player.make_bet(bb_bet)

        # Set the current bet and betting round
        self.state.current_bet = max(sb_bet, bb_bet)
        self.state.betting_round = BettingRound.PREFLOP
        self.state.current_player_idx = self.next_active_player_idx(self.state.bb_idx)

    def end_round(self):
        """
        End the current round of poker, determining the winner and distributing the pot.
        """
        # Find the winner(s) and distribute the pot
        assert (
            self.state.betting_round == BettingRound.SHOWDOWN
        ), "Cannot end round before showdown."
        scores = np.array(
            [
                self.evaluator.evaluate(player.hand, self.state.community_cards)
                for player in self.state.players
            ]
        )
        # player_in_hand = [p for p in self.state.players if p.active and not p.folded]
        winners = set()
        pots = self._construct_pots()
        for pot_amount, player_indices in pots:
            # Determine the winners for this pot
            min_score = np.min(scores[player_indices])
            pot_winners = [
                self.state.players[i] for i in player_indices if scores[i] == min_score
            ]
            assert pot_winners, "No winners found for the pot."
            # Split the pot among the winners
            split_amount = pot_amount // len(pot_winners)
            for winner in pot_winners:
                winner.give_chips(split_amount)
                winners.add(winner.idx)
                self.state.pot -= split_amount
                print(
                    f"Player {winner.idx} wins {split_amount} chips from the pot of {pot_amount}."
                )
            remainder = pot_amount % len(pot_winners)
            payout_player = self.state.players[
                self.next_active_player_idx(self.state.dealer_idx)
            ]
            while remainder > 0:
                if payout_player in pot_winners:
                    payout_player.give_chips(1)
                    self.state.pot -= 1
                    remainder -= 1
                    print(
                        f"Player {payout_player.idx} receives an extra chip from the pot of {pot_amount}."
                    )
                payout_player = self.state.players[
                    self.next_active_player_idx(payout_player.idx)
                ]
        assert (
            self.state.pot == 0
        ), f"Pot not fully distributed, remaining: {self.state.pot}"

        # Remove Players who are out of chips
        active_count = self.config.num_players
        for player in self.state.players:
            if player.chips == 0 and player.active:
                player.active = False
                self.terminations[player.idx] = True
                active_count -= 1
            elif not player.active:
                active_count -= 1
        self.state.active_players = active_count

        # Update Game State
        if active_count == 0:
            raise ValueError("All players are out of chips. Cannot continue the game.")
        elif active_count == 1:
            winners.add(
                next(player.idx for player in self.state.players if player.active)
            )
            winner = self.state.players[winners.pop()]
            self.terminations[winner.idx] = True
            winner.active = False
            self.state.betting_round = BettingRound.END
        else:
            self.state.betting_round = BettingRound.START
            self.state.round_number += 1
            self.state.dealer_idx = self.next_active_player_idx(self.state.dealer_idx)

        return winners

    def step_round(self, cards: Optional[dict[BettingRound, List[Card]]] = None):
        """
        Advance the betting round to the next stage (Flop, Turn, River).
        - If `cards` is provided (dict of betting round and list of cards), it will use those cards instead of drawing from the deck.
        TODO: Make the number of rounds dynamic based on the config.
        """
        if self.state.betting_round == BettingRound.START:
            raise ValueError("Cannot step round before starting a round.")
        if self.state.betting_round == BettingRound.PREFLOP:
            self.state.betting_round = BettingRound.FLOP
            self._collect_bets()
            self._clear_actions()
            self.state.community_cards += self._draw_for_round(BettingRound.FLOP, cards)
            self.state.current_player_idx = self.nearest_active_player_idx(
                self.state.dealer_idx
            )
        elif self.state.betting_round == BettingRound.FLOP:
            self.state.betting_round = BettingRound.TURN
            self._collect_bets()
            self._clear_actions()
            self.state.community_cards += self._draw_for_round(BettingRound.TURN, cards)
            self.state.current_player_idx = self.nearest_active_player_idx(
                self.state.dealer_idx
            )
        elif self.state.betting_round == BettingRound.TURN:
            self.state.betting_round = BettingRound.RIVER
            self._collect_bets()
            self._clear_actions()
            self.state.community_cards += self._draw_for_round(
                BettingRound.RIVER, cards
            )
            self.state.current_player_idx = self.nearest_active_player_idx(
                self.state.dealer_idx
            )
        elif self.state.betting_round == BettingRound.RIVER:
            self._collect_bets()
            self._clear_actions()
            self.state.betting_round = BettingRound.SHOWDOWN

    def _collect_bets(self):
        """
        Collect bets from all active players.
        This is called at the end of each betting round to update the pot and player contributions.
        """
        for player in self.state.players:
            amount = player.bet
            if amount != self.state.current_bet:
                if player.active and not player.folded and not player.all_in:
                    raise ValueError(
                        f"Player {player.idx} must bet {self.state.current_bet}, but bet {amount}."
                    )
            self.state.pot += amount
            player.bet = 0
        self.state.current_bet = 0

    def _clear_actions(self):
        """
        Clear the actions of all players at the end of a betting round.
        This is called to reset the state for the next betting round.
        """
        for player in self.state.players:
            player.last_action = None

    def _draw_for_round(self, round: BettingRound, cards: Optional[List[Card]] = None):
        """
        Draw cards for the specified betting round.
        - If `cards` is provided, it will use those cards instead of drawing from the deck.
        """
        if round not in self.config.cards_per_round:
            raise ValueError(f"Invalid betting round: {round}")

        # Draw predetermined cards if provided
        drawn = []
        if cards:
            cards_round = cards.get(round, [])
            if len(cards_round) > self.config.cards_per_round[round]:
                raise ValueError(
                    f"Expected at most {self.config.cards_per_round[round]} cards for {round}, got {len(cards)}."
                )
            for card in cards_round:
                self.state.deck.cards.remove(card)
                drawn.append(card)

        # Draw from the deck if no / not enough cards were provided
        if len(drawn) < self.config.cards_per_round[round]:
            remaining = self.config.cards_per_round[round] - len(drawn)
            for _ in range(remaining):
                drawn.append(self.state.deck.draw())
        return drawn

    # Utility Methods
    def _find_active_player(
        self, start_idx: int, direction: int, include_start: bool
    ) -> int:
        """
        Finds an active and non-folded player starting from `start_idx`, moving in `direction` (+1 or -1).
        If `include_start` is True, considers `start_idx` as the first valid candidate.
        """
        num_players = self.config.num_players
        offsets = range(num_players) if include_start else range(1, num_players + 1)

        for offset in offsets:
            idx = (start_idx + direction * offset) % num_players
            player = self.state.players[idx]
            if player.active and not player.folded:
                return idx
        raise ValueError("No active and non-folded player found.")

    def next_active_player_idx(self, idx):
        return self._find_active_player(idx, 1, include_start=False)

    def nearest_active_player_idx(self, idx):
        return self._find_active_player(idx, 1, include_start=True)

    def render(self):
        if self.render_mode == "terminal":
            terminal_render(self.state, self.evaluator)

    def _construct_pots(self) -> List[Tuple[int, List[int]]]:
        """
        Construct pots based on player contributions (tracked by player.total_contribution).
        Returns a list of tuples, where each tuple contains the pot amount and a list of player indices who contributed to that pot.
        TODO: Handle edge cases like players going all-in and uneven contributions.
        """
        contributions = [
            (i, p.total_contribution, p.active and not p.folded)
            for i, p in enumerate(self.state.players)
            if p.total_contribution > 0
        ]
        contributions.sort(key=lambda x: x[1])
        pots = []
        while contributions:
            min_contribution = contributions[0][1]  # Smallest contribution
            eligible_players = []
            pot_amount = 0
            for idx, contribution, eligible in contributions:
                pot_amount += min_contribution
                if eligible:
                    eligible_players.append(idx)
            pots.append((pot_amount, eligible_players))

            # Subtract min_contrib from all and remove 0s
            contributions = [
                (idx, contribution - min_contribution, eligible)
                for idx, contribution, eligible in contributions
                if contribution - min_contribution > 0
            ]
        return pots
