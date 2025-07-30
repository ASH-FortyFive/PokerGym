import functools
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from deuces import Card, Deck, Evaluator
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers

from pokergym.env.cards import WORST_RANK, SeededDeck, card_to_int
from pokergym.env.config import PokerConfig
from pokergym.env.custom_spaces import MaskableBox
from pokergym.env.enums import Action, BettingRound
from pokergym.env.player import Player
from pokergym.env.utils import cards_pretty_str, join_player_ids
from pokergym.visualise.terminal_vis import terminal_render


@dataclass
class PokerGameState:
    players: List[Player] = field(default_factory=list)
    community_cards: List[Card] = field(default_factory=list)
    pot: int = 0
    current_idx: int = 0
    dealer_idx: int = 0
    bb_idx: int = 0
    sb_idx: int = 0
    current_bet: int = 0
    betting_round: BettingRound = BettingRound.START
    round_number: int = 0
    stack_size: int = 0
    num_players: int = 0
    deck: SeededDeck = field(default_factory=SeededDeck)

    def reset_for_new_hand(self):
        """
        Reset the game state for a new hand.
        This includes resetting player states, community cards, pot, and betting round.
        """
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.deck.shuffle()
        for player in self.players:
            player.reset_for_new_hand()

    def reset(self, seed: Optional[int] = None):
        """
        Fully reset the game state, including the deck.
        If a seed is provided, it will create a seeded deck for reproducibility.
        """
        self.betting_round = BettingRound.START
        self.round_number = 0
        self.dealer_idx = 0
        self.deck = SeededDeck(seed) if seed is not None else SeededDeck()
        self.players = [
            Player(idx=i, _chips=self.stack_size) for i in range(self.num_players)
        ]
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
        self.game_state = PokerGameState(
            stack_size=config.starting_stack,
            num_players=config.num_players,
        )
        self.MAX_CHIPS = self.config.starting_stack * self.config.num_players
        self.evaluator = Evaluator()

        # AECEnv Attributes
        ## Following https://pettingzoo.farama.org/api/aec/
        self.possible_agents = list(range(config.num_players))
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # Players have identical observation and action spaces
        self.observation_spaces = {
            agent: self.observation_space(agent) for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: self.action_space(agent) for agent in self.possible_agents
        }

    def _was_dead_step(self, action):
        # Want to update state and do my own agent selection
        agent = self.agent_selection
        self.game_state.players[agent].active = False
        super()._was_dead_step(action)
        self.agent_selection = agent
        # self._agent_selector.reinit(self.agents)
        # self._agent_selector._current_agent = self.agent_selection
        # self._agent_selector.selected_agent = self.agent_selection

    # Gymnasium Environment Methods
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment to the initial state.
        """
        self.seed = seed if seed is not None else self.seed
        self.game_state.reset(self.seed)

        self.agents = self.possible_agents.copy()
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # self._agent_selector = AgentSelector(self.agents)
        # self.agent_selection = self._agent_selector.next()
        self.agent_selection = 3 % len(
            self.agents
        )  # Start with a fixed agent for testing

        self.start_round(cards=options.get("cards") if options else None)
        return self.game_state, {}

    def close(self):
        pass

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent=None):
        """
        Get the space of all possible actions for the agent.
        """
        seed = self.seed + agent if self.seed is not None else None
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
        obs = self.state()
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
        obs["relative_dealer_position"] = float(
            (self.game_state.dealer_idx - agent)
            % self.config.num_players
            / self.config.num_players
        )
        return obs

    def state(self):
        """
        Get the observation visible to all agents.
        This includes the community cards, pot, current bet, and betting round.
        """
        obs = {
            "community_cards": [
                (
                    card_to_int(self.game_state.community_cards[i])
                    if i < len(self.game_state.community_cards)
                    else 0
                )
                for i in range(self.config.max_community_cards)
            ],
            "pot": self.game_state.pot / self.MAX_CHIPS,
            "current_bet": self.game_state.current_bet / self.MAX_CHIPS,
            "betting_round": self.game_state.betting_round.value,
            "chip_counts": [
                player.chips / self.MAX_CHIPS for player in self.game_state.players
            ],
            "bets": [player.bet / self.MAX_CHIPS for player in self.game_state.players],
            "total_contribution": [
                player.total_contribution / self.MAX_CHIPS
                for player in self.game_state.players
            ],
            "folded": [player.folded for player in self.game_state.players],
            "all_in": [player.all_in for player in self.game_state.players],
            "active": [player.idx in self.agents for player in self.game_state.players],
            "round_number": (
                self.game_state.round_number / self.config.max_rounds
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
        player = self.game_state.players[agent]
        return {
            "hand": [card_to_int(card) for card in player.hand],
        }

    def _get_action_mask(self, agent):
        """
        Mask actions based on the current game state.
        """
        player = self.game_state.players[agent]
        mask = [False] * len(Action)  # Initialize all actions as invalid
        min_raise = 0.0
        max_raise = 0.0

        # Check if player can "pass", an action that allows "skipping" their turn
        players_in_hand = [
            p
            for p in self.game_state.players
            if (p.idx in self.agents) and not p.folded and not p.all_in
        ]
        if (
            player.folded  # Folded players must "pass"
            or not player.idx in self.agents  # Non-agent players must "pass"
            or player.all_in  # All-in players must "pass"
            or (
                len(players_in_hand) == 1 and player.bet == self.game_state.current_bet
            )  # Last player must "pass"
            or agent
            != self.game_state.current_idx  # If the agent is not the current player, they must "pass"
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
        if player.bet == self.game_state.current_bet:
            mask[Action.CHECK.value] = True
        # Player's bet is less than the current bet. They can CALL or .
        elif player.bet < self.game_state.current_bet:
            # Player can call (or go all-in) if they have any chips left.
            if player.chips > 0:
                mask[Action.CALL.value] = True
        else:
            # This case should not happen in a valid game state.
            raise ValueError("Player bet exceeds current bet, inconsistent state.")

        # To raise, they must be able to at least match the call amount plus a minimum raise.
        ## An all-in raise for less than a min-raise is also possible
        to_raise = self.game_state.current_bet - player.bet + self.config.min_raise
        if player.chips >= to_raise and player.last_action != Action.RAISE:
            other_max = max(
                [
                    p.bet + p.chips
                    for p in self.game_state.players
                    if (p.idx in self.agents) and not p.folded and p.idx != player.idx
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

        min_raise /= self.MAX_CHIPS
        max_raise /= self.MAX_CHIPS

        return {
            "action_mask": {
                "action": np.array(mask, dtype=np.int8),
                "raise_amount": np.array([min_raise, max_raise], dtype=np.float32),
            }
        }

    def step(self, action: Dict):
        """
        Execute one time step within the environment.
        Must take null "pass" actions for agents if they cannot make a legal move or are about to be removed from the game.
        Input:
        - action: A dictionary containing the "action" and optional "raise_amount".
        """
        player = self.game_state.players[self.agent_selection]
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            if self.agent_selection == self.game_state.current_idx:
                self.game_state.current_idx = self.next_active_player_idx(
                    self.agent_selection
                )
            self.agent_selection = self.next_active_player_idx(self.agent_selection)
            # self.agent_selection = self._agent_selector.next()
            return

        if self.game_state.betting_round == BettingRound.END:
            # If last remaining player game must end
            active_players = [
                p.idx
                for p in self.game_state.players
                if not p.folded and not p.all_in and p.active
            ]
            if len(active_players) == 1 and self.agent_selection in active_players:
                self.terminations[self.agent_selection] = True
            if self.agent_selection == self.game_state.current_idx:
                self.game_state.current_idx = self.next_active_player_idx(
                    self.agent_selection
                )
            self.agent_selection = self.next_active_player_idx(self.agent_selection)

            # self.agent_selection = self._agent_selector.next()
            return

        # Take the Action
        _action = Action(action["action"])
        self._handle_action(player.idx, _action, action)

        if player.idx == 5 and self.game_state.betting_round == BettingRound.TURN:
            pass

        # Check if betting round is finished:
        active_players = [
            p
            for p in self.game_state.players
            if not p.folded and not p.all_in and p.active
        ]
        round_over = len(active_players) == 0
        if not round_over:
            round_over = True
            for p in active_players:
                if p.last_action is None or p.bet != self.game_state.current_bet:
                    round_over = False
                    break
        if not round_over:
            raisers = [p.idx for p in active_players if p.last_action == Action.RAISE]
            next_idk = self.next_active_player_idx(player.idx, check_termination=True)
            if next_idk in raisers and len(raisers) == 1:
                round_over = True

        if len(active_players) == 1:
            # Only one player remains, so skip to showdown
            while self.game_state.betting_round != BettingRound.SHOWDOWN:
                self.step_round()
        elif round_over:
            self.step_round()
        elif player.idx == self.game_state.current_idx:
            # If not all players have acted, continue to the next player
            self.game_state.current_idx = self.next_active_player_idx(player.idx)

        if self.game_state.betting_round == BettingRound.SHOWDOWN:
            self.end_round()
            if self.game_state.betting_round == BettingRound.START:
                self.start_round()

        self.agent_selection = self.next_active_player_idx(self.agent_selection)
        # self.agent_selection = self._agent_selector.next()
        return

    def _handle_action(self, agent: int, action_enum: Action, action: Dict):
        """
        Apply the action taken by the agent to the game state.
        Allows for all actions defined in the Action enum.
        Includes error handling for invalid actions.
        """
        player = self.game_state.players[agent]

        assert action_enum is not None, "Action cannot be None."

        if action_enum is Action.FOLD:
            player.folded = True
        elif action_enum is Action.CHECK:
            assert (
                player.bet == self.game_state.current_bet
            ), "Player can only check if their bet matches the current bet."
        elif action_enum is Action.CALL:
            assert player.chips > 0, "Player cannot call with no chips."
            to_call = min(self.game_state.current_bet - player.bet, player.chips)
            player.make_bet(to_call)
        elif action_enum is Action.RAISE:
            # Convert normalized raise amount to absolute chips
            extra_bet = action.get("raise_amount", 0.0)
            extra_chips = round(extra_bet * self.MAX_CHIPS)
            other_max = max(
                [
                    p.bet + p.chips
                    for p in self.game_state.players
                    if p.idx in self.agents and not p.folded and p.idx != player.idx
                ]
            )
            min_bet = self.game_state.current_bet + self.config.min_raise
            min_bet = min(min_bet, other_max)
            assert (
                player.bet + extra_chips >= min_bet
            ), "Player must raise at least the minimum raise amount."
            player.make_bet(extra_chips)
            self.game_state.current_bet = player.bet
        elif action_enum is Action.PASS:
            pass
        else:
            raise ValueError(f"Invalid action: {action} for player {agent}.")

        if not action_enum is Action.PASS:
            # Update the player's last action
            player.last_action = action_enum
        return True

    # Poker Logic
    def start_round(self, cards: Optional[dict[int, List[Card]]] = None):
        """
        Start a new round of poker, dealing cards and setting blinds.
        - If `cards` is provided (dict of player index and list of cards), it will use those cards instead of drawing from the deck.
        """
        assert (
            self.game_state.betting_round == BettingRound.START
        ), "Cannot start a new round before ending the current one."
        # Reset the game state for a new round
        self.game_state.reset_for_new_hand()
        for p in self.game_state.players:
            p.reset_for_new_hand()

        # Deal initial cards
        ## If cards are provided, use those cards for the players
        if cards:
            for i, _cards in cards.items():
                if len(_cards) != self.config.max_hand_cards:
                    raise ValueError(
                        f"Player {i} must receive exactly {self.config.max_hand_cards} cards."
                    )
                p = self.game_state.players[i]
                if not p.idx in self.agents:
                    continue
                p.hand = _cards
                for card in _cards:
                    self.game_state.deck.cards.remove(card)
        ## Otherwise, draw cards from the deck or draw to complete the hands
        for i in range(self.config.max_hand_cards * self.config.num_players):
            p = self.game_state.players[i % self.config.num_players]
            if len(p.hand) >= self.config.max_hand_cards:
                continue
            if not p.idx in self.agents:
                continue
            card = self.game_state.deck.draw(1)
            p.add_card(card)

        # Set blinds
        self.game_state.sb_idx = self.next_active_player_idx(
            self.game_state.dealer_idx, check_termination=True
        )

        sb_player = self.game_state.players[self.game_state.sb_idx]
        sb_bet = self.config.small_blind
        if sb_player.chips < sb_bet:
            sb_bet = sb_player.chips
            assert sb_bet > 0, "Small blind cannot be zero."
            sb_player.all_in = True
        sb_player.make_bet(sb_bet)

        self.game_state.bb_idx = self.next_active_player_idx(
            self.game_state.sb_idx, check_termination=True
        )
        bb_player = self.game_state.players[self.game_state.bb_idx]
        bb_bet = self.config.big_blind
        if bb_player.chips < bb_bet:
            bb_bet = bb_player.chips
            assert bb_bet > 0, "Big blind cannot be zero."
            bb_player.all_in = True
        bb_player.make_bet(bb_bet)

        # Set the current bet and betting round
        self.game_state.current_bet = max(sb_bet, bb_bet)
        self.game_state.betting_round = BettingRound.PREFLOP
        self.game_state.current_idx = self.next_active_player_idx(
            self.game_state.bb_idx, check_termination=True
        )

    def end_round(self):
        """
        End the current round of poker, determining the winner and distributing the pot.
        """
        # Find the winner(s) and distribute the pot
        assert (
            self.game_state.betting_round == BettingRound.SHOWDOWN
        ), "Cannot end round before showdown."
        scores = np.array(
            [
                self.evaluator.evaluate(player.hand, self.game_state.community_cards)
                for player in self.game_state.players
            ]
        )
        winners = set()
        pots = self._construct_pots()
        for pot_amount, player_indices in pots:
            # Determine the winners for this pot
            min_score = np.min(scores[player_indices])
            pot_winners = [
                self.game_state.players[i]
                for i in player_indices
                if scores[i] == min_score
            ]
            assert pot_winners, "No winners found for the pot."
            # Split the pot among the winners
            split_amount = pot_amount // len(pot_winners)
            for winner in pot_winners:
                winner.give_chips(split_amount)
                winners.add(winner.idx)
                self.game_state.pot -= split_amount
            remainder = pot_amount % len(pot_winners)
            payout_player = self.game_state.players[
                self.next_active_player_idx(self.game_state.dealer_idx)
            ]
            while remainder > 0:
                if payout_player in pot_winners:
                    payout_player.give_chips(1)
                    self.game_state.pot -= 1
                    remainder -= 1
                payout_player = self.game_state.players[
                    self.next_active_player_idx(payout_player.idx)
                ]
        assert (
            self.game_state.pot == 0
        ), f"Pot not fully distributed, remaining: {self.game_state.pot}"

        # Remove Players who are out of chips
        active_count = self.config.num_players
        for p in self.game_state.players:
            if p.chips == 0 and p.idx in self.agents:
                self.terminations[p.idx] = True
                active_count -= 1
            elif not p.idx in self.agents:
                active_count -= 1

        # Update Game State
        if active_count == 0:
            raise ValueError("All players are out of chips. Cannot continue the game.")
        elif active_count == 1:
            winners.add(
                next(p.idx for p in self.game_state.players if p.idx in self.agents)
            )
            winner = self.game_state.players[winners.pop()]
            self.terminations[winner.idx] = True
            self.game_state.betting_round = BettingRound.END
        else:
            self.game_state.betting_round = BettingRound.START
            self.game_state.round_number += 1
            self.game_state.dealer_idx = self.next_active_player_idx(
                self.game_state.dealer_idx, check_termination=True
            )

        if self.game_state.round_number == self.config.max_rounds:
            for agents in self.truncations:
                self.truncations[agents] = True
            self.game_state.betting_round = BettingRound.END

        return winners

    def step_round(self, cards: Optional[dict[BettingRound, List[Card]]] = None):
        """
        Advance the betting round to the next stage (Flop, Turn, River).
        - If `cards` is provided (dict of betting round and list of cards), it will use those cards instead of drawing from the deck.
        TODO: Make the number of rounds dynamic based on the config.
        """
        if self.game_state.betting_round == BettingRound.START:
            raise ValueError("Cannot step round before starting a round.")
        if self.game_state.betting_round == BettingRound.PREFLOP:
            self.game_state.betting_round = BettingRound.FLOP
            self._collect_bets()
            self._clear_actions()
            self.game_state.community_cards += self._draw_for_round(
                BettingRound.FLOP, cards
            )
            self.game_state.current_idx = self.nearest_active_player_idx(
                self.game_state.dealer_idx
            )
        elif self.game_state.betting_round == BettingRound.FLOP:
            self.game_state.betting_round = BettingRound.TURN
            self._collect_bets()
            self._clear_actions()
            self.game_state.community_cards += self._draw_for_round(
                BettingRound.TURN, cards
            )
            self.game_state.current_idx = self.nearest_active_player_idx(
                self.game_state.dealer_idx
            )
        elif self.game_state.betting_round == BettingRound.TURN:
            self.game_state.betting_round = BettingRound.RIVER
            self._collect_bets()
            self._clear_actions()
            self.game_state.community_cards += self._draw_for_round(
                BettingRound.RIVER, cards
            )
            self.game_state.current_idx = self.nearest_active_player_idx(
                self.game_state.dealer_idx
            )
        elif self.game_state.betting_round == BettingRound.RIVER:
            self._collect_bets()
            self._clear_actions()
            self.game_state.betting_round = BettingRound.SHOWDOWN

    def _collect_bets(self):
        """
        Collect bets from all active players.
        This is called at the end of each betting round to update the pot and player contributions.
        """
        for p in self.game_state.players:
            amount = p.bet
            if amount != self.game_state.current_bet:
                if (
                    (p.idx in self.agents)
                    and not p.folded
                    and not p.all_in
                    and self.terminations.get(p.idx, False)
                    and self.truncations.get(p.idx, False)
                ):
                    raise ValueError(
                        f"Player {p.idx} must bet {self.game_state.current_bet}, but bet {amount}."
                    )
            self.game_state.pot += amount
            p.bet = 0
        self.game_state.current_bet = 0

    def _clear_actions(self):
        """
        Clear the actions of all players at the end of a betting round.
        This is called to reset the state for the next betting round.
        """
        for p in self.game_state.players:
            p.last_action = None

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
                self.game_state.deck.cards.remove(card)
                drawn.append(card)

        # Draw from the deck if no / not enough cards were provided
        if len(drawn) < self.config.cards_per_round[round]:
            remaining = self.config.cards_per_round[round] - len(drawn)
            for _ in range(remaining):
                drawn.append(self.game_state.deck.draw())
        return drawn

    # Utility Methods
    def _rotate_to_idx(self, observation, idx):
        """
        Rotate an observation to be start at given idx.
        """
        return observation[idx:] + observation[:idx]

    def _find_active_player(
        self,
        start_idx: int,
        direction: int,
        include_start: bool,
        check_termination: bool = False,
    ) -> int:
        """
        Finds an active and non-folded player starting from `start_idx`, moving in `direction` (+1 or -1).
        If `include_start` is True, considers `start_idx` as the first valid candidate.
        """
        num_players = self.config.num_players
        offsets = range(num_players) if include_start else range(1, num_players + 1)

        for offset in offsets:
            idx = (start_idx + direction * offset) % num_players
            player = self.game_state.players[idx]
            active = player.idx in self.agents
            terminated = (
                self.terminations.get(player.idx, False)
                or self.truncations.get(player.idx)
                if check_termination
                else False
            )
            if active and not terminated:
                return idx
        return None

    def next_active_player_idx(self, idx, check_termination: bool = False):
        return self._find_active_player(
            idx, 1, include_start=False, check_termination=check_termination
        )

    def nearest_active_player_idx(self, idx, check_termination: bool = False):
        return self._find_active_player(
            idx, 1, include_start=True, check_termination=check_termination
        )

    def render(self):
        if self.render_mode == "terminal":
            terminal_render(self)

    def _construct_pots(self) -> List[Tuple[int, List[int]]]:
        """
        Construct pots based on player contributions (tracked by player.total_contribution).
        Returns a list of tuples, where each tuple contains the pot amount and a list of player indices who contributed to that pot.
        TODO: Handle edge cases like players going all-in and uneven contributions.
        """
        contributions = [
            (i, p.total_contribution, (p.idx in self.agents and not p.folded))
            for i, p in enumerate(self.game_state.players)
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


if __name__ == "__main__":
    import tyro

    # Define a wrapper to include optional seed
    @dataclass
    class ExtraArgs:
        config: PokerConfig
        seed: Optional[int] = 0  # Default seed value

    args = tyro.cli(ExtraArgs)
    env = PokerEnv(config=args.config, seed=args.seed)
    print(f"Environment: {env}")
