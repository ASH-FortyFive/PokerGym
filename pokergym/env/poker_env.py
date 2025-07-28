from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from deuces import Card, Deck, Evaluator

# import gymnasium as gym
from pettingzoo import ParallelEnv

from pokergym.env.cards import WORST_RANK, SeededDeck
from pokergym.env.config import PokerConfig
from pokergym.env.enums import Action, BettingRound
from pokergym.env.player import Player
from pokergym.env.utils import join_player_ids, short_pretty_str
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

    def reset_for_new_hand(self, seed):
        """
        Reset the game state for a new hand.
        This includes resetting player states, community cards, pot, and betting round.
        """
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.betting_round = BettingRound.START
        self.deck.seed(seed) if seed else None
        self.deck.shuffle()
        for player in self.players:
            player.reset_for_new_hand()

    def reset(self, seed: Optional[int] = None):
        """
        Fully reset the game state, including the deck.
        If a seed is provided, it will create a seeded deck for reproducibility.
        """
        self.reset_for_new_hand(seed)
        self.round_number = 0
        self.dealer_idx = 0
        self.current_player_idx = 0
        self.deck = SeededDeck(seed) if seed is not None else SeededDeck()


class PokerEnv(ParallelEnv):
    metadata = {
        "name": "PokerEnv_v0",
    }

    def __init__(self, config: PokerConfig = PokerConfig(), seed: Optional[int] = None):
        super(PokerEnv, self).__init__()

        self.config = config

        # Initialize the game state
        self.state = PokerGameState(
            players=[
                Player(idx=i, _chips=config.starting_stack)
                for i in range(config.num_players)
            ],
            active_players=config.num_players,
            dealer_idx=0,
            current_player_idx=0,
            deck=SeededDeck(seed) if seed is not None else SeededDeck(),
        )
        self.evaluator = Evaluator()

        #

    # Gymnasium Environment Methods
    def reset(self, seed: Optional[int] = None):
        """
        Reset the environment to the initial state.
        """
        self.state.reset(seed)
        return self.state, {}

    def step(self, actons):
        pass

    def observation_space(self, agent):
        return self.observation_space[agent]

    def action_space(self, agent):
        return self.action_space[agent]
    
    def _get_global_obs(self):
        """
        Get the observation visible to all agents.
        This includes the community cards, pot, current bet, and betting round.
        """
        obs = {
            "community_cards": self.state.community_cards,
            "pot": self.state.pot,
            "current_bet": self.state.current_bet,
            "betting_round": self.state.betting_round,
            "bets": [],
            "active": [],
            "all_in": [],
            "folded": [],
        }
        for i, player in enumerate(self.state.players):
            obs["bets"].append(player.total_contribution)
            obs["active"].append(player.active)
            obs["all_in"].append(player.all_in)
            obs["folded"].append(player.folded)


        return {
            "hand": player.hand,
            "community_cards": self.state.community_cards,
            "pot": self.state.pot,
            "current_bet": self.state.current_bet,
            "betting_round": self.state.betting_round,
            "player_idx": player.idx,
            "chips": player.chips,
            "active": player.active,
            "all_in": player.all_in,
        }

    def _get_agent_obs(self, agent):
        """
        Get the observation specific to the agent.
        This includes the agent's hand and the global observation.
        """
        player = self.state.players[agent]
        return {
            "hand": player.hand,
        }

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
        self.state.reset_for_new_hand(seed=None)
        self.state.deck.shuffle()
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
        TODO: Handle side pots.
        """
        # Find the winner(s) and distribute the pot
        assert (
            self.state.betting_round == BettingRound.SHOWDOWN
        ), "Cannot end round before showdown."
        scores = np.array([self.evaluator.evaluate(player.hand, self.state.community_cards) for player in self.state.players])
        pots = self._construct_pots()
        for pot_amount, player_indices in pots:
            # Determine the winners for this pot
            min_score = np.min(scores[player_indices])
            pot_winners = [self.state.players[i] for i in player_indices if scores[i] == min_score]
            assert pot_winners, "No winners found for the pot."
            # Split the pot among the winners
            split_amount = pot_amount // len(pot_winners)
            for winner in pot_winners:
                winner.give_chips(split_amount) 
                self.state.pot -= split_amount
                print(f"Player {winner.idx} wins {split_amount} chips from the pot of {pot_amount}.")
            remainder = pot_amount % len(pot_winners)
            payout_player = self.state.players[self.next_active_player_idx(self.state.dealer_idx)]
            while remainder > 0:
                if payout_player in pot_winners:
                    payout_player.give_chips(1)
                    self.state.pot -= 1
                    remainder -= 1
                    print(f"Player {payout_player.idx} receives an extra chip from the pot of {pot_amount}.")
                payout_player = self.state.players[self.next_active_player_idx(payout_player.idx)]
        assert self.state.pot == 0, f"Pot not fully distributed, remaining: {self.state.pot}"

        # Remove Players who are out of chips
        active_count = self.config.num_players
        for player in self.state.players:
            if player.chips == 0:
                player.active = False
                active_count -= 1
        self.state.active_players = active_count

        # Update Game State
        winner = None
        if active_count == 0:
            raise ValueError("All players are out of chips. Cannot continue the game.")
        elif active_count == 1:
            winner = next(player for player in self.state.players if player.active)
            self.state.betting_round = BettingRound.END
        else:
            self.state.betting_round = BettingRound.START
            self.state.round_number += 1
            self.state.dealer_idx = self.next_active_player_idx(self.state.dealer_idx)

        return winner

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
            self.state.community_cards += self._draw_for_round(BettingRound.FLOP, cards)
        elif self.state.betting_round == BettingRound.FLOP:
            self.state.betting_round = BettingRound.TURN
            self._collect_bets()
            self.state.community_cards += self._draw_for_round(BettingRound.TURN, cards)
        elif self.state.betting_round == BettingRound.TURN:
            self.state.betting_round = BettingRound.RIVER
            self._collect_bets()
            self.state.community_cards += self._draw_for_round(BettingRound.RIVER, cards)
        elif self.state.betting_round == BettingRound.RIVER:
            self._collect_bets()
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
    def next_active_player_idx(self, idx):
        idx = (idx + 1) % self.config.num_players
        while not self.state.players[idx].active:
            idx = (idx + 1) % self.config.num_players
        return idx

    def render(self):
        if self.render_mode == "terminal":
            terminal_render(self.state, self.evaluator)

    def _construct_pots(self) -> List[Tuple[int, List[int]]]:
        """
        Construct pots based on player contributions (tracked by player.total_contribution).
        Returns a list of tuples, where each tuple contains the pot amount and a list of player indices who contributed to that pot.
        """
        contributions = [
            (i, p.total_contribution, p.active and not p.folded)
            for i, p in enumerate(self.state.players)
            if p.total_contribution > 0
        ]
        contributions.sort(key=lambda x: x[1])
        pots = []
        while contributions:
            min_contribution = contributions[0][1] # Smallest contribution
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

