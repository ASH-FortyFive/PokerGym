from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
from deuces import Card, Deck, Evaluator

from pokergym.enviroment.config import PokerConfig
from pokergym.enviroment.enums import Action, BettingRound
from pokergym.enviroment.player import Player


@dataclass
class PokerGameState:
    players: List[Player]
    community_cards: List[Card] = field(default_factory=list)
    pot: int = 0
    dealer_idx: int = 0
    current_player_idx: int = 0
    current_bet: int = 0
    betting_round: BettingRound = BettingRound.START
    round_number: int = 0

    def reset(self):
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.betting_round = BettingRound.START
        for p in self.players:
            p.reset_for_new_hand()

class PokerEnv(gym.Env):
    def __init__(self, config: PokerConfig = PokerConfig()):
        super(PokerEnv, self).__init__()
        self.config = config

        self.state = PokerGameState(
            players=[Player(player_id=i, _chips=config.starting_stack) for i in range(config.num_players)],
            dealer_idx=0,
            current_player_idx=0
        )

        self.deck = Deck()
        self.evaluator = Evaluator()

    # Gymnasium Environment Methods
    ## TODO
            
    # Poker Logic
    def start_round(self):
        """
        Start a new round of poker, dealing cards and setting blinds.
        """
        assert self.state.betting_round == BettingRound.START, "Cannot start a new round before ending the current one."
        # Reset the game state for a new round
        self.state.reset()
        self.deck = Deck()
        self.deck.shuffle()
        for player in self.state.players:
            player.reset_for_new_hand()
        
        # Deal initial cards
        for i in range(self.config.max_hand_cards * self.config.num_players):
            card = self.deck.draw(1)
            player = self.state.players[i % self.config.num_players]
            player.add_card(card)

        # Set blinds
        sb_player = self.state.players[(self.state.dealer_idx + 1) % self.config.num_players]
        sb_bet = self.config.small_blind
        if sb_player.chips < sb_bet:
            sb_bet = sb_player.chips
            sb_player.all_in = True
        sb_player.make_bet(sb_bet)
        self.state.pot += sb_bet

        bb_player = self.state.players[(self.state.dealer_idx + 2) % self.config.num_players]
        bb_bet = self.config.big_blind
        if bb_player.chips < bb_bet:
            bb_bet = bb_player.chips
            bb_player.all_in = True
        bb_player.make_bet(bb_bet)
        self.state.pot += bb_bet    

        # Set the current bet and betting round
        self.state.current_bet = max(sb_bet, bb_bet)
        self.state.betting_round = BettingRound.PREFLOP
        self.state.current_player_idx = (self.state.dealer_idx + 3) % self.config.num_players

    def end_round(self):
        """
        End the current round of poker, determining the winner and distributing the pot.
        TODO: Handle side pots.
        """
        # Find the winner(s) and distribute the pot
        assert self.state.betting_round == BettingRound.SHOWDOWN, "Cannot end round before showdown."
        scores = [self.evaluator.evaluate(player.hand, self.state.community_cards) for player in self.state.players]
        winning_indexes = [i for i, score in enumerate(scores) if not self.state.players[i].folded and score == min(scores)]
        for winning_index in winning_indexes:
            self.state.players[winning_index].give_chips(self.state.pot // len(winning_indexes))
        self.state.pot = 0

        # Remove Players who are out of chips
        self.state.players = [player for player in self.state.players if player.chips > 0]

        # Update Game State
        self.state.betting_round = BettingRound.START
        self.state.round_number += 1
        self.state.dealer_idx = (self.state.dealer_idx + 1) % self.config.num_players

        
        
    def step_round(self):
        """
        Advance the betting round to the next stage (Flop, Turn, River).
        TODO: Make number of community cards configurable / turns configurable.
        """
        if self.state.betting_round == BettingRound.START:
            raise ValueError("Cannot step round before starting a round.")
        if self.state.betting_round == BettingRound.PREFLOP:
            self.state.betting_round = BettingRound.FLOP
            self.deck.draw(1) # Burn a card
            self.state.community_cards = self.deck.draw(3)
        elif self.state.betting_round == BettingRound.FLOP:
            self.state.betting_round = BettingRound.TURN
            self.deck.draw(1) # Burn a card
            self.state.community_cards.append(self.deck.draw(1))
        elif self.state.betting_round == BettingRound.TURN:
            self.state.betting_round = BettingRound.RIVER
            self.deck.draw(1) # Burn a card
            self.state.community_cards.append(self.deck.draw(1))
        elif self.state.betting_round == BettingRound.RIVER:
            self.state.betting_round = BettingRound.SHOWDOWN

    # Utility Methods
    def render(self):
        if self.render_mode == "human":
            print("========================================")
            scores = []
            folded = []
            print(f"Current Betting Round: {self.state.betting_round.name}")
            print(f"Community Cards: {[Card.int_to_pretty_str(card) for card in self.state.community_cards]}")
            for player in self.state.players:
                print(f"Player {player.player_id} - Chips: {player.chips}, Hand: {[Card.int_to_pretty_str(card) for card in player.hand]}, Bet: {player.total_contribution}, Folded: {player.folded}, All In: {player.all_in}", end="")
                folded.append(player.folded)
                if len(self.state.community_cards + player.hand) >= 5 and len(self.state.community_cards + player.hand) <= 7:
                    hand_score = self.evaluator.evaluate(player.hand, self.state.community_cards)
                    scores.append(hand_score)
                    hand_class = self.evaluator.get_rank_class(hand_score)
                    hand_class_str = self.evaluator.class_to_string(hand_class)
                    print(f", Score: {hand_score}, Class: {hand_class_str}", end="")
                print("")
            if len(scores) != 0:
                winning_score = min(scores)
                winning_indexes = [self.state.players[i] for i in range(len(self.state.players)) if not folded[i] and scores[i] == winning_score]
                print(f"Winner(s): {[player.player_id for player in winning_indexes]} with score {winning_score}")