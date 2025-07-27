from dataclasses import dataclass, field
from typing import List, Optional

import gymnasium as gym
import numpy as np
from deuces import Card, Deck

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
    betting_round: BettingRound = BettingRound.PREFLOP

    def reset(self):
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.betting_round = BettingRound.PREFLOP
        for p in self.players:
            p.reset_for_new_hand()

class PokerEnv(gym.Env):
    def __init__(self, config: PokerConfig = PokerConfig()):
        super(PokerEnv, self).__init__()
        self.config = config

        players = [Player(player_id=i, chips=config.starting_stack) for i in range(config.num_players)]

        self.state = PokerGameState(
            players=players,
            dealer_idx=0,
            current_player_idx=0
        )