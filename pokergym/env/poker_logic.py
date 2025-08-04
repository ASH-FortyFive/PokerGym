from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, TypedDict, Union

import numpy as np
from deuces import Card, Evaluator
from numpy.typing import NDArray

from pokergym.env.cards import WORST_RANK, SeededDeck
from pokergym.env.config import PokerConfig
from pokergym.env.enums import Action, BettingRound
from pokergym.env.states import PlayerState, PokerGameState
from pokergym.env.utils import action_pretty_str
from pokergym.visualise.terminal_vis import terminal_render


class ActionDict(TypedDict):
    action: Set[Action]
    total_bet: NDArray[np.int32]

class Poker:
    def __init__(
        self,
        config: PokerConfig = PokerConfig(),
        seed: Optional[int] = None,
        autorender: bool = False,
    ):
        """Initialize the Poker environment.
        Sets up the game with the specified configuration and optional seed for reproducibility.
        Args:
            config (PokerConfig): Configuration object defining game parameters (e.g., number of players, blinds).
            seed (Optional[int]): Seed for random number generation, passed to the game state.
        """
        self.config = config
        self.render_mode = config.render_mode
        self.autorender = autorender
        self.seed = seed

        # Initialize the game state
        self.game_state = PokerGameState(
            starting_stack=config.starting_chips,
            num_players=config.num_players,
            first_dealer=config.first_dealer,
            seed=self.seed,
        )
        self.MAX_CHIPS = self.config.starting_chips * self.config.num_players
        self.evaluator = Evaluator()

    # Environment Methods
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to its initial state and start a new hand.
        Resets the game state and optionally uses predetermined cards for testing purposes.
        Args:
            seed (Optional[int]): Seed for random number generation, overriding any previous seed.
            options (Optional[dict]): Dictionary with optional settings, such as 'cards' mapping hand numbers
                                      to predetermined card assignments.
        """
        self.seed = seed if seed is not None else self.seed
        self.game_state.reset(self.seed)
        self._cards = options.get("cards", {}) if options else {}
        self.game_state.reset_for_new_hand()
        self.start_hand()
        self.render() if self.autorender else None
        return

    # Action Methods
    def legal_actions(self, player: PlayerState = None, idx: int = None) -> ActionDict:
        """Determine the legal actions available to the specified player in the current game state.
        Returns an `ActionDict` containing a set of possible actions and, for RAISE, a range of allowed
        total bet amounts. Ensures the player is the current actor.
        Args:
            player (Player): The player to query for legal actions [or idx]
            idx (int): The index of the player to query for legal actions [or player]
        Returns:
            ActionDict: Dictionary with:
                - 'action' (Set[Action]): Possible actions (e.g., FOLD, CHECK, CALL, RAISE, PASS).
                - 'total_bet' (NDArray[np.int32]): For RAISE, a 2-element array [min_total_bet, max_total_bet];
                                                    otherwise, [0, 0].
        Raises:
            ValueError: If the player is not the current player.
        """
        assert (
            player is not None or idx is not None
        ), "Player or index must be provided."
        player = player if player is not None else self.game_state.players[idx]
        legal_actions = ActionDict(
            action=set(), total_bet=np.array([0, 0], dtype=np.int32)
        )
        if player.idx != self.game_state.current_idx:
            # print(f"Player {player.idx} is not the current player. Current player is {self.game_state.current_idx}.")
            return legal_actions  # Not the current player, no legal actions

        if player.folded or player.all_in or not player.active:
            legal_actions["action"].add(Action.PASS)
            return legal_actions

        to_call = self.game_state.current_bet - player.bet
        if to_call == 0:
            legal_actions["action"].add(Action.CHECK)
        if to_call > 0:
            legal_actions["action"].add(Action.CALL)
            legal_actions["action"].add(Action.FOLD)
        if to_call < player.chips:
            min_raise_amount = self.game_state.current_bet + self.config.min_raise
            total_bet_if_all_in = player.bet + player.chips
            if total_bet_if_all_in >= min_raise_amount:
                legal_actions["action"].add(Action.RAISE)
                legal_actions["total_bet"] = np.array(
                    [min_raise_amount, total_bet_if_all_in], dtype=np.int32
                )
        return legal_actions

    def take_action(
        self, player: PlayerState = None, action_dict: ActionDict = None, idx: int = None, 
    ):
        """Execute the player's chosen action and update the game state accordingly.
        Processes the action specified in `action_dict`, ensuring it's valid for the current player.
        Updates player chips, bet amounts, and status (e.g., folded, all-in) as needed.
        Args:
            player (Player): The player taking the action [or idx].
            idx (int): The index of the player taking the action [or player].
            action_dict (ActionDict): Dictionary containing:
                - 'action' (Action): The chosen action (e.g., FOLD, CHECK, CALL, RAISE, PASS).
                - 'total_bet' (NDArray[np.int32]): For RAISE, the specific total bet amount chosen,
                                                    which must be within the range from `legal_actions`.
        """
        assert (
            player is not None or idx is not None
        ), "Player or index must be provided."
        assert action_dict is not None, "Action dictionary must be provided."
        player = player if player is not None else self.game_state.players[idx]
        if player.idx != self.game_state.current_idx:
            raise ValueError(
                f"Player {player.idx} is not the current player. Current player is {self.game_state.current_idx}."
            )
        action = action_dict["action"]
        if action == Action.PASS:
            pass
        elif action == Action.FOLD:
            player.folded = True
            player.last_action = Action.FOLD
        elif action == Action.CHECK:
            assert player.bet == self.game_state.current_bet
            player.last_action = Action.CHECK
        elif action == Action.CALL:
            to_call = self.game_state.current_bet - player.bet
            to_call = min(to_call, player.chips)
            player.make_bet(to_call)
            self.game_state.current_bet = max(player.bet, self.game_state.current_bet)
            player.last_action = Action.CALL
        elif action == Action.RAISE:
            total_bet = action_dict["total_bet"]
            raise_amount = total_bet - player.bet
            assert (
                total_bet >= self.game_state.current_bet + self.config.min_raise
            ), "Player must raise at least the minimum raise amount."
            player.make_bet(raise_amount)
            self.game_state.current_bet = player.bet
            player.last_action = Action.RAISE
        else:
            raise ValueError(f"Invalid action: {action}. Must be one of {Action}.")

        return

    # Start Hand Methods
    def start_hand(self):
        """Start a new hand of poker.
        Resets the game state for a new hand, deals cards to players (using predetermined cards if provided),
        sets the small and big blinds, and advances the betting round to PREFLOP.
        """
        assert (
            self.game_state.betting_round == BettingRound.START
        ), "Cannot start a new round before ending the current one."
        # Deal initial cards
        predetermined_cards = self._get_predetermined_cards()
        if predetermined_cards:
            ## If given use predetermined cards
            for p_idx, cards in predetermined_cards.items():
                assert (
                    len(cards) == self.config.max_hand_cards
                ), f"Player {p_idx} must receive at most {self.config.max_hand_cards} cards."
                if p_idx >= len(self.game_state.players):
                    continue  # Skip if player index is out of range
                p = self.game_state.players[p_idx]
                if not p.active:
                    continue
                p.hand = cards
                for card in cards:
                    self.game_state.deck.cards.remove(card)
        # Draw from deck if no predetermined cards were given or not enough cards
        for p in self.game_state.players:
            if len(p.hand) >= self.config.max_hand_cards:
                continue
            if not p.active:
                continue
            while len(p.hand) < self.config.max_hand_cards:
                card = self.game_state.deck.draw(1)
                p.add_card(card)

        # Set blinds
        self.game_state.sb_idx = self.next_idx(self.game_state.dealer_idx)

        sb_player = self.game_state.players[self.game_state.sb_idx]
        sb_bet = self.config.small_blind
        if sb_player.chips < sb_bet:
            sb_bet = sb_player.chips
            assert sb_bet > 0, "Small blind cannot be zero."
        sb_player.make_bet(sb_bet)

        self.game_state.bb_idx = self.next_idx(self.game_state.sb_idx)
        bb_player = self.game_state.players[self.game_state.bb_idx]
        bb_bet = self.config.big_blind
        if bb_player.chips < bb_bet:
            bb_bet = bb_player.chips
            assert bb_bet > 0, "Big blind cannot be zero."
        bb_player.make_bet(bb_bet)

        # Set the current bet and betting round
        self.game_state.current_bet = max(sb_bet, bb_bet)
        self.game_state.betting_round = BettingRound.PREFLOP
        self.game_state.current_idx = self.next_idx(self.game_state.bb_idx)

    def _get_predetermined_cards(self) -> Union[List[Card], dict[int, List[Card]]]:
        """Retrieve predetermined cards for the current hand and betting round, if specified.
        Returns:
            Union[List[Card], dict[int, List[Card]]]:
                - For BettingRound.START: A dict mapping player indices to their hole cards.
                - For other rounds: A list of community cards for the current round.
                - None: If no predetermined cards are set for the current hand and round.
        """
        cards_for_hand = self._cards.get(self.game_state.hand_number)
        if cards_for_hand:
            return cards_for_hand.get(self.game_state.betting_round, None)

    # End of Hand Methods
    def end_hand(self) -> dict[int, int]:
        """Conclude the current hand and distribute the pot.
        Determines the winner(s) based on remaining players or hand evaluation, handles pot distribution
        (including side pots), and increments the hand number.
        Returns:
            dict[int, int]: Dictionary mapping player indices to their chip deltas after the hand ends.
        """
        assert (
            self.game_state.betting_round == BettingRound.SHOWDOWN
        ), "Cannot end hand before reaching SHOWDOWN."
        # Collect bets and contributions
        pots = self._construct_pots()
        player_deltas = {p.idx: -p.total_contribution for p in self.game_state.players}
        players_in_hand = [
            p for p in self.game_state.players if not p.folded and p.active
        ]
        if len(players_in_hand) == 1:
            # If only one player is left, they win the pot
            winner = players_in_hand[0]
            winner.chips += self.game_state.pot
            player_deltas[winner.idx] += self.game_state.pot
        else:
            scores = np.array(
                [
                    (
                        self.evaluator.evaluate(p.hand, self.game_state.community_cards)
                        if not p.folded and p.active
                        else WORST_RANK
                    )
                    for p in self.game_state.players
                ]
            )
            for pot_amount, player_indices in pots:
                assert (
                    player_indices
                ), "There must be at least one player contributing to the pot."
                # Find the best score among the players who contributed to this pot
                best_score = min(scores[player_indices])
                winners = [i for i in player_indices if scores[i] == best_score]
                split_deltas = self._split_pot(pot_amount, winners)
                # Merge deltas into the main player_deltas
                for idx, delta in split_deltas.items():
                    self.game_state.players[idx].chips += delta
                    player_deltas[idx] += delta
        self.game_state.hand_number += 1
        self.game_state.reset_for_new_hand()
        return player_deltas

    def _split_pot(self, pot_amount: int, winners: List[int]) -> dict[int, int]:
        """Distribute a pot amount equally among the winning players.
        Divides the pot evenly, distributing any remainder one chip at a time starting from the dealer.
        Args:
            pot_amount (int): Total chips in the pot to be split.
            winners (List[int]): Indices of the winning players.
        Returns:
            dict[int, int]: Dictionary mapping player indices to their chip deltas after the split.
        """
        player_deltas = {p.idx: 0 for p in self.game_state.players}
        if not winners:
            return
        share = pot_amount // len(winners)
        remainder = pot_amount % len(winners)
        for winner_idx in winners:
            player_deltas[winner_idx] += share

        remainder = pot_amount % len(winners)
        if remainder > 0:
            ordered_winners = sorted((idx, i) for i, idx in enumerate(winners))
            start_i = min(
                (i for idx, i in ordered_winners if idx >= self.game_state.dealer_idx),
                default=ordered_winners[0][1],
            )
            for i in range(remainder):
                idx = winners[(start_i + i) % len(winners)]
                player_deltas[idx] += 1
        return player_deltas

    def _construct_pots(self) -> List[Tuple[int, List[int]]]:
        """Construct the main pot and any side pots based on player contributions.
        Analyzes player bets to create a list of pots, each with an amount and eligible players.
        Returns:
            List[Tuple[int, List[int]]]: List of tuples, each containing:
                - Pot amount (int): Total chips in the pot.
                - Player indices (List[int]): Indices of players eligible to win the pot.
        """
        contributions = [
            (i, p.total_contribution, (p.active and not p.folded))
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

    # Stepping Methods
    def step_round(self):
        """Advance the betting round to the next stage (e.g., PREFLOP to FLOP).
        Collects bets, clears player actions, draws community cards (using predetermined cards if provided),
        and sets the next player to act.
        TODO: Make the number of rounds dynamic based on the config.
        """
        if self.game_state.betting_round == BettingRound.START:
            raise ValueError("Cannot step round before starting a round.")
        if self.game_state.betting_round == BettingRound.PREFLOP:
            self.game_state.betting_round = BettingRound.FLOP
            self._collect_bets()
            self._clear_actions()
            self.game_state.community_cards += self._draw_for_round(BettingRound.FLOP)
            self.game_state.current_idx = self.near_idx(self.game_state.dealer_idx)
        elif self.game_state.betting_round == BettingRound.FLOP:
            self.game_state.betting_round = BettingRound.TURN
            self._collect_bets()
            self._clear_actions()
            self.game_state.community_cards += self._draw_for_round(BettingRound.TURN)
            self.game_state.current_idx = self.near_idx(self.game_state.dealer_idx)
        elif self.game_state.betting_round == BettingRound.TURN:
            self.game_state.betting_round = BettingRound.RIVER
            self._collect_bets()
            self._clear_actions()
            self.game_state.community_cards += self._draw_for_round(BettingRound.RIVER)
            self.game_state.current_idx = self.near_idx(self.game_state.dealer_idx)
        elif self.game_state.betting_round == BettingRound.RIVER:
            self._collect_bets()
            self._clear_actions()
            self.game_state.betting_round = BettingRound.SHOWDOWN

    def _collect_bets(self):
        """Collect bets from all active players and add them to the pot.
        Ensures all active players have matched the current bet or are all-in, then resets individual bets.
        Raises:
            ValueError: If an active, non-folded, non-all-in player hasn't matched the current bet.
        """
        for p in self.game_state.players:
            amount = p.bet
            if amount != self.game_state.current_bet:
                if p.active and not p.folded and not p.all_in:
                    raise ValueError(
                        f"Player {p.idx} must bet {self.game_state.current_bet}, but bet {amount}."
                    )
            self.game_state.pot += amount
            p.bet = 0
        self.game_state.current_bet = 0

    def _clear_actions(self):
        """Reset the `last_action` attribute for all players.
        Prepares the game state for a new betting round by clearing previous actions.
        """
        for p in self.game_state.players:
            p.last_action = None

    def _draw_for_round(self, betting_round: BettingRound):
        """Draw community cards for the specified betting round.
        Burns one card before drawing, uses predetermined cards if provided, and draws from the deck as needed.
        Args:
            betting_round (BettingRound): The current betting round (e.g., FLOP, TURN).
        Returns:
            List[Card]: The drawn community cards for the round.
        """
        if betting_round not in self.config.cards_per_round:
            raise ValueError(f"Invalid betting round: {betting_round}")
        # Draw predetermined cards if provided
        drawn = []
        burn = None
        cards = self._get_predetermined_cards()
        if cards:
            if len(cards) > self.config.cards_per_round[betting_round] + 1:
                raise ValueError(
                    f"Expected at most {self.config.cards_per_round[betting_round]} cards for {betting_round}, got {len(cards)}."
                )
            for card in cards:
                if burn is None:
                    self.game_state.deck.cards.remove(card)
                    burn = card
                    continue
                self.game_state.deck.cards.remove(card)
                drawn.append(card)

        # Draw from the deck if no / not enough cards were provided
        if burn is None:
            burn = self.game_state.deck.draw(1)
        if len(drawn) < self.config.cards_per_round[betting_round]:
            remaining = self.config.cards_per_round[betting_round] - len(drawn)
            for _ in range(remaining):
                drawn.append(self.game_state.deck.draw())
        return drawn

    # Core Step Methods
    def step(
        self, player: PlayerState = None, action_dict: ActionDict = None, idx: int = None, 
    ) -> bool:
        """Advance the game by executing the player’s action and updating the game state.
        Handles the action, checks for round completion, advances rounds or ends the hand as needed,
        and starts a new hand if the game continues. Ends the game if only one player remains or the
        maximum hand limit is reached.
        Args:
            player (Player): The player taking the action [or idx].
            idx (int): The index of the player taking the action [or player].
            action_dict (ActionDict): The player’s chosen action and, if applicable, total bet amount.
        Returns:
            Optional[Dict[int,int]]: Dictionary containing the chip deltas for each player if the hand ends,
                            or None if the game continues.
        """
        assert (
            player is not None or idx is not None
        ), "Player or index must be provided."
        assert action_dict is not None, "Action dictionary must be provided."
        player = player if player is not None else self.game_state.players[idx]
        # Take Actions
        self.take_action(player=player, action_dict=action_dict)
        if self.autorender:
            self.render()
            print(f"{player.idx}: {action_pretty_str(action_dict)}")
        self.game_state.current_idx = self.next_idx(self.game_state.current_idx)
        while (
            self._is_end_of_round()
            and self.game_state.betting_round != BettingRound.SHOWDOWN
        ):
            self.step_round()
            self.render() if self.autorender else None

        # End the hand if the betting round is SHOWDOWN
        if self.game_state.betting_round == BettingRound.SHOWDOWN:
            player_deltas = self.end_hand()
            self.render() if self.autorender else None
            # Deactivate chipless players
            for p in self.game_state.players:
                if p.chips == 0:
                    p.active = False
            remaining_players = [p for p in self.game_state.players if p.active]

            # Check if the game
            if (
                len(remaining_players) == 1
                or self.game_state.hand_number == self.config.max_hands
            ):
                self.game_state.betting_round = BettingRound.END
            else:
                self.game_state.betting_round = BettingRound.START
                self.game_state.dealer_idx = self.next_idx(self.game_state.dealer_idx)
                self.start_hand()
                self.render() if self.autorender else None
            return player_deltas
        return None

    def _is_end_of_round(self):
        """Determine if the current betting round has concluded.
        Returns True if all active players have matched the current bet or are all-in, or if only
        one active player remains.
        Returns:
            bool: True if the round has ended, False otherwise.
        """
        active_players = [
            p for p in self.game_state.players if p.active and not p.folded
        ]
        assert active_players, "No active players left in the game."
        if len(active_players) <= 1:
            return True
        for p in active_players:
            if not p.all_in:
                if p.bet < self.game_state.current_bet or p.last_action is None:
                    return False  # Needs to call or raise or fold
        return True

    # Utility Methods
    def game_over(self) -> bool:
        """Check if the game is over.
        The game is considered over if the betting round is END.
        Returns:
            bool: True if the game is over, False otherwise.
        """
        return self.game_state.betting_round == BettingRound.END

    def _find_active_player(
        self,
        start_idx: int,
        direction: int,
        include_start: bool,
    ) -> int:
        """Find the next active player starting from a given index.
        Searches in the specified direction, optionally including the starting index.
        Args:
            start_idx (int): Index to start the search from.
            direction (int): Direction to search (1 for forward, -1 for backward).
            include_start (bool): Whether to consider the starting index as a candidate.
        Returns:
            int: Index of the next active player, or None if no active players are found.
        """
        num_players = self.config.num_players
        offsets = range(num_players) if include_start else range(1, num_players + 1)

        for offset in offsets:
            idx = (start_idx + direction * offset) % num_players
            p = self.game_state.players[idx]
            if p.active:
                return idx
        return None

    def next_idx(self, idx, **kwargs):
        """Get the index of the next active player after the given index.
        Args:
            idx (int): Starting index.
            **kwargs: Additional arguments passed to `_find_active_player`.
        Returns:
            int: Index of the next active player.
        """
        return self._find_active_player(idx, 1, include_start=False, **kwargs)

    def near_idx(self, idx, **kwargs):
        """Get the index of the nearest active player starting from the given index.
        Includes the starting index if the player is active.
        Args:
            idx (int): Starting index.
            **kwargs: Additional arguments passed to `_find_active_player`.
        Returns:
            int: Index of the nearest active player.
        """
        return self._find_active_player(idx, 1, include_start=True, **kwargs)

    def render(self):
        """Render the current game state.
        Displays the game state in the specified render mode, currently supporting 'terminal' mode.
        """
        if self.render_mode == "terminal":
            terminal_render(self.game_state, eval=self.evaluator)


if __name__ == "__main__":
    from pokergym.env.config import PokerConfig

    def make_ad(
        action: Action, total_bet: Optional[NDArray[np.int32]] = None
    ) -> ActionDict:
        """Helper function to create an ActionDict."""
        return ActionDict(
            action=action,
            total_bet=(
                total_bet if total_bet is not None else np.array([0], dtype=np.int32)
            ),
        )

    config = PokerConfig(
        num_players=6, starting_chips=1000, small_blind=10, big_blind=20
    )
    poker = Poker(config=config, seed=42, autorender=True)
    poker.reset()

    players = poker.game_state.players
    poker.step(players[3], make_ad(Action.CALL))
    poker.step(players[4], make_ad(Action.CALL))
    poker.step(players[5], make_ad(Action.RAISE, 40))
    poker.step(players[0], make_ad(Action.CALL))
    poker.step(players[1], make_ad(Action.FOLD))
    poker.step(players[2], make_ad(Action.FOLD))
    poker.step(players[3], make_ad(Action.CALL))
    poker.step(players[4], make_ad(Action.CALL))
    poker.step(players[0], make_ad(Action.CHECK))
    poker.step(players[1], make_ad(Action.PASS))
    poker.step(players[2], make_ad(Action.PASS))
    poker.step(players[3], make_ad(Action.CHECK))
    poker.step(players[4], make_ad(Action.CHECK))
    poker.step(players[5], make_ad(Action.CHECK))
    poker.step(players[0], make_ad(Action.CHECK))
    poker.step(players[1], make_ad(Action.PASS))
    poker.step(players[2], make_ad(Action.PASS))
    poker.step(players[3], make_ad(Action.CHECK))
    poker.step(players[4], make_ad(Action.RAISE, 60))
    poker.step(players[5], make_ad(Action.FOLD))
    poker.step(players[0], make_ad(Action.FOLD))
    poker.step(players[1], make_ad(Action.PASS))
    poker.step(players[2], make_ad(Action.PASS))
    poker.step(players[3], make_ad(Action.CALL))
    poker.step(players[0], make_ad(Action.PASS))
    poker.step(players[1], make_ad(Action.PASS))
    poker.step(players[2], make_ad(Action.PASS))
    poker.step(players[3], make_ad(Action.CHECK))
    poker.step(players[4], make_ad(Action.CHECK))

    # Round 2
    poker.step(players[4], make_ad(Action.RAISE, 1210))
    poker.step(players[5], make_ad(Action.FOLD))
    poker.step(players[0], make_ad(Action.CALL))
    poker.step(players[1], make_ad(Action.CALL))
    poker.step(players[2], make_ad(Action.FOLD))
    poker.step(players[3], make_ad(Action.FOLD))
    print("Ending hand...")
