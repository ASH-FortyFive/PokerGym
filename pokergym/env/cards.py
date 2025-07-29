# Wrappers around deuces Deck to allow for seeding and reproducibility
from random import Random
from deuces import Deck
from deuces import Card


WORST_RANK = 7463

class SeededDeck(Deck):
    """
    A seeded deck that allows for reproducible shuffling.
    """
    def __init__(self, seed=None):
        self._random = Random(seed)
        self.shuffle()

    def shuffle(self):
        # and then shuffle
        self.cards = Deck.GetFullDeck()
        self._random.shuffle(self.cards)

    def seed(self, seed):
        """
        Reset the random seed for reproducibility.
        """
        self._random.seed(seed)

    def __len__(self):
        return len(self.cards)

def card_to_int(card: Card) -> int:
    """
    Convert a deuces Card to low dim integer representation.
    0 is no card, 1-52 are the cards.
    1-13 are the ranks of Spades, 14-26 are Hearts,
    27-39 are Diamonds, and 40-52 are Clubs.
    """
    rank_int = Card.get_rank_int(card)
    suit_int = Card.get_suit_int(card) 
    suit_int = (suit_int & -suit_int).bit_length() - 1
    return rank_int + 13*suit_int + 1

def int_to_card(card_int: int) -> Card:
    """
    Convert a low dim integer representation to a deuces Card.
    0 is no card, 1-52 are the cards.
    1-13 are the ranks of Spades, 14-26 are Hearts,
    27-39 are Diamonds, and 40-52 are Clubs.
    """
    if card_int == 0:
        return None
    rank = (card_int - 1) % 13
    suit = (card_int - 1) // 13
    return Card.new(f"{Card.int_to_rank(rank)}{Card.int_to_suit(suit)}")    