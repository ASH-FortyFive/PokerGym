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
