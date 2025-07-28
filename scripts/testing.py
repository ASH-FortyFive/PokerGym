import numpy as np
from deuces import Card, Deck, Evaluator

from pokergym.env.enums import BettingRound


def two_player_showdown():
    """
    Simulates a showdown between two players with random hands and community cards.
    """
    # Example usage of enums and classes
    deck = Deck()
    hand_1 =  sorted([deck.draw(1) for _ in range(2)])
    hand_2 =  sorted([deck.draw(1) for _ in range(2)])
    community_cards = sorted([deck.draw(1) for _ in range(5)])

    evaluator = Evaluator()
    score_1 = evaluator.evaluate(hand_1, community_cards)
    score_2 = evaluator.evaluate(hand_2, community_cards)
    class_1 = evaluator.get_rank_class(score_1)
    class_2 = evaluator.get_rank_class(score_2) 
    print(f"Community Cards:")
    Card.print_pretty_cards(community_cards)
    print(f"Hand 1, Score: {score_1}, Class: {evaluator.class_to_string(class_1)}")
    Card.print_pretty_cards(hand_1)
    print(f"Hand 2, Score: {score_2}, Class: {evaluator.class_to_string(class_2)}")
    Card.print_pretty_cards(hand_2)

    if score_1 < score_2:
        print("Hand 1 wins!")
    elif score_1 > score_2:
        print("Hand 2 wins!")
    else:
        print("It's a tie!")


if __name__ == "__main__":
    scores = np.array([1,1,200])
    print(f"Scores: {np.where(scores == scores.min())}")
