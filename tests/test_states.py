import pytest
from pokergym.env.states import  PokerGameState, PlayerState
from random import random
SEEDS = [42, 0, 100]

# @pytest.mark.parametrize("seed", SEEDS)
# @pytest.mark.parametrize("expected", EXPECTIONS)
# def test_player(seed):
#     """
#     Test the legal_actions method of PokerLogic.
#     TODO: Finish this test
#     """
#     rng = random.seed(seed)
#     sample_player = PlayerState(
#         idx=rng.randint(0, 100),
#         chips=rng.randint(100, 1000),
#         hand=[0x02, 0x03],
#         folded=False,
#         all_in=False,
#         bet=50,
#         active=True
#     )
#     json_player = sample_player.to_json()
#     print("Player JSON:", json_player)
#     reconstructed_player = PlayerState.from_json(json_player)
#     print("Reconstructed Player:", reconstructed_player)
#     assert reconstructed_player == sample_player, "Reconstructed player does not match the original"

    
