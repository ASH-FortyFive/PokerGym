from pokergym.env.utils import short_pretty_str, join_player_ids

from deuces import evaluator

def terminal_render(state: "PokerGameState", eval:evaluator=None):
    if evaluator is None:
        eval = evaluator()
    header = f"{'ID':^3} | {'Curr':^4} |{'In':^4} | {'Pos':^3} | {'Chips':^6} | {'Hand':^8} | {'Bet':^6} | {'Total':^6} | {'F':^1} | {'All':^3} | {'Score':^5} | {'%':^6} | {'Hand':^20}"
    # print("=" * len(header))
    print(f"Round: {state.round_number}, Betting: {state.betting_round.name}, Pot: {state.pot}")
    print(f"Community Cards: {short_pretty_str(state.community_cards)}")
    print(header)
    print("-" * len(header))

    scores = []
    folded = []
    for player in state.players:
        i = player.idx
        print(f"{i:^3}", end=" | ")
        if player.idx == state.current_player_idx:
            print(f"{'✓':^4}", end=" | ")
        else:
            print(f"{' ':^4}", end=" | ")
        print(f"{('✓' if player.active else '✕'):^3}", end=" | ")
        if i == state.sb_idx:
            print(f"{'SB':^3}", end=" | ")
        elif i == state.bb_idx:
            print(f"{'BB':^3}", end=" | ")
        else:
            print(f"{'':^3}", end=" | ")
        print(f"{player.chips:>6}", end=" | ")
        print(f"{short_pretty_str(player.hand):^8}",end=" | ")
        print(f"{player.bet:>6}", end=" | ")
        print(f"{player.total_contribution:>6}", end=" | ")
        print(f"{'✓' if player.folded else '✕':^1}", end=" | ")
        print(f"{'✓' if player.all_in else '✕':^3}", end=" | ")


        folded.append(player.folded)

        if len(state.community_cards + player.hand) >= 5 and len(state.community_cards + player.hand) <= 7:
            hand_score = eval.evaluate(player.hand, state.community_cards)
            scores.append(hand_score)
            hand_class = eval.get_rank_class(hand_score)
            hand_class_str = eval.class_to_string(hand_class)
            percentage_rank = 1.0 - eval.get_five_card_rank_percentage(hand_score)
            print(f"{hand_score:^5}", end=" | ")
            print(f"{percentage_rank:>6.1%}", end=" | ")
            print(f"{hand_class_str:>}", end="")
        else:
            print(f"{'N/A':^5}", end=" | ")
            print(f"{'N/A':^6}", end=" | ")
            print(f"{'N/A'}", end="")
        print("")

    if len(scores) != 0:
        winning_score = min(scores)
        winning_indexes = [
            state.players[i]
            for i in range(len(state.players))
            if not folded[i] and scores[i] == winning_score
        ]
        winner_str = join_player_ids(winning_indexes)
        if len(winning_indexes) == 1:
            print(f"Hand Winner: {winner_str} with score {winning_score}")
        else:
            print(f"Hand Winners: {winner_str} with score {winning_score}")
    print("=" * len(header))

