from pokergym.env.utils import cards_pretty_str, join_player_ids
from pokergym.env.cards import WORST_RANK
from rich.console import Console

from deuces import evaluator



def terminal_render(state: "PokerGameState", eval:evaluator=None):
    console = Console()
    if evaluator is None:
        eval = evaluator()
    # header = f"{'ID':^3} | {'Curr':^4} |{'In':^4} | {'Pos':^3} | {'Chips':^6} | {'Hand':^8} | {'Bet':^6} | {'Total':^6} | {'F':^1} | {'All':^3} | {'Action':>6} | {'Score':^5} | {'%':^6} | {'Hand':^20}"

    items = {
        "ID": (3, "^"),
        "Out": (3, "^"),
        "Chips": (6, ">"),
        "Pos": (3, "^"),
        "Cards" : (8, "^"),
        "Bet": (6, ">"),
        "Contrib" : (7, ">"),
        "F": (1, "^"),
        "All": (3, "^"),
        "Action": (6, ">"),
        "Score": (5, "^"),
        "Hand": (20, "<")
    }
    header = " | ".join([f"{item:^{width}}" for item, (width, _) in items.items()])

    # header = f"{'ID':^3} |{'In':^4} | {'Chips':^6} | {'Pos':^3} | {'Cards':^8} | {'Bet':^6} | {'Total':^6} | {'F':^1} | {'All':^3} | {'Action':>6} | {'Score':^5} | {'Hand':^20}"
    # print("=" * len(header))
    print(f"Round: {state.round_number}, Betting: {state.betting_round.name}, Pot: {state.pot}")
    print(f"Community Cards: {cards_pretty_str(state.community_cards)}")
    print(header)
    print("-" * len(header))

    scores = []
    folded = []
    for player in state.players:
        folded.append(player.folded)
        if len(state.community_cards + player.hand) >= 5 and len(state.community_cards + player.hand) <= 7:
            hand_score = eval.evaluate(player.hand, state.community_cards)
        else:
            hand_score = WORST_RANK
        scores.append(hand_score)
    best_score = min(scores)

    for player in state.players:
        # Calculations per player

        line = ""
        for item, (width,align) in items.items():
            if item == "ID":
                id = f"{player.idx:{align}{width}}"
                if player.idx == state.current_player_idx:
                    line += make_green(id)
                elif not player.active:
                    line += make_red(id)
                else:
                    line += id
            elif item == "Out":
                if not player.active or player.folded or player.all_in:
                    check = make_red(f"{'✓':{align}{width}}")
                else:
                    check = f"{'✕':{align}{width}}"
                line += check
            elif item == "Chips":
                line += f"{player.chips:{align}{width}}"
            elif item == "Pos":
                if player.idx == state.sb_idx:
                    line += f"{'SB':^{width}}"
                elif player.idx == state.bb_idx:
                    line += f"{'BB':^{width}}"
                else:
                    line += f"{'':^{width}}"
            elif item == "Cards":
                line += f"{cards_pretty_str(player.hand):^{width}}"
            elif item == "Bet":
                line += f"{player.bet:{align}{width}}"
            elif item == "Contrib":
                line += f"{player.total_contribution:{align}{width}}"
            elif item == "F":
                if player.folded:
                    check = make_red(f"{'✓':{align}{width}}")
                else:
                    check = f"{'✕':{align}{width}}"
                line += check
            elif item == "All":
                if player.all_in:
                    check = make_red(f"{'✓':{align}{width}}")
                else:
                    check = f"{'✕':{align}{width}}"
                line += check
            elif item == "Action":
                if player.last_action is not None:
                    line += f"{player.last_action.name:{align}{width}}"
                else:
                    line += f"{'':{align}{width}}"
            elif item == "Score":
                if scores[player.idx] == WORST_RANK:
                    score = "N/A"
                else:
                    score = f"{scores[player.idx]}"
                score = f"{score:{align}{width}}"
                if scores[player.idx] == best_non_folded(scores, folded):
                    score = make_green(score)
                line += score
            elif item == "Hand":
                if scores[player.idx] == WORST_RANK:
                    hand = "N/A"
                else:
                    hand = eval.class_to_string(eval.get_rank_class(scores[player.idx]))
                hand = f"{hand:{align}{width}}"
                if scores[player.idx] == best_non_folded(scores, folded):
                    hand = make_green(hand)
                line += hand
            # If not last, add separator
            if item != list(items.keys())[-1]:
                line += " | "
        print(line)


    # if len(scores) != 0:
    #     winning_score = min(scores)
    #     winning_indexes = [
    #         state.players[i]
    #         for i in range(len(state.players))
    #         if not folded[i] and scores[i] == winning_score
    #     ]
    #     winner_str = join_player_ids(winning_indexes)
    #     if len(winning_indexes) == 1:
    #         print(f"Hand Winner: {winner_str} with score {winning_score}")
    #     else:
    #         print(f"Hand Winners: {winner_str} with score {winning_score}")

def make_bold(line: str) -> str:
    return f"\033[1m{line}\033[0m"

def make_red(line: str) -> str:
    return f"\033[91m{line}\033[0m"

def make_green(line: str) -> str:
    return f"\033[92m{line}\033[0m"

def best_non_folded(scores, folded):
    return min((score for score, f in zip(scores, folded) if not f), default=WORST_RANK)