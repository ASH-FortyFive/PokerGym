from pokergym.env.utils import cards_pretty_str
from pokergym.env.cards import WORST_RANK
from typing import TYPE_CHECKING

from deuces import Evaluator

from pokergym.env.states import PokerGameState



def terminal_render(state: PokerGameState, eval: Evaluator=None) -> None:
    eval = Evaluator() if eval is None else eval
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
    print(f"Round: {state.hand_number + 1}, Betting: {state.betting_round.name}, Pot: {state.pot}, Player: {state.current_idx}")
    print(f"Community Cards: {cards_pretty_str(state.community_cards)}")
    print(header)
    print("-" * len(header))

    scores = []
    folded = []
    for p in state.players:
        folded.append(p.folded)
        if len(state.community_cards + p.hand) >= 5 and len(state.community_cards + p.hand) <= 7:
            hand_score = eval.evaluate(p.hand, state.community_cards)
        else:
            hand_score = WORST_RANK
        scores.append(hand_score)
    best_score = min(scores)

    for p in state.players:
        # Calculations per player

        line = ""
        for item, (width,align) in items.items():
            if item == "ID":
                id = f"{p.idx:{align}{width}}"
                if p.idx == state.current_idx:
                    line += make_green(id)
                elif not p.active:
                    line += make_red(id)
                else:
                    line += id
            elif item == "Out":
                if not p.active or p.folded or p.all_in:
                    check = make_red(f"{'✓':{align}{width}}")
                else:
                    check = f"{'✕':{align}{width}}"
                line += check
            elif item == "Chips":
                line += f"{p.chips:{align}{width}}"
            elif item == "Pos":
                if p.idx == state.sb_idx:
                    line += f"{'SB':^{width}}"
                elif p.idx == state.bb_idx:
                    line += f"{'BB':^{width}}"
                else:
                    line += f"{'':^{width}}"
            elif item == "Cards":
                line += f"{cards_pretty_str(p.hand):^{width}}"
            elif item == "Bet":
                line += f"{p.bet:{align}{width}}"
            elif item == "Contrib":
                line += f"{p.total_contribution:{align}{width}}"
            elif item == "F":
                if p.folded:
                    check = make_red(f"{'✓':{align}{width}}")
                else:
                    check = f"{'✕':{align}{width}}"
                line += check
            elif item == "All":
                if p.all_in:
                    check = make_red(f"{'✓':{align}{width}}")
                else:
                    check = f"{'✕':{align}{width}}"
                line += check
            elif item == "Action":
                if p.last_action is not None:
                    line += f"{p.last_action.name:{align}{width}}"
                else:
                    line += f"{'':{align}{width}}"
            elif item == "Score":
                if scores[p.idx] == WORST_RANK:
                    score = "N/A"
                else:
                    score = f"{scores[p.idx]}"
                score = f"{score:{align}{width}}"
                if scores[p.idx] == best_non_folded(scores, folded):
                    score = make_green(score)
                line += score
            elif item == "Hand":
                if scores[p.idx] == WORST_RANK:
                    hand = "N/A"
                else:
                    hand = eval.class_to_string(eval.get_rank_class(scores[p.idx]))
                hand = f"{hand:{align}{width}}"
                if scores[p.idx] == best_non_folded(scores, folded):
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