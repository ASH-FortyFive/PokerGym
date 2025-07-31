from typing import Dict
from deuces import Card
from pokergym.env.enums import Action, BettingRound


def join_player_ids(players):
    ids = [str(p.idx) for p in players]
    if len(ids) == 1:
        return ids[0]
    elif len(ids) == 2:
        return " and ".join(ids)
    elif len(ids) == 0:
        return ""
    else:
        return ", ".join(ids[:-1]) + ", and " + ids[-1]
    

def cards_pretty_str(cards):
    return ''.join([Card.int_to_pretty_str(card).replace(' ','') for card in cards])

def action_pretty_str(action_dict: Dict) -> str:
    if action_dict is None:
        return "None"
    action = Action(action_dict["action"])
    if action == Action.RAISE:
        raise_amount = action_dict["total_bet"]
        string = f"{action.name} {raise_amount:.2f}"
    else:
        string = action.name
    return string


def action_mask_pretty_str(action_mask_dict: Dict, max_chips:int = None) -> str:
    actions = [Action(i) for i, available in enumerate(action_mask_dict["action"]) if available]
    # masked_actions = [Action(i) for i, available in enumerate(action_mask_dict["action"]) if not available]
    raise_mask = action_mask_dict["total_bet"]
    if max_chips:
        raise_mask = [round(r * max_chips) for r in raise_mask]
    else:
        raise_mask = [float(r) for r in raise_mask]
    string = f"Available actions: {', '.join([a.name for a in actions])}, betting ranges from {raise_mask[0]} to {raise_mask[1]}"
    return string
    