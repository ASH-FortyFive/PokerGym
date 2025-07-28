from deuces import Card

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
    

def short_pretty_str(cards):
    return ''.join([Card.int_to_pretty_str(card).replace(' ','') for card in cards])