def dummy():
    print("Here we need to change the action name + args based on the direction of the move")

import re

def parse_atom(atom):
    s = str(atom)
    m = re.match(r'(\w+)\((.*)\)', s)
    if not m:
        return None, []
    pred = m.group(1)
    args = [a.strip() for a in m.group(2).split(',')]
    return pred, args

def extract_info(state):
    player_pos = None              # 'pos-i-j:location'
    stones = {}                    # stone_name -> pos
    goals = set()                  # set of pos
    nongoals = set()

    for lit in state.literals:
        pred, args = parse_atom(lit)
        if pred is None:
            continue

        if pred == "at":
            obj, loc = args
            if obj.startswith("player-"):
                player_pos = loc
            elif obj.startswith("stone-"):
                stones[obj] = loc

        elif pred == "is-goal":
            (loc,) = args
            goals.add(loc)

        elif pred == "is-nongoal":
            (loc,) = args
            nongoals.add(loc)

    return player_pos, stones, goals, nongoals

from lib.rendering.sokoban.sokoban_render import build_layout
import numpy as np

def detect_action(prev_state, curr_state):

    # extract information
    prev_l = build_layout(prev_state.literals)
    new_l = build_layout(curr_state.literals)

    # get x and y position of players
    px, py = np.argwhere(prev_l == 1)[0].tolist()
    nx, ny = np.argwhere(new_l == 1)[0].tolist()

    # get dimension
    n = prev_l.shape[0]



    # case division based on movement
    op1_x, op1_y = nx, ny
    op2_x, op2_y = nx, ny
    direction = ""
    if px == nx and py + 1 == ny:
        op2_y = ny+1
        direction = "right"
    elif px == nx and py - 1 == ny:
        op2_y = ny-1
        direction = "left"
    elif px + 1 == nx and py == ny:
        op2_x = nx+1
        direction = "down"
    elif px - 1 == nx and py == ny:
        op2_x = nx-1
        direction = "up"
    else:
        raise ValueError("No valid sokoban action detected between states.")


    
    VERSION_TO_CHOOSE = "NEW" # OLD or NEW

    if VERSION_TO_CHOOSE == "OLD":
        # case division based on what is at the op positions
        if prev_l[op1_x, op1_y] == 2:

            # to goal?
            # if new_l[op2_x, op2_y] != 3:
            #     return "push_no_goal", f"p{op2_y+1 + n*(op2_x)}:position"
            # else:
            #     return "push", f"p{op2_y+1 + n*(op2_x)}:position"
            return "push_box", [f"p{op2_y+1 + n*(op2_x)}:position"]
        else:
            return f"move", [f"p{op1_y+1 + n*(op1_x)}:position"]
    
    elif VERSION_TO_CHOOSE == "NEW": # TODO: adapt this one
        # case division based on what is at the op positions
        if prev_l[op1_x, op1_y] == 2:

            # to goal?
            # if new_l[op2_x, op2_y] != 3:
            #     return "push_no_goal", f"p{op2_y+1 + n*(op2_x)}:position"
            # else:
            #     return "push", f"p{op2_y+1 + n*(op2_x)}:position"
            return "push", [f"p{op1_y+1 + n*(op1_x)}:position", f"p{op2_y+1 + n*(op2_x)}:position"] # TODO: add second argument
        else:
            return f"move", [f"p{ny+1 + n*(nx)}:position"]


    # 1 = player, 2 = stone, 4 = goal, 3 = stone at goal, 0 = clear, 5 = wall