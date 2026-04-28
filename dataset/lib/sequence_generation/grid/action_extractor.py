
from lib.rendering.grid.layout_builder import build_layout
import numpy as np

def detect_action(prev_state, curr_state):

    # extract information
    prev_l = build_layout(prev_state.literals)
    new_l = build_layout(curr_state.literals) 


    # get x and y position of players
    px, py = np.argwhere(prev_l['player'])[0].tolist()
    nx, ny = np.argwhere(new_l['player'])[0].tolist()



    # get dimension
    n = prev_l['player'].shape[0]


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
        direction = "none"


    # convert states to lists
    p_s_lst = list(prev_state.literals)
    c_s_lst = list(curr_state.literals)



    # previous version
    # # move action
    # if direction != "none":
    #     return f"move_{direction}", f"p{op1_y+1 + n*(op1_x)}:position"

    if direction != "none":
        # check if arm empty or holding a key
        pred_names = sorted([i.predicate.name for i in p_s_lst])
        arm_empty = "arm-empty" in pred_names
        if arm_empty:
            return f"move_{direction}_hand_empty", f"p{op1_y+1 + n*(op1_x)}:position"


        # get the key shape if the player is holding a key
        if not arm_empty:
            # find the shape of the key that the player is holding
            holding_pred = [i for i in p_s_lst if i.predicate.name == "holding"][0]
            variables = holding_pred.variables
            key_name = variables[0]
            key_shapes = [i for i in p_s_lst if i.predicate.name == "key-shape"]
            key_shape_id = None
            for ks in key_shapes:
                variables = ks.variables
                k_name = variables[0]
                shape = variables[1]
                if k_name == key_name:
                    key_shape_id = shape
                    break
            shape_id = int(key_shape_id[5:])+1
            return f"move_{direction}_with_key_type_{shape_id}", f"p{op1_y+1 + n*(op1_x)}:position"




    
    # # unlock action
    # if not (prev_l['door_open'] == new_l['door_open']).all():
    #     return "unlock_door", f"p{op1_y+1 + n*(op1_x)}:position"    


    # unlock action
    diff = np.argwhere(prev_l['door_open'] != new_l['door_open'])
    if diff.size > 0:
        dx, dy = diff[0].tolist()


        # find the shape of the door that was unlocked
        locked_doors = [i for i in p_s_lst if i.predicate.name == "lock-shape"]

        this_doors_shape = None
        for ld in locked_doors:
            variables = ld.variables
            door = variables[0]
            shape = variables[1]
            if door.startswith(f"p-{dy}-{dx}"):
                this_doors_shape = shape
                break

        if this_doors_shape is None:
            raise ValueError("Couldn't find the shape of the door that was unlocked")

        shape_id = int(this_doors_shape[5:])+1
        return f"unlock_door_with_key_type_{shape_id}", f"p{dy+1 + n*(dx)}:position"






    # pickup action
    if prev_l['keys'][px, py] > 0 and new_l['keys'][nx, ny] == 0:


        # 1. find the original pddl identifier for the key at that position ("key2" for example)
        keys_at_positions = [i for i in p_s_lst if i.predicate.name == "at"]
        key_name_pddl = None
        for key in keys_at_positions:
            variables = key.variables
            key_name = variables[0]
            position = variables[1]
            if position.startswith(f"p-{py}-{px}"):
                key_name_pddl = key_name
                break


        # 2. find key shape
        key_shapes = [i for i in p_s_lst if i.predicate.name == "key-shape"]
        key_shape_id = None
        for ks in key_shapes:
            variables = ks.variables
            k_name = variables[0]
            shape = variables[1]
            if k_name == key_name_pddl:
                key_shape_id = shape
                break

        shape_id = int(key_shape_id[5:])+1
        return f"pick_up_key_type_{shape_id}", f"p{op1_y+1 + n*(op1_x)}:position"
    



    
    # pickup_and_loose_action
    if prev_l['keys'][px, py] > 0 and new_l['keys'][nx, ny] > 0:


        # A: FIND THE NEW KEY SHAPE FIRST
        # 1. find the original pddl identifier for the key at that position ("key2" for example)
        keys_at_positions = [i for i in p_s_lst if i.predicate.name == "at"]
        key_name_pddl = None
        for key in keys_at_positions:
            variables = key.variables
            key_name = variables[0]
            position = variables[1]
            if position.startswith(f"p-{py}-{px}"):
                key_name_pddl = key_name
                break


        # 2. find key shape
        key_shapes = [i for i in p_s_lst if i.predicate.name == "key-shape"]
        key_shape_id = None
        for ks in key_shapes:
            variables = ks.variables
            k_name = variables[0]
            shape = variables[1]
            if k_name == key_name_pddl:
                key_shape_id = shape
                break

        new_key_shape_id = int(key_shape_id[5:])+1

        # B: FIND THE OLD KEY SHAPE FIRST
        # 1. find the original pddl identifier for the key at that position ("key2" for example)
        keys_at_positions = [i for i in c_s_lst if i.predicate.name == "at"]
        key_name_pddl = None
        for key in keys_at_positions:
            variables = key.variables
            key_name = variables[0]
            position = variables[1]
            if position.startswith(f"p-{py}-{px}"):
                key_name_pddl = key_name
                break


        # 2. find key shape
        key_shapes = [i for i in c_s_lst if i.predicate.name == "key-shape"]
        key_shape_id = None
        for ks in key_shapes:
            variables = ks.variables
            k_name = variables[0]
            shape = variables[1]
            if k_name == key_name_pddl:
                key_shape_id = shape
                break

        old_key_shape_id = int(key_shape_id[5:])+1




    return f"pick_up_key_type_{new_key_shape_id}_and_loose_key_type_{old_key_shape_id}", f"p{op1_y+1 + n*(op1_x)}:position"



    # putdown action
    if prev_l['keys'][px, py] == 0 and new_l['keys'][nx, ny] > 0:
        return ValueError("Planner tried to schedule the 'putdown' action.This should not happen, player should not put down a key without picking up a key. We basically removed this action.")
        return "put_down_key", f"p{op1_y+1 + n*(op1_x)}:position"


    # TODO: detect other actions as well
    import pdb;pdb.set_trace() # any not detected actions????