import random
import os








def infer_left_predicates(horizontally_dominoes, on_relations):
    left_predicates = set()

    for i in range(len(on_relations)):
        for j in range(i + 1, len(on_relations)):
            a1, a2 = on_relations[i]
            b1, b2 = on_relations[j]

            # Rule 1: same support
            if a2 == b2:
                left_predicates.add((a1, b1))

            # Rule 2: one block stacked on another
            if a1 == b1:
                left_predicates.add((a2, b2))
    new_lefts = set(left_predicates)  # Start with the existing ones

    # Build a reverse index: maps domino to list of dominoes on top of it
    on_top_of = {}
    for top, bottom in on_relations:
        if bottom not in on_top_of:
            on_top_of[bottom] = []
        on_top_of[bottom].append(top)

    for a, b in left_predicates:
        if a in horizontally_dominoes and b in horizontally_dominoes:
            tops_a = on_top_of.get(a, [])
            tops_b = on_top_of.get(b, [])
            for c in tops_a:
                for d in tops_b:
                    new_lefts.add((c, d))

    return list(new_lefts)

def get_left_clear(left_predicates, all_dominoes):
    # Get all dominoes that appear as 'a' (left of someone)
    left_doms = {a for a, _ in left_predicates}
    
    # Get all dominoes that appear as 'b' (someone is left of them)
    right_doms = {b for _, b in left_predicates}
    
    # Dominoes that are not on the right of any other — nothing is to their left
    left_clear = list(left_doms - right_doms)

    # Add all dominoes that are not in the left_predicates
    left_clear += [dom for dom in all_dominoes if dom not in left_doms and dom not in right_doms]
    
    return left_clear
def get_right_clear(right_predicates, all_dominoes):
    # Get all dominoes that appear as 'a' (right of someone)
    right_doms = {a for a, _ in right_predicates}
    
    # Get all dominoes that appear as 'b' (someone is right of them)
    left_doms = {b for _, b in right_predicates}
    
    # Dominoes that are not on the left of any other — nothing is to their right
    right_clear = list(right_doms - left_doms)

    # Add all dominoes that are not in the left_predicates
    right_clear += [dom for dom in all_dominoes if dom not in left_doms and dom not in right_doms]
    
    return right_clear



def infer_right_predicates(horizontally_dominoes, on_relations):
    return [(b,a) for a, b in infer_left_predicates(horizontally_dominoes, on_relations)]

def infer_clear_top_horizontally_predicates(clear_dominoes, horizontally_dominoes):
    return set(clear_dominoes).intersection(set(horizontally_dominoes))

def infer_clear_top_vertically_predicates(all_dominoes, on_relations, vertically_dominoes, left_predicates, right_predicates):
    clear_top_left_predicates = set()
    clear_top_right_predicates = set()

    for domino in all_dominoes:
        if domino not in vertically_dominoes:
            continue

        # Check if the domino is clear to the top left
        left_neighbor = next((a for a, b in left_predicates if b == domino), None)
        if left_neighbor is None:
            clear_top_left_predicates.add(domino)
        else:
            # Seach for in common horizontal block on the left side
            shared_child = next(
                (d for d, x in on_relations if x == domino and any(d == y for y, z in on_relations if z == left_neighbor)),
                None
            )
            if shared_child is None:
                clear_top_left_predicates.add(domino)
            


        # Check if the domino is clear to the top right
        right_neighbor = next((a for a, b in right_predicates if b == domino), None)
        if right_neighbor is None:
            clear_top_right_predicates.add(domino)
        else:
            # Seach for in common horizontal block on the right side
            shared_child = next(
                (d for d, x in on_relations if x == domino and any(d == y for y, z in on_relations if z == right_neighbor)),
                None
            )
            if shared_child is None:
                clear_top_right_predicates.add(domino)



    return list(clear_top_left_predicates), list(clear_top_right_predicates)







































def generate_random_pddl_problem(pddl_dir, problem_name="random_problem.pddl", num_dominoes=7, num_piles=7):
    """
    Generates a random PDDL problem file for the Blocks domain.

    Args:
        pddl_dir (str): Path to the directory containing the PDDL domain file.
        problem_name (str): Name of the problem file to be created.
        num_dominoes (int): Number of dominoes to use in the problem.

    Returns:
        str: Path to the generated problem file.
    """

    def calc_num_ground_row(num_total):
        summ = 1
        num_row = 1
        while summ < num_total:
            num_row += 2
            summ += num_row
        import math
        return math.ceil(num_row/2)



    unused_dominoes = [chr(97 + i) for i in range(num_dominoes)]  # Generate domino names a, b, c, ...
    random.shuffle(unused_dominoes)

    # List of elements
    robot = "robot"
    clear_dominoes = []
    ontable_dominoes = []
    horizontally_dominoes = []
    vertically_dominoes = []
    on_relations = []

    # Create dominoe towers
    import copy
    unused_dominoes_copy = copy.deepcopy(unused_dominoes)
    num_unused_dominoes = num_dominoes
    num_used_dominoes = 0
    while num_unused_dominoes > 0:
        num_dominoes_tower = random.randint(1, num_unused_dominoes)
        num_dominoes_tower = num_unused_dominoes # TODO: remove this later on but for now this is necessary to create correct random states
        num_used_dominoes += num_dominoes_tower
        num_unused_dominoes -= num_dominoes_tower

        # Create tower:
        vertically = True
        last_layer = []
        curr_layer = []

        ground = calc_num_ground_row(num_dominoes_tower) # start new ground layer
        for i in range(ground):
            ontable_dominoes.append(unused_dominoes_copy[i])
            last_layer.append(unused_dominoes_copy[i])
            vertically_dominoes.append(unused_dominoes_copy[i])
        unused_dominoes_copy = unused_dominoes_copy[ground:]
        num_dominoes_tower -= ground

        while num_dominoes_tower > 0:
            if vertically:
                ground -= 1
            vertically = not vertically

            for i in range(ground):
                if num_dominoes_tower > 0:
                    on_relations.append((unused_dominoes_copy[0], last_layer[i]))
                    if vertically: 
                        vertically_dominoes.append(unused_dominoes_copy[0])
                    else:
                        horizontally_dominoes.append(unused_dominoes_copy[0])
                    if not vertically:
                        on_relations.append((unused_dominoes_copy[0], last_layer[i+1]))
                    curr_layer.append(unused_dominoes_copy[0])
                    unused_dominoes_copy = unused_dominoes_copy[1:]
                    num_dominoes_tower -= 1
                else:
                    for j in range(i, ground):
                        clear_dominoes.append(last_layer[j])
                    break
            last_layer = curr_layer
            curr_layer = []

        for ele in last_layer:
            clear_dominoes.append(ele)


    # Compute things for new domain version
    clear_top_predicates = infer_clear_top_horizontally_predicates(clear_dominoes, horizontally_dominoes)
    left_predicates = infer_left_predicates(horizontally_dominoes, on_relations)
    right_predicates = infer_right_predicates(horizontally_dominoes, on_relations)
    left_clear = get_left_clear(left_predicates, unused_dominoes)
    right_clear = get_right_clear(right_predicates, unused_dominoes)
    clear_top_left_predicates, clear_top_right_predicates = infer_clear_top_vertically_predicates(unused_dominoes, on_relations, vertically_dominoes, left_predicates, right_predicates)

    # Write the problem file
    problem_file_path = os.path.join(pddl_dir, problem_name)
    with open(problem_file_path, "w") as f:
        f.write(f"(define (problem {problem_name[:-5]})\n")
        f.write("    (:domain dominoes)\n")
        f.write("    (:objects \n")
        for b in unused_dominoes:
            f.write("        " + b + " - domino \n")
        f.write("        " + robot + " - robot)\n")
        f.write("    (:init\n")

        # Write (ontable) facts
        for domino in ontable_dominoes:
            f.write(f"        (ontable {domino})\n")

        # Write (on) relationships
        for domino, below_block in on_relations:
            f.write(f"        (on {domino} {below_block})\n")

        # Write (clear_top) facts
        for domino in clear_top_predicates:
            f.write(f"        (clear_top {domino})\n")

        # Write (clear_top_left) facts
        for domino in clear_top_left_predicates:
            f.write(f"        (clear_top_left {domino})\n")
        # Write (clear_top_right) facts
        for domino in clear_top_right_predicates:
            f.write(f"        (clear_top_right {domino})\n")

        # Write (horizontally) facts
        for domino in horizontally_dominoes:
            f.write(f"        (horizontally {domino})\n")
        # Write (vertically) facts
        for domino in vertically_dominoes:
            f.write(f"        (vertically {domino})\n")

        # Write (left) and (right) predicates
        for a, b in left_predicates:
            f.write(f"        (left {a} {b})\n")
        for a, b in right_predicates:
            f.write(f"        (right {a} {b})\n")

        # Write (clear_left) and (clear_right) facts
        for domino in left_clear:
            f.write(f"        (clear_left {domino})\n")
        for domino in right_clear:
            f.write(f"        (clear_right {domino})\n")

        # Write handempty fact
        f.write(f"        (handempty robot)\n")

        # Write all available actions
        f.write("\n\n        ; action literals\n")
        for domino in unused_dominoes:
            f.write(f"        (pickup_lc {domino})\n")
            f.write(f"        (pickup_rc {domino})\n")

            f.write(f"        (putdown_lc {domino})\n")
            f.write(f"        (putdown_rc {domino})\n")
            
            f.write(f"        (rotate_vertically {domino})\n")
            f.write(f"        (rotate_horizontally {domino})\n")
            
            f.write(f"        (unstack_vertically_lv_rv {domino})\n")
            f.write(f"        (unstack_vertically_lv_r {domino})\n")
            f.write(f"        (unstack_vertically_l_rv {domino})\n")
            f.write(f"        (unstack_vertically_l_r {domino})\n")
            
            f.write(f"        (unstack_horizontally_lh_rh {domino})\n")
            f.write(f"        (unstack_horizontally_l_rh {domino})\n")
            f.write(f"        (unstack_horizontally_lh_r {domino})\n")
            f.write(f"        (unstack_horizontally_l_r {domino})\n")
            for other in unused_dominoes:
                if domino != other:
                    f.write(f"        (stack_vertically_lhv_rhv {domino} {other})\n")
                    f.write(f"        (stack_vertically_lh_rhv {domino} {other})\n")
                    f.write(f"        (stack_vertically_l_rhv {domino} {other})\n")
                    f.write(f"        (stack_vertically_lhv_rh {domino} {other})\n")
                    f.write(f"        (stack_vertically_lh_rh {domino} {other})\n")
                    f.write(f"        (stack_vertically_l_rh {domino} {other})\n")
                    f.write(f"        (stack_vertically_lhv_r {domino} {other})\n")
                    f.write(f"        (stack_vertically_lh_r {domino} {other})\n")
                    f.write(f"        (stack_vertically_l_r {domino} {other})\n")
                    for other2 in unused_dominoes:
                        if domino != other and domino != other2 and other != other2:
                            f.write(f"        (stack_horizontally_lh_rh {domino} {other} {other2})\n")
                            f.write(f"        (stack_horizontally_l_rh {domino} {other} {other2})\n")
                            f.write(f"        (stack_horizontally_lh_r {domino} {other} {other2})\n")
                            f.write(f"        (stack_horizontally_l_r {domino} {other} {other2})\n")

        f.write("    )\n")


        
        # Set a random goal
        goal_blocks = random.sample(unused_dominoes, min(len(unused_dominoes) - 1, 3))
        f.write("    (:goal (and\n")
        for i in range(len(goal_blocks) - 1):
            f.write(f"        (on {goal_blocks[i]} {goal_blocks[i+1]})\n")
        f.write("    ))\n")
        f.write(")\n")

    return problem_file_path
