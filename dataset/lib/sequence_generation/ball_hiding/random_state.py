import random
import os

def generate_random_pddl_problem(pddl_dir, problem_name="random_problem.pddl", num_blocks=7, num_balls=1, num_piles=7):
    """
    Generates a random PDDL problem file for the Blocks domain.

    Args:
        pddl_dir (str): Path to the directory containing the PDDL domain file.
        problem_name (str): Name of the problem file to be created.
        num_blocks (int): Number of blocks to use in the problem.

    Returns:
        str: Path to the generated problem file.
    """

    blocks = [chr(97 + i) for i in range(num_blocks)]  # Generate block names a, b, c, ...
    random.shuffle(blocks)
    balls = [chr(97 + i) for i in range(num_blocks, num_blocks + num_balls)]  # Generate ball names a, b, c, ...
    random.shuffle(balls)

    robot = "robot"

    # Choose random blocks to be on the table
    num_ontable = random.randint(1, min(num_blocks, num_piles - len(balls)))  # Random number of blocks on table
    ontable_blocks = random.sample(blocks, num_ontable)

    # Define clear blocks (last block in each stack)
    clear_blocks = set(ontable_blocks)
    ontable_blocks.extend(balls)  # Add balls to the ontable blocks

    # Create stacking relationships
    on_relations = []
    for block in blocks:
        if block not in ontable_blocks:
            if len(clear_blocks) == 0:
                break
            below_block = random.choice(list(clear_blocks))  # Pick a block already placed
            on_relations.append((block, below_block))
            if below_block in clear_blocks:
                clear_blocks.remove(below_block)
            clear_blocks.add(block)

    # Decide if the robot is holding a block or empty
    holding_block = None
    if random.random() > 0.5 and len(clear_blocks) != 0:  # 50% chance the robot is holding a block
        holding_block = random.choice(list(clear_blocks))
        if holding_block in ontable_blocks:
            ontable_blocks.remove(holding_block)
            try:
                clear_blocks.remove(holding_block)
            except: 
                pass
        for block, below_block in on_relations:
            if block == holding_block:
                clear_blocks.remove(block)
                clear_blocks.add(below_block)
                on_relations.remove((block, below_block))
                break

    # Write the problem file
    problem_file_path = os.path.join(pddl_dir, problem_name)
    with open(problem_file_path, "w") as f:
        f.write(f"(define (problem {problem_name[:-5]})\n")
        f.write("    (:domain ball_hiding)\n")
        f.write("    (:objects \n")
        for b in blocks:
            f.write("        " + b + " - block \n")
        for b in balls:
            f.write("        " + b + " - ball \n")
        f.write("        " + robot + " - robot)\n")
        f.write("    (:init\n")

        # Write (ontable) facts
        for block in ontable_blocks:
            f.write(f"        (ontable {block})\n")

        # Write (on) relationships
        for block, below_block in on_relations:
            f.write(f"        (on {block} {below_block})\n")

        # Write (clear) facts
        for block in clear_blocks:
            f.write(f"        (clear {block})\n")

        # Write (ball) facts
        for block in blocks:
            f.write(f"        (block_empty {block})\n")

        # Write robot hand status
        if holding_block:
            f.write(f"        (holding {holding_block})\n")
            f.write(f"        (handfull {robot})\n")
        else:
            f.write(f"        (handempty {robot})\n")

        # Write all available actions
        f.write("\n\n        ; action literals\n")
        for block in blocks:
            f.write(f"        (pickup_block {block})\n")
            f.write(f"        (putdown_block {block})\n")
            f.write(f"        (unstack_block {block})\n")
            for other in blocks:
                if block != other:
                    f.write(f"        (stack_block {block} {other})\n")
        for ball in balls:
            f.write(f"        (pickup_ball {ball})\n")
            f.write(f"        (putdown_ball {ball})\n")
            f.write(f"        (unstack_ball {ball})\n")
            for block in blocks:
                f.write(f"        (stack_ball {ball} {block})\n")
                f.write(f"        (hide_ball {ball} {block})\n")
                f.write(f"        (unhide_ball {ball} {block})\n")

        f.write("    )\n")
        
        # Set a random goal
        goal_balls = random.sample(balls, min(len(balls), 2))
        goal_blocks = random.sample(blocks, min(len(blocks) - 1, 2))
        f.write("    (:goal (and\n")
        amnt = min(len(goal_balls), len(goal_blocks))
        for i in range(amnt):
            f.write(f"        (in {goal_balls[i]} {goal_blocks[i]})\n")
        f.write("    ))\n")
        f.write(")\n")

    return problem_file_path
