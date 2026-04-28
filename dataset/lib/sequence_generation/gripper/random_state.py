import random
import os

def generate_random_pddl_problem(pddl_dir, problem_name="random_problem.pddl", num_blocks=7):
    """
    Generates a random PDDL problem file for the Blocks domain.

    Args:
        pddl_dir (str): Path to the directory containing the PDDL domain file.
        problem_name (str): Name of the problem file to be created.
        num_blocks (int): Number of blocks to use in the problem.

    Returns:
        str: Path to the generated problem file.
    """

    balls = [f"ball{i}" for i in range(1, num_blocks+1)]  # Generate block names a, b, c, ...
    random.shuffle(balls)

    rooma = [b for b in balls if random.randint(0,1) == 0]
    roomb = [b for b in balls if b not in rooma]

    # Write the problem file
    problem_file_path = os.path.join(pddl_dir, problem_name)
    with open(problem_file_path, "w") as f:
        f.write(f"(define (problem {problem_name[:-5]})\n")
        f.write("    (:domain gripper)\n")
        f.write("    (:objects \n")
        f.write("        " + "rooma roomb" + " - room \n")
        f.write("        " + " ".join(balls) + " - ball \n")
        f.write("        " + "left right" + " - gripper)\n")
        f.write("    (:init\n")


        if random.randint(0,1) == 0:
            f.write(f"        (at-robby rooma)\n")
        else:
            f.write(f"        (at-robby roomb)\n")
        f.write(f"        (isroom-a rooma)\n")
        f.write(f"        (isroom-b roomb)\n")
        f.write(f"        (above rooma roomb)\n")
        f.write(f"        (free left)\n")
        f.write(f"        (free right)\n")

        # Write (at) relationships
        for ball in rooma:
            f.write(f"        (at {ball} rooma)\n")
        for ball in roomb:
            f.write(f"        (at {ball} roomb)\n")


        # Write all available actions 
        f.write("\n\n        ; action literals\n")
        f.write(f"        (up rooma roomb)\n")
        f.write(f"        (down roomb rooma)\n")
        f.write(f"        (down rooma roomb)\n")
        f.write(f"        (up roomb rooma)\n")
        for ball in balls:
            for room in ["rooma", "roomb"]:
                for gripper in ["left", "right"]:
                    f.write(f"        (pick {ball} {room} {gripper})\n")
                    f.write(f"        (drop {ball} {room} {gripper})\n")

        f.write("    )\n")
        
        # Set a random goal
        random.shuffle(balls)

        rooma = [b for b in balls if random.randint(0,1) == 0]
        roomb = [b for b in balls if b not in rooma]        
        
        
        f.write("    (:goal (and\n")
        for ball in rooma:
            f.write(f"        (at {ball} rooma)\n")
        for ball in roomb:
            f.write(f"        (at {ball} roomb)\n")
        f.write("    ))\n")
        f.write(")\n")

    return problem_file_path
