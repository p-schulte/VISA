import random
import os

def generate_random_pddl_problem(pddl_dir, problem_name="random_problem.pddl", num_discs=5, num_pegs=3):
    """
    Generates a random PDDL problem file for the Towers of Hanoi domain.

    Args:
        pddl_dir (str): Path to the directory containing the PDDL domain file.
        problem_name (str): Name of the problem file to be created.
        num_discs (int): Number of disks to use in the problem.
        num_pegs (int): Number of pegs to use in the problem.

    Returns:
        str: Path to the generated problem file.
    """

    disks = [f"d{i}" for i in range(1, num_discs + 1)]  # Disk names: d1, d2, d3, ...
    pegs = [f"peg{i}" for i in range(1, num_pegs + 1)]  # Peg names: peg1, peg2, peg3, ...
    piles = [[peg] for peg in pegs]


    # Randomly generate piles of disks
    for disk in disks[::-1]:
        rnd_peg = random.choice(pegs)
        piles[pegs.index(rnd_peg)].append(disk)
    
    # Construct the representation of the piles
    on_relations = []
    smaller_relations = []
    clear_objects = set()
    num_piles = len(piles)
    for i, pile in enumerate(piles):
        # Assign all disks to the starting peg in correct order (largest at bottom)
        for j in range(len(pile) - 1):
            on_relations.append((pile[j+1], pile[j]))  # Bottom disk on the peg

        # The topmost disk is clear
        clear_objects.add(pile[-1])

    # Define smaller relationships (smaller disk can be on larger disk)
    smaller_relations = [(disks[i], disks[j]) for i in range(len(disks)) for j in range(i + 1, len(disks))]


    # Randomly choose a goal peg
    goal_peg = random.choice(pegs)

    # Define goal state: move the tower to goal peg
    goal_relations = [(disks[i], disks[i + 1]) for i in range(len(disks) - 1)]
    goal_relations.append((disks[-1], goal_peg))

    # Write the problem file
    problem_file_path = os.path.join(pddl_dir, problem_name)
    with open(problem_file_path, "w") as f:
        f.write(f"(define (problem {problem_name[:-5]})\n")
        f.write("    (:domain hanoi)\n")
        f.write("    (:objects ")
        
        # Define disks and pegs
        for d in disks:
            f.write(f"{d} ")
        for p in pegs:
            f.write(f"{p} ")

        f.write(")\n")
        f.write("    (:init\n")

        # Write (smaller) facts
        for d1, d2 in smaller_relations:
            f.write(f"        (smaller {d1} {d2})\n")

        # All disks are smaller than all pegs
        for d in disks:
            for p in pegs:
                f.write(f"        (smaller {d} {p})\n")

        # Write (on) facts for initial state
        for d1, d2 in on_relations:
            f.write(f"        (on {d1} {d2})\n")

        # Write (clear) facts
        for d in clear_objects:
            f.write(f"        (clear {d})\n")

        # Write all possible move actions
        f.write("\n\n        ; action literals\n")
        for d in disks:
            for p in pegs:
                f.write(f"        (move {d} {p})\n")
            for other_d in disks:
                if d != other_d:
                    f.write(f"        (move {d} {other_d})\n")

        f.write("    )\n")
        
        # Set the goal state
        f.write("    (:goal (and\n")
        for d1, d2 in goal_relations:
            f.write(f"        (on {d1} {d2})\n")
        f.write("    ))\n")
        f.write(")\n")


        

    return problem_file_path

# Example usage
if __name__ == "__main__":
    pddl_directory = "./pddl_problems"  # Change this to your actual directory
    os.makedirs(pddl_directory, exist_ok=True)  # Ensure the directory exists
    problem_path = generate_random_pddl_problem(pddl_directory, num_discs=5, num_pegs=3)
    print(f"PDDL problem generated: {problem_path}")
