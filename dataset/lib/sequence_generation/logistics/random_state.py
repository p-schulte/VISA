import random
import os



def generate_random_pddl_problem(
    pddl_dir,
    problem_name="random_problem.pddl",
    num_packages=6,
    num_goals=3,
):
    """
    Generates a random PDDL problem file for the Logistics domain.

    Args:
        pddl_dir (str): Path to the directory containing the PDDL domain file.
        problem_name (str): Name of the problem file to be created.
        num_packages (int): Number of packages to include.
        num_goals (int): Number of goal conditions.

    Returns:
        str: Path to the generated problem file.
    """

    # --- Objects ---
    packages = [f"package{i}" for i in range(1, num_packages + 1)]
    # trucks = ["truckred", "trucklime"]
    # planes = ["planeblue", "planeyellow"]
    trucks = ["truckred"]
    planes = ["planeblue"]
    cities = ["city1", "city2"]
    locations = ["city1-1", "city2-1"]
    airports = ["city1-2", "city2-2"]

    all_locs = locations + airports
    all_transports = trucks + planes

    # --- File path ---
    problem_file_path = os.path.join(pddl_dir, problem_name)

    with open(problem_file_path, "w") as f:
        f.write(f"(define (problem {problem_name[:-5]})\n")
        f.write("   (:domain logistics)\n")

        # Objects
        f.write("   (:objects\n")
        f.write("      " + " ".join(packages) + " - obj\n")
        f.write("      " + " ".join(cities) + " - city\n")
        f.write("      " + " ".join(trucks) + " - truck\n")
        f.write("      " + " ".join(planes) + " - airplane\n")
        f.write("      " + " ".join(locations) + " - location\n")
        f.write("      " + " ".join(airports) + " - airport)\n")

        # Init
        f.write("   (:init\n")
        # in-city facts
        f.write("      (in-city city1-1 city1)\n")
        f.write("      (in-city city1-2 city1)\n")
        f.write("      (in-city city2-1 city2)\n")
        f.write("      (in-city city2-2 city2)\n")

        # different facts (no equality)
        for l1 in all_locs:
            for l2 in all_locs:
                if l1 != l2:
                    f.write(f"      (different {l1} {l2})\n")


        # random initial locations
        for truck in trucks:
            loc = random.choice(locations)  # trucks at ground locations
            f.write(f"      (at {truck} {loc})\n")
        for plane in planes:
            loc = random.choice(airports)  # planes at airports
            f.write(f"      (at {plane} {loc})\n")
        for package in packages:
            # either on ground, or inside a random transport
            if random.random() < 0.5:
                loc = random.choice(all_locs)
                f.write(f"      (at {package} {loc})\n")
            else:
                t = random.choice(all_transports)
                f.write(f"      (in {package} {t})\n")

        f.write("   )\n")

        # Goal
        goal_pkgs = random.sample(packages, min(num_goals, len(packages)))
        f.write("   (:goal (and\n")
        for pkg in goal_pkgs:
            loc = random.choice(all_locs)
            f.write(f"      (at {pkg} {loc})\n")
        f.write("   ))\n")
        f.write(")\n")

    return problem_file_path