import random
import os

def _inversion_parity(tiles):
    seq = [t for t in tiles if t != 't0']
    inv = 0
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if int(seq[i][1:]) > int(seq[j][1:]):
                inv += 1
    return inv % 2

def _neighbors(blank_pos, size):
    """Return legal neighbor blank positions for given (r,c)."""
    r, c = blank_pos
    moves = []
    if r > 0: moves.append((r-1, c))
    if r < size-1: moves.append((r+1, c))
    if c > 0: moves.append((r, c-1))
    if c < size-1: moves.append((r, c+1))
    return moves

def _apply_move(tiles, blank_pos, new_blank_pos, size):
    """Swap blank with tile at new_blank_pos. Returns new state + new blank_pos."""
    r1, c1 = blank_pos
    r2, c2 = new_blank_pos
    i1 = r1 * size + c1
    i2 = r2 * size + c2
    tiles = tiles[:]  # copy
    tiles[i1], tiles[i2] = tiles[i2], tiles[i1]
    return tiles, (r2, c2)

def generate_random_pddl_problem(pddl_dir, problem_name="random_problem.pddl", size=3, ensure_goal_diff=True):
    """
    Generates a random, solvable slide-tile PDDL problem.
    - For 3x3: parity-checked random init/goal
    - For 4x4: goal via ≤12 random moves from init (to avoid hard planning)
    """
    tiles = [f"t{i}" for i in range(0, size*size)]  # t0 is blank
    positions = [f"x{c+1}" for c in range(size)] + [f"y{r+1}" for r in range(size)]

    # --- Initial permutation ---
    random.shuffle(tiles)
    init_tiles = tiles[:]

    if size == 3:
        init_parity = _inversion_parity(init_tiles)
        # goal with same parity
        goal_tiles = init_tiles[:]
        while True:
            random.shuffle(goal_tiles)
            if ensure_goal_diff and goal_tiles == init_tiles:
                continue
            if _inversion_parity(goal_tiles) == init_parity:
                break
    else:
        # === 4x4 special case: derive goal by random moves (<=12) ===
        goal_tiles = init_tiles[:]
        blank_idx = goal_tiles.index("t0")
        blank_pos = (blank_idx // size, blank_idx % size)
        steps = random.randint(1, 50)
        for _ in range(steps):
            nbrs = _neighbors(blank_pos, size)
            new_blank_pos = random.choice(nbrs)
            goal_tiles, blank_pos = _apply_move(goal_tiles, blank_pos, new_blank_pos, size)

    # Write the problem file
    problem_file_path = os.path.join(pddl_dir, problem_name)
    with open(problem_file_path, "w") as f:
        f.write(f"(define (problem {problem_name[:-5]})\n")
        f.write("    (:domain slidetile)\n")

        f.write("    (:objects \n")
        for t in init_tiles:
            if t == "t0":
                continue
            f.write(f"        {t}\n")
        for i in range(1, size+1):
            f.write(f"        x{i} y{i}\n")
        f.write("    )\n")

        # Init state
        f.write("    (:init\n")
        for t in init_tiles:
            if t != "t0":
                f.write(f"        (tile {t})\n")
        for i in range(1, size+1):
            f.write(f"        (position x{i})\n")
            f.write(f"        (position y{i})\n")

        for idx, t in enumerate(init_tiles):
            x_pos = (idx % size) + 1
            y_pos = (idx // size) + 1
            if t == "t0":
                f.write(f"        (blank x{x_pos} y{y_pos})\n")
            else:
                f.write(f"        (at {t} x{x_pos} y{y_pos})\n")

        # inc/dec relations
        for i in range(1, size):
            f.write(f"        (inc x{i} x{i+1})\n")
            f.write(f"        (dec x{i+1} x{i})\n")
            f.write(f"        (inc y{i} y{i+1})\n")
            f.write(f"        (dec y{i+1} y{i})\n")

        f.write("    )\n")

        # Goal
        f.write("    (:goal (and\n")
        for idx, t in enumerate(goal_tiles):
            if t == "t0":
                continue
            x_pos = (idx % size) + 1
            y_pos = (idx // size) + 1
            f.write(f"        (at {t} x{x_pos} y{y_pos})\n")
        f.write("    ))\n")
        f.write(")\n")

    return problem_file_path
