import os
import random


def pos_name(x, y):
    return f"pos-{x}-{y}"


def neighbors_4(x, y, grid_size):
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        nx, ny = x + dx, y + dy
        if 1 <= nx <= grid_size and 1 <= ny <= grid_size:
            yield nx, ny


def generate_floor_layout(grid_size, num_stones,
                          target_floor_ratio=0.4):
    """
    Generate a random connected set of floor cells (the rest are walls).

    We ensure:
      - at least num_stones + 2 floor cells (stones + player + a bit of slack)
      - the floor is one connected component (no isolated islands).
    """
    total_cells = grid_size * grid_size
    min_floor = num_stones + 2
    target_floor = max(min_floor, int(total_cells * target_floor_ratio))

    # Randomized DFS expansion from a random start cell
    start = (random.randint(1, grid_size), random.randint(1, grid_size))
    floor = {start}
    stack = [start]

    while len(floor) < target_floor and stack:
        cx, cy = stack.pop()
        neighs = list(neighbors_4(cx, cy, grid_size))
        random.shuffle(neighs)
        for nx, ny in neighs:
            if (nx, ny) not in floor:
                floor.add((nx, ny))
                stack.append((nx, ny))
                if len(floor) >= target_floor:
                    break

    # If for some reason we still have too few floor cells, just add random ones
    all_cells = [(x, y) for x in range(1, grid_size + 1)
                        for y in range(1, grid_size + 1)]
    remaining = [c for c in all_cells if c not in floor]
    random.shuffle(remaining)
    while len(floor) < min_floor and remaining:
        floor.add(remaining.pop())

    if len(floor) < min_floor:
        raise RuntimeError("Could not create enough floor cells.")

    return floor  # set of (x, y)


def generate_sokoban_state_on_floor(grid_size, num_stones, floor_cells,
                                    walk_length=50, min_pushes=2,
                                    max_tries=30):
    """
    NEW semantics: forward random walk from a random initial state on floor_cells.

    Returns:
        player_init_xy: (x, y)
        stones_init_dict: {idx: (x, y)}  -- initial stone positions
        goals_xy: list[(x, y)]           -- final stone positions (goal cells)

    Ensures:
        - at least min_pushes pushes were made
        - final set of stone positions != initial set
        - no initial stone position is a goal cell
    """
    floor_cells = set(floor_cells)
    floor_list = list(floor_cells)

    for _ in range(max_tries):
        # 1) random initial stone positions (distinct floor cells)
        stone_init_positions = random.sample(floor_list, num_stones)
        stones = {i: stone_init_positions[i] for i in range(num_stones)}

        # 2) player on a free floor cell
        free_cells = [c for c in floor_list if c not in stone_init_positions]
        if not free_cells:
            continue
        player_init = random.choice(free_cells)
        player = player_init

        pushes = 0

        # 3) forward random walk
        for _ in range(walk_length):
            legal_moves = []
            stone_positions = set(stones.values())

            for dx, dy, dir_name in [
                (0, -1, "dir-up"),
                (0,  1, "dir-down"),
                (-1, 0, "dir-left"),
                (1,  0, "dir-right"),
            ]:
                px, py = player
                nx, ny = px + dx, py + dy
                target = (nx, ny)

                # must be on floor
                if target not in floor_cells:
                    continue

                if target in stone_positions:
                    # attempt push
                    bx, by = nx + dx, ny + dy
                    beyond = (bx, by)
                    if beyond not in floor_cells:
                        continue
                    if beyond in stone_positions:
                        continue

                    # which stone?
                    stone_idx = None
                    for i, pos in stones.items():
                        if pos == target:
                            stone_idx = i
                            break
                    if stone_idx is None:
                        continue
                    legal_moves.append(("push", dir_name, target, beyond, stone_idx))
                else:
                    # normal move
                    legal_moves.append(("move", dir_name, target, None, None))

            if not legal_moves:
                break  # stuck; end this walk

            # Slight bias towards pushes so it's not just walking around
            pushes_available = [m for m in legal_moves if m[0] == "push"]
            if pushes < min_pushes and pushes_available:
                move = random.choice(pushes_available)
            else:
                move = random.choice(legal_moves)

            kind, dir_name, target, beyond, stone_idx = move
            if kind == "move":
                player = target
            else:  # push
                player = target
                stones[stone_idx] = beyond
                pushes += 1

        # 4) evaluate the resulting state
        stone_final_positions = [stones[i] for i in range(num_stones)]

        # enough pushes?
        if pushes < min_pushes:
            continue

        # non-trivial: final set != initial set
        if set(stone_final_positions) == set(stone_init_positions):
            continue

        # ensure NO initial stone is on a goal cell
        goal_set = set(stone_final_positions)
        if any(init in goal_set for init in stone_init_positions):
            continue

        stones_init_dict = {i: stone_init_positions[i] for i in range(num_stones)}
        return player_init, stones_init_dict, stone_final_positions

    raise RuntimeError("Failed to generate non-trivial solvable state on floor layout.")


def generate_random_pddl_problem(
    pddl_dir: str,
    problem_name: str = "random_problem.pddl",
    grid_size: int = 5,
    num_stones: int = 2,
    walk_length: int = 20,
    min_pushes: int = 3,
    floor_ratio: float = 0.5,
    size: int = 3,
):
    """
    Full pipeline:
      1) generate random connected floor layout (walls elsewhere),
      2) generate a forward random plan from a random initial state,
      3) use that initial state as :init and final stone positions as goals,
      4) output PDDL compatible with your sokoban domain.
    """
    os.makedirs(pddl_dir, exist_ok=True)

    # 1) floor layout with walls
    floor_cells = generate_floor_layout(
        grid_size, num_stones, target_floor_ratio=floor_ratio
    )

    # 2) solvable state on that layout (forward simulation)
    player_xy, stones_dict, goals_xy = generate_sokoban_state_on_floor(
        grid_size, num_stones, floor_cells,
        walk_length=walk_length, min_pushes=min_pushes
    )

    # Useful maps
    coord_to_name = {
        (x, y): pos_name(x, y)
        for x in range(1, grid_size + 1)
        for y in range(1, grid_size + 1)
    }

    player_pos_name = coord_to_name[player_xy]
    stones = [f"stone-{i+1:02d}" for i in range(num_stones)]
    stone_pos_names = {
        stones[i]: coord_to_name[stones_dict[i]] for i in range(num_stones)
    }
    goals = [coord_to_name[g] for g in goals_xy]

    # All positions (locations) in the grid
    positions = [coord_to_name[(x, y)]
                 for x in range(1, grid_size + 1)
                 for y in range(1, grid_size + 1)]

    # Directions & player name
    directions = ["dir-down", "dir-left", "dir-right", "dir-up"]
    player_name = "player-01"

    problem_file_path = os.path.join(pddl_dir, problem_name)

    with open(problem_file_path, "w") as f:
        f.write(f"(define (problem {problem_name[:-5]}) (:domain sokoban)\n")

        # Objects
        f.write("  (:objects\n")
        for d in directions:
            f.write(f"        {d} - direction\n")
        f.write(f"        {player_name} - thing\n")
        for pos in positions:
            f.write(f"        {pos} - location\n")
        for s in stones:
            f.write(f"        {s} - thing\n")
        f.write("  )\n")

        # Goals: at-goal stone-X (using is-goal locations)
        f.write("  (:goal (and\n")
        for s in stones:
            f.write(f"        (at-goal {s})\n")
        f.write("  ))\n")

        # Init
        f.write("  (:init \n")

        # Player
        f.write(f"        (at {player_name} {player_pos_name})\n")

        # Stones initial positions
        for s, pos in stone_pos_names.items():
            f.write(f"        (at {s} {pos})\n")

        # Clear: only floor cells that are not occupied
        occupied = {player_pos_name} | set(stone_pos_names.values())
        for (x, y) in floor_cells:
            pos = coord_to_name[(x, y)]
            if pos not in occupied:
                f.write(f"        (clear {pos})\n")

        # Goals (final stone positions)
        for pos in goals:
            f.write(f"        (is-goal {pos})\n")

        # Mark all positions as non-goal (matches your Microban file)
        for pos in positions:
            f.write(f"        (is-nongoal {pos})\n")

        # Type markers
        f.write(f"        (is-player {player_name})\n")
        for s in stones:
            f.write(f"        (is-stone {s})\n")

        # Allowed directions
        for d in directions:
            f.write(f"        (move {d})\n")

        # move-dir ONLY between neighboring floor cells
        floor_cells_set = set(floor_cells)
        for (x, y) in floor_cells_set:
            from_pos = coord_to_name[(x, y)]
            # right, left, down, up
            if (x + 1, y) in floor_cells_set:
                f.write(f"        (move-dir {from_pos} {coord_to_name[(x+1, y)]} dir-right)\n")
            if (x - 1, y) in floor_cells_set:
                f.write(f"        (move-dir {from_pos} {coord_to_name[(x-1, y)]} dir-left)\n")
            if (x, y + 1) in floor_cells_set:
                f.write(f"        (move-dir {from_pos} {coord_to_name[(x, y+1)]} dir-down)\n")
            if (x, y - 1) in floor_cells_set:
                f.write(f"        (move-dir {from_pos} {coord_to_name[(x, y-1)]} dir-up)\n")

        f.write("  )\n")
        f.write(")\n")

    return problem_file_path
