import sys, argparse


import random


def generate_cross_with_doors(h=5, w=5, door_prob=0.3):
    """
    Generate a grid:
      - background: '.'
      - a vertical and horizontal line crossing (the 'cross')
      - on that cross: every cell is either 'w' (wall) or 'L' (door)

    Works for both odd and even h, w (>= 3).
    For even sizes, the cross is slightly shifted (still 4 rooms).
    """
    assert h >= 3 and w >= 3, "Need at least 3x3 to draw a cross"

    # start with all floor
    grid = [["." for _ in range(w)] for _ in range(h)]

    # "center" row/column (for even: one of the two middle ones)
    my = h // 2  # row index
    mx = w // 2  # column index

    # draw the cross as walls
    for x in range(w):
        grid[my][x] = "w"      # horizontal arm
    for y in range(h):
        grid[y][mx] = "w"      # vertical arm

    # now randomly turn some cross cells into doors
    for y in range(h):
        for x in range(w):
            if (y == my or x == mx):
                # keep the intersection always a wall
                if not (y == my and x == mx):
                    if random.random() < door_prob:
                        grid[y][x] = "L"

    # (my, mx) is always the intersection, keep it as 'w'
    # grid[my][mx] is already 'w' from drawing the cross

    return ["".join(row) for row in grid]


def colored(x, *args, **kwargs):
    return x

def cell_name(c: int, r: int) -> str:
    """Name for a grid cell at column c, row r."""
    return f"p-{c}-{r}"



from pathlib import Path
from typing import Set, List, Optional
import logging
logging.disable(logging.CRITICAL)   # suppress all messages (<= CRITICAL)
import random
from copy import deepcopy
from collections import deque

class Map:
    def __init__(self, nodes: int):
        self.nodes = nodes
        self.edges = [[] for n in range(self.nodes)]

    def add_edge(self, src: int, dst: int):
        if dst not in self.edges[src]:
            self.edges[src].append(dst)
            self.edges[dst].append(src)

    def remove_edges_at(self, node: int):
        #print(f'remove_edges_at: node={node}')
        assert 0 <= node and node <= self.nodes
        for i, edges in enumerate(self.edges):
            self.edges[i] = [dst for dst in edges if dst != node]
        self.edges[node] = []

    def clone(self):
        return deepcopy(self)

    # do BFS exploration
    def path(self, init: int, goal: int):
        assert 0 <= init and init < self.nodes and goal < self.nodes
        q = deque()
        q.append(init)
        explored = set([init])
        parent = [None for _ in range(self.nodes)]
        while len(q) > 0:
            src = q.popleft()
            assert src in explored
            if src == goal: break
            for dst in random.sample(self.edges[src], len(self.edges[src])):
                if dst not in explored:
                    explored.add(dst)
                    parent[dst] = src
                    q.append(dst)

        if goal < 0:
            return explored
        elif parent[goal] is None:
            return None
        else:
            src = goal
            rpath = []
            while src != init:
                rpath.append(src)
                src = parent[src]
                assert src is not None
            rpath.append(init)
            return list(reversed(rpath))

    def reachable_nodes(self, init: int):
        return self.path(init, -1)

class Floorplan:
    def __init__(self, fname: Path, logger):
        # read template file
        with fname.open('r') as fd:
            self.lines = [line.strip('\n') for line in fd.readlines()]

        # initialize data structures
        self.cells, self.cmap = [], dict()
        self.locks = set()
        self.walls = set()
        self.logger = logger
        self.robot = None
        self.goal = None

        # first pass, read cells
        for r, line in enumerate(self.lines):
            for c, char in enumerate(line):
                if char not in [' ']:
                    cell = (c, r)
                    if char not in ['w']:
                        self.cmap[cell] = len(self.cells)
                        self.cells.append(cell)
                    if char == 'L': self.locks.add(cell)
                    if char == 'w': self.walls.add(cell)
                    if char == 'R': self.robot = cell
                    if char == 'G': self.goal = cell

        # second pass, read connections
        self.map = Map(len(self.cells))
        for (c, r) in self.cells:
            cell = (c, r)
            src = self.cmap[cell]
            leftcell = (c-1, r)
            if leftcell in self.cmap:
                dst = self.cmap[leftcell]
                self.map.add_edge(src, dst)
            abovecell = (c, r-1)
            if abovecell in self.cmap:
                dst = self.cmap[abovecell]
                self.map.add_edge(src, dst)
        logger.info(f'Floorplan: ncells={len(self.cells)}, locks={self.locks}')

    def get_map(self, locks: Optional[Set] = None):
        #print(f'get_map: locks={self.locks}')
        map = self.map.clone()
        if locks != None:
            #self.logger.info(f'get_map: locks={locks}')
            for lock in locks:
                map.remove_edges_at(self.cmap[lock])
        return map


class Instance:
    def __init__(self, name: str, floorplan: Floorplan, nshapes: int, logger):
        self.name = name
        self.logger = logger
        self.floorplan = floorplan

        # assign shapes to locks, there would be one key per shape
        nlocks = len(floorplan.locks)
        if nshapes == 0:
            nshapes = random.randint(1, nlocks) if nlocks > 0 else 0
        self.lock_shapes = list(map(lambda x: x, random.choices(range(nshapes), k=nlocks)))
        self.key_shapes = list(range(0, nshapes))

        self.shape_map = dict()
        for i, cell in enumerate(floorplan.locks):
            self.shape_map[cell] = self.lock_shapes[i]
        logger.info(f'Locks: nlocks={nlocks}, nshapes={nshapes}, shape_map={self.shape_map}')

        # place robot and goal
        free_cells = list(set(floorplan.cells) - floorplan.locks)
        self.robot = random.sample(free_cells, k=1)[0] if floorplan.robot is None else floorplan.robot
        if floorplan.goal is None:
            reachable_cells = floorplan.map.reachable_nodes(floorplan.cmap[self.robot])
            self.goal = floorplan.cells[random.sample(list(reachable_cells), k=1)[0]]
        else:
            self.goal = floorplan.goal
        logger.debug(f'Init/goal: robot={self.robot}, goal={self.goal}')

        # construct high-level path from robot to goal
        robot_to_goal_path = floorplan.map.path(floorplan.cmap[self.robot], floorplan.cmap[self.goal])
        assert robot_to_goal_path is not None, f"With all locks open, cell {self.goal} isn't reachable from {self.robot}"
        lock_path = [floorplan.cells[cell] for cell in robot_to_goal_path if floorplan.cells[cell] in floorplan.locks]
        logger.info(f'Path: init={self.robot}, goal={self.goal}, path={list(map(lambda i: floorplan.cells[i], robot_to_goal_path))}, locks={lock_path}')

        # place keys so that resulting placement is solvable
        locks = set(floorplan.locks)
        self.key_locations = [None for _ in range(nshapes)] # one key per shape
        current_map = floorplan.get_map(locks)
        current_pos = floorplan.cmap[self.robot]
        previous_loc = set([current_pos])
        for i, lock in enumerate(lock_path):
            shape = self.shape_map[lock]
            logger.debug(f'i={i}, current_pos={floorplan.cells[current_pos]}, lock={lock}, shape={shape}')
            if self.key_locations[shape] == None:
                reachable_cells = current_map.reachable_nodes(current_pos) - previous_loc
                self.key_locations[shape] = random.sample(list(reachable_cells), k=1)[0]
                logger.debug(f'key_location: shape={shape}, loc={floorplan.cells[self.key_locations[shape]]}, reachable_cells={list(map(lambda i: floorplan.cells[i], reachable_cells))}')
                current_pos = self.key_locations[shape]
                previous_loc.add(current_pos)
                open_locks = set([lock for lock in floorplan.locks if self.shape_map[lock] == shape])
                logger.debug(f'open_locks={open_locks}')
                locks -= open_locks
                current_map = floorplan.get_map(locks)

        # keys that have not yet placed can be anywhere
        for i, loc in enumerate(self.key_locations):
            if loc is None:
                self.key_locations[i] = floorplan.cmap[random.sample(list(floorplan.cells), k=1)[0]]



        # check for any anomalies
        key_poss = set()
        for i, loc in enumerate(self.key_locations):
            pos = self.floorplan.cells[loc]

            # check if a key was placed at a lock cell
            if pos in self.floorplan.locks:
                # print(f'Error: key {i} placed at lock cell {pos}')
                raise ValueError(f'key {i} placed at lock cell {pos}')

            # check if keys double at a location
            if pos in key_poss:
                # print(f'Error: key {i} placed at already occupied cell {pos}')
                raise ValueError(f'key {i} placed at already occupied cell {pos}')
            key_poss.add(pos)



    def write(self, fname: Path):
        with fname.open('w') as fd:
            fd.write(f'(define (problem {self.name})\n')
            fd.write(f'  (:domain grid)\n')

            # ---------- OBJECTS ----------
            fd.write('  (:objects\n')
            fd.write('   ')
            # one location object per non-wall cell, named by its coordinates
            for (c, r) in self.floorplan.cells:
                fd.write(f' {cell_name(c, r)}')
            fd.write(f'\n   ')
            for i, shape in enumerate(self.key_shapes):
                fd.write(f' shape{shape}')
            fd.write(f'\n   ')
            for i, shape in enumerate(self.key_shapes):
                fd.write(f' key{i}')
            fd.write('\n  )\n')

            # ---------- INIT ----------
            fd.write('  (:init')
            fd.write('\n    ; Object types\n   ')
            # place(...)
            for (c, r) in self.floorplan.cells:
                fd.write(f' (place {cell_name(c, r)})')
            fd.write(f'\n   ')
            for i, shape in enumerate(self.key_shapes):
                fd.write(f' (shape shape{shape})')
            fd.write(f'\n   ')
            for i, shape in enumerate(self.key_shapes):
                fd.write(f' (key key{i})')

            # ---- open / locked ----
            fd.write('\n    ; Open/locked cells\n   ')
            for (c, r) in self.floorplan.cells:
                if (c, r) not in self.floorplan.locks:
                    fd.write(f' (open {cell_name(c, r)})')
            if len(self.floorplan.locks) > 0:
                fd.write('\n   ')
                for (c, r) in self.floorplan.cells:
                    if (c, r) in self.floorplan.locks:
                        fd.write(f' (locked {cell_name(c, r)})')
            fd.write('\n')

            # ---- connectivity graph ----
            fd.write('    ; Connected cells\n')
            for src in range(self.floorplan.map.nodes):
                (cs, rs) = self.floorplan.cells[src]
                src_name = cell_name(cs, rs)
                for dst in self.floorplan.map.edges[src]:
                    (cd, rd) = self.floorplan.cells[dst]
                    dst_name = cell_name(cd, rd)
                    fd.write(f'    (conn {src_name} {dst_name})\n')

            # ---- lock and key shapes ----
            fd.write('    ; Lock and key shapes\n')
            for cell in self.shape_map:
                c, r = cell
                fd.write(
                    f'    (lock-shape {cell_name(c, r)} '
                    f'shape{self.shape_map[cell]})\n'
                )
            for i, shape in enumerate(self.key_shapes):
                fd.write(f'    (key-shape key{i} shape{shape})\n')

            # ---- key placement ----
            fd.write('    ; Key placement\n')
            for i, loc in enumerate(self.key_locations):
                c, r = self.floorplan.cells[loc]   # loc is an index
                fd.write(f'    (at key{i} {cell_name(c, r)})\n')

            # ---- robot placement ----
            fd.write('    ; Robot placement\n')
            rc, rr = self.robot
            fd.write(f'    (at-robot {cell_name(rc, rr)})\n')
            fd.write('    (arm-empty)\n')
            fd.write('  )\n')

            # ---------- GOAL ----------
            gc, gr = self.goal
            fd.write(f'  (:goal (at-robot {cell_name(gc, gr)}))\n')
            fd.write(')\n')



def get_logger(name: str, log_file: Path, level = logging.INFO):
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level)

    # add stdout handler
    #formatter = logging.Formatter('[%(levelname)s] %(message)s')
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s')
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # add file handler
    if log_file != '':
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s')
        file_handler = logging.FileHandler(str(log_file), 'a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def close_logger(logger):
    handlers = logger.handlers
    for handler in handlers:
       logger.removeHandler(handler)
       handler.close()

def get_args():
    # argument parser
    default_seed = 0
    default_debug_level = 0
    default_num_instances = 1
    parser = argparse.ArgumentParser('mini_grid.py')
    parser.add_argument('--seed', type=int, default=default_seed, help=f'Seed for random generator (default={default_seed})')
    parser.add_argument('--debug_level', type=int, default=default_debug_level, help=f'Set debug level (default={default_debug_level})')
    parser.add_argument('--num_instances', type=int, default=default_num_instances, help=f'Number of instances to generate (default={default_num_instances})')
    parser.add_argument('floorplan', type=Path, help='Filename for floorplan')
    parser.add_argument('nshapes', type=int, help='Number of shapes for locks (0 means choose it randomly)')

    # paths
    default_results_path = ''
    default_floorplans_path = ''
    paths = parser.add_argument_group('paths')
    paths.add_argument('--results', type=Path, default=default_results_path, help=f"Path to results folders (default='{default_results_path}')")
    paths.add_argument('--floorplans_path', type=Path, default=default_floorplans_path, help=f"Path to floorplans (default='{default_floorplans_path}')")

    # parse arguments
    args = parser.parse_args()
    return args

def main(args: argparse.Namespace):
    # setup logger and identify call
    log_file = Path('./log.txt')
    log_level = logging.INFO if args.debug_level == 0 else logging.DEBUG
    logger = get_logger('mini_grid.py', log_file, log_level)
    logger.info(colored(f"Using log file '{log_file}'", 'green'))
    logger.info(f'call=|{" ".join(sys.argv)}|')

    # set random seed
    random.seed(args.seed)

    # read floorplan
    floorplan = Floorplan(args.floorplans_path / args.floorplan, logger)

    # generate random instances
    name = args.floorplan.name
    args.results.mkdir(parents=True, exist_ok=True)
    for i in range(args.num_instances):
        name = f'grid_{name}_s{args.nshapes}_seed{args.seed}_n{i}'.replace('.', '_')
        pddl_name = args.results / Path(name).with_suffix('.pddl')
        instance = Instance(name, floorplan, args.nshapes, logger)
        instance.write(pddl_name)
        logger.info(colored(f'{pddl_name} written!', 'blue'))




import os
from pathlib import Path

# You can adjust this to whatever floorplan you want to use by default
DEFAULT_FLOORPLAN_PATH = Path("pddl_files/grid/floorplans") / "floor.fpl"  # example name


def generate_random_pddl_problem(
    pddl_dir: str,
    problem_name: str = "random_problem.pddl",
    grid_size: int = 5,      # unused here (kept for API compatibility)
    num_stones: int = 2,     # unused
    walk_length: int = 20,   # unused
    min_pushes: int = 3,     # unused
    floor_ratio: float = 0.5,# unused
    size: int = 3,           # we reuse this as 'nshapes'
):
    """
    Grid / key-lock version.

    Uses the existing Floorplan + Instance classes to generate a random
    solvable instance for the 'grid' domain.

    Arguments:
        pddl_dir: directory where the problem file is written.
        problem_name: name of the PDDL problem file.
        size: number of shapes/keys (nshapes argument to Instance).
               If 0, the generator itself will randomize it.

    All other arguments are kept only for interface compatibility with the
    Sokoban version and are ignored here.
    """
    os.makedirs(pddl_dir, exist_ok=True)
    pddl_dir_path = Path(pddl_dir)
    problem_file_path = pddl_dir_path / problem_name

    # --- logger setup (no log file, only console) ---
    log_level = logging.INFO
    logger = get_logger(
        name=f"mini_grid.generate_random_pddl_problem",
        log_file="",   # empty => no file handler
        level=log_level,
    )
    # then call your function


    GENERATE_RANDOM_FLOORPLAN = True  # set to True to generate a random floorplan each time
    if GENERATE_RANDOM_FLOORPLAN:
        m = generate_cross_with_doors(grid_size, grid_size, door_prob=0.2)
        with open((DEFAULT_FLOORPLAN_PATH), 'w') as fd:
            for line in m:
                fd.write(f"{line}\n")



    try:
        # --- load floorplan template ---
        floorplan_path = DEFAULT_FLOORPLAN_PATH
        if not floorplan_path.exists():
            raise FileNotFoundError(
                f"Floorplan template not found at {floorplan_path}. "
                f"Adjust DEFAULT_FLOORPLAN_PATH to point to your .txt floorplan."
            )

        floorplan = Floorplan(floorplan_path, logger)

        # --- create random instance ---
        nshapes = max(0, size)  # 0 => let Instance choose randomly
        nshapes = 2  # FIX IT TO TWO SHAPES
        base_name = Path(problem_name).with_suffix("").name

        done = False
        import time;start = time.time()
        while not done:
            try:
                instance = Instance(base_name, floorplan, nshapes, logger)
                done = True
            except:
                if time.time() - start > 10.0:
                    raise TimeoutError("Timed out generating a valid instance.")
        # end = time.time()
        # print(f"Instance generation took {end - start:.2f} seconds.")


        # --- write PDDL file ---
        instance.write(problem_file_path)
        logger.info(colored(f"{problem_file_path} written!", "blue"))

    finally:
        # Clean up logger handlers so we don't duplicate handlers on reuse
        close_logger(logger)

    return str(problem_file_path)
