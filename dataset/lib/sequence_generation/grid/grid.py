import pddlgym
import numpy as np
from gym.envs.registration import register
from lib.rendering.grid import grid_render

def register_grid_env(initialize_randomly=False, size=3, num_stones=2):
    """
    Registers a custom PDDL environment.
    This function defines the environment name and PDDL directory, registers the environment with the specified 
    domain and problem files, and then tests the environment by resetting it and returning the initial observation.
    Returns:
        env (pddlgym.PDDLEnv): The registered PDDL environment.
        obs (object): The initial observation after resetting the environment.
    """
    # Define environment name and PDDL directory
    ENV_NAME = "MyCustomGridEnv-v0"
    PDDL_DOMAIN_DIR = "pddl_files/grid"
    PDDL_PROBLEMS_DIR = "pddl_files/grid/random_problems"

    done = False
    while not done:
        try:
            #initialize_randomly = False
            if initialize_randomly:
                # Generate a random PDDL problem
                PDDL_PROBLEMS_DIR = "pddl_files/grid/random_problems"
                import lib.sequence_generation.grid.random_state as random_state
                import os
                random_state.generate_random_pddl_problem(PDDL_PROBLEMS_DIR, size=size, grid_size=size, num_stones = num_stones)
            done = True
        except Exception as e:
            print(f"Error generating random PDDL problem: {e}")

    # Register the environment
    register(
        id=ENV_NAME,
        entry_point="pddlgym.core:PDDLEnv",
        kwargs={
            "domain_file": f"{PDDL_DOMAIN_DIR}/domain.pddl",
            "problem_dir": f"{PDDL_PROBLEMS_DIR}/",
            "render": grid_render.render,
            "operators_as_actions": True,      # <-- key fix
            # optional, but usually nice for planning-style envs:
            # "dynamic_action_space": True,
        },
    )


    # Test the environment
    env = pddlgym.make(ENV_NAME)
    return env




def create_fresh_env(config, test_set=False):
    """
    Creates a fresh environment with the specified configuration.
    This function registers the environment with the specified configuration, resets the environment, and returns
    the environment and the initial observation.
    Args:
        config (dict): A dictionary containing the configuration parameters.
    Returns:
        env (pddlgym.PDDLEnv): The registered PDDL environment.
        obs (object): The initial observation after resetting the environment.
    """

    # randomize number of blocks
    JITTER_POS_XY = config['test_jitter_pos_xy'] if test_set else config['train_jitter_pos_xy']
    JITTER_SIZE_XY = config['test_jitter_size_xy'] if test_set else config['train_jitter_size_xy']

    # Randomize number of blocks
    NUM_BLOCKS = np.random.randint(config['train_num_blocks_range'][0], config['train_num_blocks_range'][1]+1) \
                 if not test_set else \
                 np.random.randint(config['test_num_blocks_range'][0], config['test_num_blocks_range'][1]+1)
    # Randomize number of piles
    NUM_STONES = np.random.randint(config['train_num_stones_range'][0], config['train_num_stones_range'][1]+1) \
                 if not test_set else \
                 np.random.randint(config['test_num_stones_range'][0], config['test_num_stones_range'][1]+1)
    config['max_number_piles'] = NUM_STONES
    RANDOMIZE_X_PILE_POSITIONS = config['randomize_x_position_of_piles']
    RENDER_REALISTICALLY = config['render_realistically']
    size = NUM_BLOCKS


    # Reset the environment with a random initial state
    env = register_grid_env(initialize_randomly=True, size=size, num_stones=NUM_STONES)
    obs, _ = env.reset()


    # Jitter the positions of the blocks
    env.jitter_pos_xy_vec = JITTER_POS_XY
    env.jitter_size_xy_vec = JITTER_SIZE_XY
    env.train_set = not test_set
    env.num_blocks = NUM_BLOCKS
    env.render_realistically = RENDER_REALISTICALLY

    return env, obs