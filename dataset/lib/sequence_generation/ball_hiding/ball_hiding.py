import pddlgym
import numpy as np
from gym.envs.registration import register
from lib.rendering.ball_hiding import ball_hiding_render
from lib.rendering.utils import PileTracker

def register_blocks_env(initialize_randomly=False, num_blocks=7, num_balls=1, num_piles=7):
    """
    Registers a custom PDDL environment.
    This function defines the environment name and PDDL directory, registers the environment with the specified 
    domain and problem files, and then tests the environment by resetting it and returning the initial observation.
    Returns:
        env (pddlgym.PDDLEnv): The registered PDDL environment.
        obs (object): The initial observation after resetting the environment.
    """
    # Define environment name and PDDL directory
    ENV_NAME = "MyCustomBallHidingEnv-v0"
    PDDL_DOMAIN_DIR = "pddl_files/ball_hiding"
    PDDL_PROBLEMS_DIR = "pddl_files/ball_hiding/problems"

    if initialize_randomly:
        # Generate a random PDDL problem
        PDDL_PROBLEMS_DIR = "pddl_files/ball_hiding/random_problems"
        import lib.sequence_generation.ball_hiding.random_state as random_state
        import os
        random_state.generate_random_pddl_problem(PDDL_PROBLEMS_DIR, num_blocks=num_blocks, num_balls=num_balls, num_piles=num_piles)

    # Register the environment
    register(
        id=ENV_NAME,  
        entry_point="pddlgym.core:PDDLEnv",
        kwargs={"domain_file": f"{PDDL_DOMAIN_DIR}/domain.pddl", "problem_dir": f"{PDDL_PROBLEMS_DIR}/", 'render' : ball_hiding_render.render},
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
    # Randomize number of balls
    NUM_BALLS = np.random.randint(config['train_num_balls_range'][0], config['train_num_balls_range'][1]+1) \
                 if not test_set else \
                 np.random.randint(config['test_num_balls_range'][0], config['test_num_balls_range'][1]+1)
    # Randomize number of piles
    MAX_PILES = np.random.randint(config['train_num_piles_range'][0], config['train_num_piles_range'][1]+1) \
                 if not test_set else \
                 np.random.randint(config['test_num_piles_range'][0], config['test_num_piles_range'][1]+1)
    config['max_number_piles'] = MAX_PILES
    RANDOMIZE_X_PILE_POSITIONS = config['randomize_x_position_of_piles']
    RENDER_REALISTICALLY = config['render_realistically']


    # Reset the environment with a random initial state
    env = register_blocks_env(initialize_randomly=True, num_blocks=NUM_BLOCKS, num_balls=NUM_BALLS, num_piles=MAX_PILES)
    obs, _ = env.reset()
    NUM_BLOCKS += NUM_BALLS # add balls to the number of blocks


    # Randomize colors
    from lib.rendering.ball_hiding import ball_hiding_render
    ball_hiding_render._block_name_to_color = {}


    
    # create pile tracker
    env.pile_tracker = PileTracker(max_piles=MAX_PILES, num_blocks=NUM_BLOCKS, rand_x_pos=RANDOMIZE_X_PILE_POSITIONS)

    
    # Jitter the positions of the blocks
    env.jitter_pos_xy_vec = JITTER_POS_XY
    env.jitter_size_xy_vec = JITTER_SIZE_XY
    env.train_set = not test_set
    env.num_blocks = NUM_BLOCKS
    env.render_realistically = RENDER_REALISTICALLY

    return env, obs