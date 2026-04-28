import pddlgym
import numpy as np
from gym.envs.registration import register
from lib.rendering.dominoes import dominoes_render
from lib.rendering.utils import PileTracker

def register_dominoes_env(initialize_randomly=False, num_dominoes=7, num_piles=7):
    """
    Registers a custom PDDL environment.
    This function defines the environment name and PDDL directory, registers the environment with the specified 
    domain and problem files, and then tests the environment by resetting it and returning the initial observation.
    Returns:
        env (pddlgym.PDDLEnv): The registered PDDL environment.
        obs (object): The initial observation after resetting the environment.
    """
    # Define environment name and PDDL directory
    ENV_NAME = "MyCustomDominoesEnv-v0"
    PDDL_DOMAIN_DIR = "pddl_files/dominoes"
    PDDL_PROBLEMS_DIR = "pddl_files/dominoes/problems"

    if initialize_randomly:
        # Generate a random PDDL problem
        PDDL_PROBLEMS_DIR = "pddl_files/dominoes/random_problems"
        import lib.sequence_generation.dominoes.random_state as random_state
        import os
        random_state.generate_random_pddl_problem(PDDL_PROBLEMS_DIR, num_dominoes=num_dominoes, num_piles=num_piles)


    # Register the environment
    register(
        id=ENV_NAME,  
        entry_point="pddlgym.core:PDDLEnv",
        kwargs={"domain_file": f"{PDDL_DOMAIN_DIR}/domain.pddl", "problem_dir": f"{PDDL_PROBLEMS_DIR}/", 'render' : dominoes_render.render},
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

    # randomize number of dominoes
    JITTER_POS_XY = config['test_jitter_pos_xy'] if test_set else config['train_jitter_pos_xy']
    JITTER_SIZE_XY = config['test_jitter_size_xy'] if test_set else config['train_jitter_size_xy']

    # Randomize number of dominoes
    NUM_BLOCKS = np.random.randint(config['train_num_dominoes_range'][0], config['train_num_dominoes_range'][1]+1) \
                 if not test_set else \
                 np.random.randint(config['test_num_dominoes_range'][0], config['test_num_dominoes_range'][1]+1)
    # Randomize number of piles
    MAX_PILES = np.random.randint(config['train_num_piles_range'][0], config['train_num_piles_range'][1]+1) \
                 if not test_set else \
                 np.random.randint(config['test_num_piles_range'][0], config['test_num_piles_range'][1]+1)
    config['max_number_piles'] = MAX_PILES
    RANDOMIZE_X_PILE_POSITIONS = config['randomize_x_position_of_piles']
    RENDER_REALISTICALLY = config['render_realistically']


    # Reset the environment with a random initial state
    env = register_dominoes_env(initialize_randomly=True, num_dominoes=NUM_BLOCKS, num_piles=MAX_PILES)
    obs, _ = env.reset()


    # Randomize colors
    from lib.rendering.dominoes import dominoes_render
    dominoes_render._block_name_to_color = {}
    dominoes_render._domino_name_to_x_pos = {}
    
    # Jitter the positions of the dominoes
    env.jitter_pos_xy_vec = JITTER_POS_XY
    env.jitter_size_xy_vec = JITTER_SIZE_XY
    env.train_set = not test_set
    env.num_dominoes = NUM_BLOCKS
    env.render_realistically = RENDER_REALISTICALLY

    return env, obs



def test_obs_legit(obs):
    lits = list(obs.literals)
    lits_str = [str(lit) for lit in lits]
    lit_map = parse_literal_map(lits_str)

    for dom in lit_map.keys():
        pred_list = [a[0] for a in lit_map[dom]]


        
        # Check exclusiveness of horizontally and vertically
        assert not ('horizontal' in pred_list and 'vertical' in pred_list), f"\n {dom} \n{lit_map[dom]}"
        assert not ('horizontal' in pred_list and 'clear_top_left' in pred_list), f"\n {dom} \n{lit_map[dom]}"
        assert not ('horizontal' in pred_list and 'clear_top_right' in pred_list), f"\n {dom} \n{lit_map[dom]}"
        assert not ('vertical' in pred_list and 'clear_top' in pred_list), f"\n {dom} \n{lit_map[dom]}"


        # Check left/clear_left
        assert not ('right' in pred_list and 'clear_left' in pred_list), f"\n {dom} \n{lit_map[dom]}"
        assert not ('left' in pred_list and 'clear_right' in pred_list), f"\n {dom} \n{lit_map[dom]}"

        for i in range(len(lit_map[dom])):
            pred, _args = lit_map[dom][i]

            # check mutuatilty for left and rigth
            if pred == 'left':
                checked = False
                for pred_right, args_right in lit_map[_args[0]]:
                    if pred_right == 'right':
                        assert args_right == (dom,), f"\n {dom} \n{lit_map[dom]}     \n {_args[0]} \n{lit_map[_args[0]]})"
                        checked = True
                assert checked
            elif pred == 'right':
                checked = False
                for pred_left, args_left in lit_map[_args[0]]:
                    if pred_left == 'left':
                        assert args_left == (dom,)
                        checked = True
                assert checked

            # Check exclusiveness for left/clear_left
            if pred == 'clear_left':
                for dom2 in lit_map.keys():
                    for j in range(len(lit_map[dom2])):
                        pred_left, args_left = lit_map[dom2][j]
                        if pred_left == 'left':
                            assert args_left != (dom,)
            elif pred == 'clear_right':
                for dom2 in lit_map.keys():
                    for j in range(len(lit_map[dom2])):
                        pred_right, args_right = lit_map[dom2][j]
                        if pred_right == 'right':
                            assert args_right != (dom,)


        # TODO: Add more here
    #import pdb;pdb.set_trace()
    return True




# Parse literals and group by object
from collections import defaultdict
import re

def parse_literal_map(literals):
    obj_map = defaultdict(list)
    for lit in literals:
        match = re.match(r"(\w+)\(([^)]+)\)", lit)
        if match:
            pred, args = match.group(1), match.group(2).split(",")
            args_clean = [a.strip().split(":")[0] for a in args]
            if args_clean:
                obj = args_clean[0]
                other_args = args_clean[1:]
                obj_map[obj].append((pred, tuple(other_args)))
    return dict(obj_map)