import numpy as np
import sys, os, shutil


# local imports
from lib.sequence_generation.blocks.blocks import create_fresh_env as new_env_blocks
from lib.sequence_generation.ball_hiding.ball_hiding import create_fresh_env as new_env_ball_hiding
from lib.sequence_generation.hanoi.hanoi import create_fresh_env as new_env_hanoi
from lib.sequence_generation.hanoi_gripper.hanoi_gripper import create_fresh_env as new_env_hanoi_gripper
from lib.sequence_generation.dominoes.dominoes import create_fresh_env as new_env_dominoes
from lib.sequence_generation.general import generate_sequence, generate_train_test_split, write_metadata, write_bounding_boxes, write_actions
from lib.sequence_generation.slidetile.slidetile import create_fresh_env as new_env_slidetile
from lib.sequence_generation.gripper.gripper import create_fresh_env as new_env_gripper
from lib.sequence_generation.logistics.logistics import create_fresh_env as new_env_logistics
from lib.sequence_generation.sokoban.sokoban import create_fresh_env as new_env_sokoban
from lib.sequence_generation.grid.grid import create_fresh_env as new_env_grid

# global directory dependencies
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config_loader import CONFIG





def generate_dataset(config):
    """
    Generates a dataset of synthetic planning domains based on the provided configuration.
    Args:
        config (dict): Configuration dictionary.
    Returns:
        None
    """    
    # Configure constants
    DOMAIN_NAME = config["domain_name"]
    OUTPUT_DIR = config["output_dir"]
    NAME_EXTENSION = config["name_extension"]
    if NAME_EXTENSION != "":
        OUTPUT_DIR = f"{OUTPUT_DIR}/{DOMAIN_NAME}_{NAME_EXTENSION}"
    else:
        OUTPUT_DIR = f"{OUTPUT_DIR}/{DOMAIN_NAME}"

    # Constants
    TRAIN_RATIO = config["train_ratio"]
    NUM_SEQUENCES = config['num_sequences']
    config["train_imgs"] = [] # these are for storing the image file names for splitting
    config["test_imgs"] = []


    # Init planner
    from pddlgym_planners.fd import FD
    planner = FD()


    # create fresh environment
    if DOMAIN_NAME == "blocks":
        env, obs = new_env_blocks(config)
    elif DOMAIN_NAME == "hanoi":
        env, obs = new_env_hanoi(config)
    elif DOMAIN_NAME == "hanoi_gripper":
        env, obs = new_env_hanoi_gripper(config)
    elif DOMAIN_NAME == "dominoes":
        env, obs = new_env_dominoes(config)
    elif DOMAIN_NAME == "ball_hiding":
        env, obs = new_env_ball_hiding(config)
    elif DOMAIN_NAME == "slidetile":
        env, obs = new_env_slidetile(config)
    elif DOMAIN_NAME == "gripper":
        env, obs = new_env_gripper(config)
    elif DOMAIN_NAME == "logistics":
        env, obs = new_env_logistics(config)
    elif DOMAIN_NAME == "sokoban":
        env, obs = new_env_sokoban(config)
    elif DOMAIN_NAME == "grid":
        env, obs = new_env_grid(config)


    # Clean the output directory first or create if it does not exist
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # generate sequences
    import time
    time_per_sequence = []
    frames_object_information = []
    print("Generating dataset...")
    i = 0
    while i <= NUM_SEQUENCES:

        # log information about the generation process
        start_time = time.time()
        if len(time_per_sequence) == 0:
            time_mean = 1
        else:
            time_mean = np.mean(time_per_sequence)
        remaining_time = (NUM_SEQUENCES - i) * time_mean
        mins, secs = divmod(remaining_time, 60)
        print(f"\rSequence: {i}/{NUM_SEQUENCES}; time per sequence: {time_mean:.2f} sec; estimated remaining time: {int(mins)} min {int(secs)} sec", end="")

        #prefix name for frames in current sequence
        curr_im_dir = f"{i:06d}_"

        # generate sequence
        try:
            env, obs, single_frame_object_information, seq_imgs = generate_sequence(env, obs, config, curr_im_dir, planner)
        except Exception as e:
            print(f"\nError generating sequence {i}: {e}")
            print("Retrying with a new environment...")
            # create fresh environment
            if DOMAIN_NAME == "dominoes":
                env, obs = new_env_dominoes(config)
            elif DOMAIN_NAME == "ball_hiding":
                env, obs = new_env_ball_hiding(config)
            elif DOMAIN_NAME == "logistics":
                env, obs = new_env_logistics(config)
            elif DOMAIN_NAME == "sokoban":
                env, obs = new_env_sokoban(config)
            elif DOMAIN_NAME == "grid":
                env, obs = new_env_grid(config)
            elif DOMAIN_NAME == "blocks":
                env, obs = new_env_blocks(config)
            else:
                raise NotImplementedError(f"Environment {DOMAIN_NAME} not implemented for retrying.")
            continue

        # add respective image file names to train or test list for current sequence
        config["train_imgs" if (i+1) < TRAIN_RATIO * NUM_SEQUENCES else "test_imgs"].append(seq_imgs)

        # add set information
        for obj in single_frame_object_information:
            obj["set"] = "train" if (i+1) < TRAIN_RATIO * NUM_SEQUENCES else "test"
        frames_object_information.extend(single_frame_object_information)

        # determine whether next sequence is in train or test set
        test_set = (i+1) >= TRAIN_RATIO * NUM_SEQUENCES
            
        # create fresh environment
        if DOMAIN_NAME == "blocks":

            # randomize env until a valid one is found
            done = False
            while not done:
                env, obs = new_env_blocks(config, test_set)

                # Control the number of 'ontable' literals
                ontable_count = sum(1 for literal in obs.literals if literal.predicate.name == "ontable")
                done = ontable_count <= config["max_number_piles"]
        elif DOMAIN_NAME == "hanoi":
            env, obs = new_env_hanoi(config, test_set)
        elif DOMAIN_NAME == "hanoi_gripper":
            env, obs = new_env_hanoi_gripper(config, test_set)
        elif DOMAIN_NAME == "dominoes":
            env, obs = new_env_dominoes(config, test_set)
        elif DOMAIN_NAME == "ball_hiding":
            env, obs = new_env_ball_hiding(config, test_set)
        elif DOMAIN_NAME == "slidetile":
            env, obs = new_env_slidetile(config, test_set)
        elif DOMAIN_NAME == "gripper":
            env, obs = new_env_gripper(config, test_set)
        elif DOMAIN_NAME == "logistics":
            env, obs = new_env_logistics(config, test_set)
        elif DOMAIN_NAME == "sokoban":
            env, obs = new_env_sokoban(config, test_set)
        elif DOMAIN_NAME == "grid":
            env, obs = new_env_grid(config, test_set)


        # calculate time taken for this sequence (tqdm-like)
        time_per_sequence.append(time.time() - start_time)

        # increment sequence counter
        i += 1

    # print final message and time information
    print("\nDataset generation complete.")
    print(f"Total time taken: {sum(time_per_sequence):.2f} sec")
    print(f"Average time per sequence: {np.mean(time_per_sequence):.2f} sec")

    # write metadata
    write_metadata(config)
    
    # Write bounding boxes and relationships
    write_bounding_boxes(config, frames_object_information)

    # write actions 
    write_actions(config, frames_object_information)

    # generate train test split   
    generate_train_test_split(config)

    # save config file in annotations
    ANNO_OUTPUT_DIR = f"{OUTPUT_DIR}/annotations/"
    import json
    with open(f"{ANNO_OUTPUT_DIR}/dataset_generation_config_file.json", "w") as file:
        json.dump(config, file, indent=4)



def run():
    # Disabling warnings since gym logger is annoying when reloading the environment
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # Loading domain information from main config file
    config = CONFIG.dset

    # Load specific configuration from domain json file
    import json
    try:
        with open(config['domain_file'], "r") as file:
            config = json.load(file)
    except:
        print(f"Domain {config['domain_file']} is not supported.")
        sys.exit(1)

    # generate the dataset
    generate_dataset(config)

# Entry point of the script
if __name__ == "__main__":
    run()
