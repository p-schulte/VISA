import lib.rendering.xml_parser as xml_parser

def render_state_img(img, scene_data, output_dir, file_name, render_annotations=False):
    """
    Renders the current state of the environment and saves it as an image file.
    Args:
        img: The image to be saved.
        output_dir (str): The directory where the image file will be saved.
        file_name (str): The name of the image file to be saved.
    Returns:
        None
    """

    if render_annotations:
        # Draw bounding boxes on the image
        import cv2
        for obj in scene_data:
            x, y2, x2, y = obj['bbox_pixel']
            cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0, 255), 2)  # Draw a red bounding box

        # Save the image with bounding boxes
        #imageio.imsave(f"{output_dir}/{file_name}.png", img)
        import cv2
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        cv2.imwrite(f"{output_dir}/{file_name}", img)
    else:
        #imageio.imsave(f"{output_dir}/{file_name}", img)
        import cv2
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        cv2.imwrite(f"{output_dir}/{file_name}", img)
    del img


def extract_scene_information(scene_data):
    """
    Extracts and processes scene information from the given environment.
    This function renders the environment with object labels and processes the 
    scene data to ensure all elements are XML serializable. NumPy arrays and 
    tuples within the scene data are converted to lists.
    Args:
        scene_data: The scene data containing information about the objects in the environment.
    Returns:
        A list of dictionaries containing the scene data, with all elements 
        converted to be XML serializable.
    """
    import numpy as np
    
    # Convert NumPy arrays to lists for XML serialization
    def convert_for_xml(obj):
        if isinstance(obj, np.ndarray):  # Convert NumPy arrays to lists
            return obj.tolist()
        elif isinstance(obj, tuple):  # Convert tuples to lists
            return list(obj)
        return obj

    # Process scene_data to ensure all elements are XML serializable
    scene_data_serializable = [
        {key: convert_for_xml(value) for key, value in obj.items()}
        for obj in scene_data
    ]

    return scene_data_serializable

def encode_state_xml(scene_data, output_dir, file_name):
    """
    Encode the state of the environment into an XML file and optionally render annotations.
    Args:

    Returns:
        None
    This function performs the following steps:
    1. Renders the environment to obtain an image and scene data.
    2. Converts NumPy arrays and tuples in the scene data to lists for XML serialization.
    3. Encodes the scene data into an XML file and saves it to the specified directory.
    4. If render_annotations is True, draws bounding boxes on the image and saves the annotated image.
    """
    scene_data_serializable = extract_scene_information(scene_data)

    # Save scene data as xml file
    xml_parser.encode_scene_xml(scene_data_serializable, output_dir, file_name)



def generate_sequence(env, obs, config, frame_prefix, planner):
    """
    Generate a sequence of images and annotations by simulating actions in the environment.
    This function generates a sequence of images and corresponding annotations by simulating
    actions in the given environment. It ensures that each state is unique by keeping track
    of visited states and avoids repeating patterns by shuffling possible actions.
    Args:
        env: The environment object that supports the PDDL action space.
        obs: The initial observation of the environment.
        config (dict): A dictionary containing the configuration parameters.
        frame_prefix (str): The prefix to be used for the frame names.
    Returns:
        tuple: A tuple containing the final environment and observation after generating the sequence.
    """
    import os
    import copy
    import random

    # constants
    DOMAIN_NAME = config["domain_name"]
    OUTPUT_DIR = config["output_dir"]
    NAME_EXTENSION = config["name_extension"]
    if NAME_EXTENSION != "":
        OUTPUT_DIR = f"{OUTPUT_DIR}/{DOMAIN_NAME}_{NAME_EXTENSION}"
    else:
        OUTPUT_DIR = f"{OUTPUT_DIR}/{DOMAIN_NAME}"
    IMG_OUTPUT_DIR = f"{OUTPUT_DIR}/frames/"    
    ANNO_XML_OUTPUT_DIR = f"{OUTPUT_DIR}/annotations/xml/"  
    RENDER_ANNOTATIONS = config['render_annotated_images']
    SEQUENCE_LENGTH_METHOD = config['sequence_length_method']
    if SEQUENCE_LENGTH_METHOD == "fixed":
        NUM_IMG_PER_SEQUENCE_RANGE = config['num_img_per_sequence']
        if type(NUM_IMG_PER_SEQUENCE_RANGE) == int:
            NUM_IMG_PER_SEQUENCE_RANGE = (NUM_IMG_PER_SEQUENCE_RANGE, NUM_IMG_PER_SEQUENCE_RANGE)
        NUM_IMG_PER_SEQUENCE = random.choice(range(NUM_IMG_PER_SEQUENCE_RANGE[0], NUM_IMG_PER_SEQUENCE_RANGE[1]+1))
    USE_PLANNER = config['use_planner']
    USE_INTERPOLATION = config['interpolation']['use_interpolation']
    if USE_INTERPOLATION:
        INTERPOLATION_METHOD = config['interpolation']['method']
        INTERPOLATION_NUM_FRAMES = config['interpolation']['num_frames']

    MAX_IMG_PER_SEQUENCE = config['max_img_per_sequence']


    # Maintaining order of objects
    num_objs = len(obs.objects) - 4 # exclude grippers and rooms
    indices = list(range(num_objs))
    random.shuffle(indices)
    env.obj_order = indices

    # Maintaining order of elements on truck, plane and city
    env.load_order = { "truckred": [None for i in range(5)], "planeblue": [None for i in range(5)], "city1-1": [None for i in range(5)], "city2-1": [None for i in range(5)], "city1-2": [None for i in range(5)], "city2-2": [None for i in range(5)] }

    # rendering the current frame 
    img, scene_data = env.render(add_object_labels=RENDER_ANNOTATIONS, env=env)
    last_state = copy.deepcopy(scene_data)
    last_obs = copy.deepcopy(obs)

    # saving object information
    frames_object_info = []
    frame_information = {'objects': extract_scene_information(scene_data)} # first frame
    frame_name = f"{frame_prefix}{1:06d}.png"
    frame_name_xml = f"{frame_prefix}{1:06d}.xml"
    frame_information['frame_name'] = frame_name
    frame_information['action'] = ""
    frames_object_info.append(frame_information)


    os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)
    os.makedirs(ANNO_XML_OUTPUT_DIR, exist_ok=True)

    # Render initial state
    render_state_img(img, scene_data, IMG_OUTPUT_DIR, frame_name, render_annotations=RENDER_ANNOTATIONS)
    encode_state_xml(scene_data, ANNO_XML_OUTPUT_DIR, frame_name_xml)


    # Generate Action with Planner
    if USE_PLANNER:
        plan = planner(env.domain, obs)
        if SEQUENCE_LENGTH_METHOD == "planner":
            NUM_IMG_PER_SEQUENCE = len(plan) + 1
        
    # Use frozenset to store state representations
    visited_states = set()
    visited_states.add(frozenset(obs.literals))  # Store the initial state
    state_counter = 2
    img_counter = 2
    while state_counter <= NUM_IMG_PER_SEQUENCE:

        if img_counter >= MAX_IMG_PER_SEQUENCE or state_counter > MAX_IMG_PER_SEQUENCE:
            NUM_IMG_PER_SEQUENCE = img_counter
            break
        
        possible_actions = list(env.action_space.all_ground_literals(obs, valid_only=True))  # Get all valid actions
        random.shuffle(possible_actions)  # Shuffle the actions to avoid repeating patterns
        
        valid_action = None

        # set mode
        mode = "planner" if USE_PLANNER else "random"

        # Use planned action
        if mode == "planner":
            if state_counter-2 >= len(plan):
                mode = "random"
                if state_counter-2 == len(plan):
                    print("No more actions in plan. Now generating random actions.")
            else:
                valid_action = plan[state_counter-2]

        # Generate Random Action
        if mode == "random":
            for ind, action in enumerate(possible_actions):
                # print(f"{ind}/{len(possible_actions)} {action}", end="")
                temp_env = copy.deepcopy(env)  # Simulate action on a copy
                try:
                    temp_obs, _, temp_done, _, _ = temp_env.step(action)
                except:
                    print(action)
                    import pdb;pdb.set_trace()
                    temp_obs, _, temp_done, _, _ = temp_env.step(action)

                # check if enough piles are available
                if DOMAIN_NAME == "blocks":
                    ontable_count = sum(1 for literal in temp_obs.literals if literal.predicate.name == "ontable")
                    if ontable_count > config['max_number_piles']:
                        continue
                elif DOMAIN_NAME == "hanoi":
                    pass # TODO: implement hanoi
                elif DOMAIN_NAME == "dominoes":
                    # Checking if the action is valid
                    temp_env.render(add_object_labels=RENDER_ANNOTATIONS, env=temp_env)

                
                if frozenset(temp_obs.literals) not in visited_states and temp_obs != obs:
                    valid_action = action
                    break  # Use the first valid action found
                    
        if valid_action is None:
            visited_states.clear()  # Reset visited states so all actions are available again
            visited_states.add(frozenset(obs.literals))  # Store the current state
            continue  # Retry from the same step without incrementing i

        # Execute the chosen action
        prev_obs = copy.deepcopy(obs)
        obs, reward, done, truncated, debug_info = env.step(valid_action)

        # Test for inconsistencies
        if DOMAIN_NAME == "dominoes":
            from lib.sequence_generation.dominoes.dominoes import test_obs_legit
            test_obs_legit(obs)
        visited_states.add(frozenset(obs.literals))  # Store the new state

        # INTERPOLATE
        if USE_INTERPOLATION:
            for frame in range(INTERPOLATION_NUM_FRAMES):
                # rendering the current frame 
                interpolation = {
                    'framerate': INTERPOLATION_NUM_FRAMES,
                    'frame_current': frame,
                    'last_state': copy.deepcopy(last_state),
                    'last_obs': copy.deepcopy(last_obs).literals,
                    'new_action': valid_action.predicate,
                }

                img, (scene_data, action_frame_index) = env.render(add_object_labels=RENDER_ANNOTATIONS, env=env, interpolation = interpolation)
                if frame == INTERPOLATION_NUM_FRAMES-1:
                    last_state = copy.deepcopy(scene_data)
                    last_obs = copy.deepcopy(obs)

                # Determine action name and variables
                if frame == action_frame_index:
                    predicate = valid_action.predicate.name
                    variables = valid_action.variables
                elif frame < action_frame_index:
                    predicate = 'approaching'
                    if valid_action.predicate.name in ['pickup', 'unstack']:
                        variables = [valid_action.variables[0]]
                    elif valid_action.predicate.name in ['stack']:
                        variables = [valid_action.variables[1]]
                    else:# approaching table
                        predicate = 'approaching_table'
                        variables = []
                        
                elif frame > action_frame_index:
                    predicate = 'resetting'
                    variables = []
                
                # save object information
                frame_name = f"{frame_prefix}{img_counter:06d}.png"
                frame_name_xml = f"{frame_prefix}{img_counter:06d}.xml"
                frame_information = {'objects': extract_scene_information(scene_data)}
                frame_information['frame_name'] = frame_name
                frame_information['action'] = {
                    'predicate': predicate,
                    'args': [
                        {
                            'class': str(var.var_type),
                            'identifier': str(var.name)
                        } for var in variables
                    ]
                }
                frames_object_info.append(frame_information)

                # Render the state
                render_state_img(img, scene_data, IMG_OUTPUT_DIR, frame_name, render_annotations=RENDER_ANNOTATIONS)
                encode_state_xml(scene_data, ANNO_XML_OUTPUT_DIR, frame_name_xml)
                img_counter+=1

            state_counter+=1




        else:
            img, scene_data = env.render(add_object_labels=RENDER_ANNOTATIONS, env=env)

            # save object information
            frame_information = {'objects': extract_scene_information(scene_data)}
            frame_name = f"{frame_prefix}{state_counter:06d}.png"
            frame_name_xml = f"{frame_prefix}{state_counter:06d}.xml"
            frame_information['frame_name'] = frame_name
            predicate = valid_action.predicate.name
            variables = valid_action.variables

            if env.env.domain.domain_name == "gripper": 
                if len(variables) == 3:
                    variables = variables[:-1] # remove room variable for gripper actions


            if DOMAIN_NAME == "sokoban":
                from lib.sequence_generation.sokoban.utils import detect_action
                action, arguments = detect_action(prev_obs, obs)
                args_identifiers = [arg.split(":")[0] for arg in arguments]
                args_classes = [arg.split(":")[1] for arg in arguments]

                frame_information['action'] = {
                    'predicate': action,
                    'args': [
                        {
                            'class': arg_class,
                            'identifier': arg_identifier
                        } for arg_class, arg_identifier in zip(args_classes, args_identifiers)
                    ]
                }
            elif DOMAIN_NAME == "grid":
                from lib.sequence_generation.grid.action_extractor import detect_action
                action, arg = detect_action(prev_obs, obs)
                arg_identifier = arg.split(":")[0]
                arg_class = arg.split(":")[1]

                frame_information['action'] = {
                    'predicate': action,
                    'args': [
                        {
                            'class': arg_class,
                            'identifier': arg_identifier
                        }
                    ]
                }
            else:
                frame_information['action'] = {
                    'predicate': predicate,
                    'args': [
                        {
                            'class': str(var.var_type),
                            'identifier': str(var.name)
                        } for var in variables
                    ]
                }
            frames_object_info.append(frame_information)

            # Render the state
            render_state_img(img, scene_data, IMG_OUTPUT_DIR, frame_name, render_annotations=RENDER_ANNOTATIONS)
            encode_state_xml(scene_data, ANNO_XML_OUTPUT_DIR, frame_name_xml)
            state_counter+=1

    if USE_INTERPOLATION:
        NUM_IMG_PER_SEQUENCE = img_counter - 1

    return env, obs, frames_object_info, NUM_IMG_PER_SEQUENCE


def generate_train_test_split(config, randomize_sets=False):
    """
    Generates train/test split files for the dataset.
    Args:
        config (dict): A dictionary containing the configuration parameters.
        randomize_sets (bool, optional): If True, randomize the order of the sequences. Defaults to False.
    Returns:
        None
    """
    DOMAIN_NAME = config["domain_name"]
    OUTPUT_DIR = config["output_dir"]
    NAME_EXTENSION = config["name_extension"]
    if NAME_EXTENSION != "":
        OUTPUT_DIR = f"{OUTPUT_DIR}/{DOMAIN_NAME}_{NAME_EXTENSION}"
    else:
        OUTPUT_DIR = f"{OUTPUT_DIR}/{DOMAIN_NAME}"
    TRAIN_RATIO = config["train_ratio"]
    NUM_SEQUENCES = config['num_sequences']
    SET_OUTPUT_DIR = f"{OUTPUT_DIR}/ImageSets/Main/" 

    import random
    import os

    # Create output directory if it doesn't exist
    os.makedirs(SET_OUTPUT_DIR, exist_ok=True)

    # Generate image indices
    indices = list(range(0, NUM_SEQUENCES + 0))

    # Shuffle indices
    if randomize_sets:
        random.seed(42)
        random.shuffle(indices)

    # Split indices into train and test sets
    train_size = int(TRAIN_RATIO * NUM_SEQUENCES)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Write trainval.txt
    with open(os.path.join(SET_OUTPUT_DIR, "trainval.txt"), "w") as f:
        for idx, img_nums in enumerate(config['train_imgs']):
            for idx_im in range(1, img_nums+1):
                f.write(f"{idx:06d}_{idx_im:06d}\n")

    # Write test.txt
    with open(os.path.join(SET_OUTPUT_DIR, "test.txt"), "w") as f:
        for idx, img_nums in enumerate(config['test_imgs']):
            idx += len(config['train_imgs']) # adjust index for test set
            for idx_im in range(1, img_nums+1):
                f.write(f"{idx:06d}_{idx_im:06d}\n")


def write_metadata(config):
    """
    Creates metadata files for the dataset.
    Args:
        config (dict): A dictionary containing the configuration parameters.
    Returns:
        None
    """
    DOMAIN_NAME = config["domain_name"]
    OUTPUT_DIR = config["output_dir"]
    NAME_EXTENSION = config["name_extension"]
    if NAME_EXTENSION != "":
        OUTPUT_DIR = f"{OUTPUT_DIR}/{DOMAIN_NAME}_{NAME_EXTENSION}"
    else:
        OUTPUT_DIR = f"{OUTPUT_DIR}/{DOMAIN_NAME}"
    ANNO_OUTPUT_DIR = f"{OUTPUT_DIR}/annotations/"
    
    import os
    os.makedirs(ANNO_OUTPUT_DIR, exist_ok=True)

    # Create frame_list.txt
    with open(os.path.join(ANNO_OUTPUT_DIR, "frame_list.txt"), "w") as f:
        for idx, img_nums in enumerate(config['train_imgs']):
            for idx_im in range(1, img_nums+1):
                f.write(f"{idx:06d}_{idx_im:06d}.png\n")
        for idx, img_nums in enumerate(config['test_imgs']):
            idx += len(config['train_imgs']) # adjust index for test set
            for idx_im in range(1, img_nums+1):
                f.write(f"{idx:06d}_{idx_im:06d}.png\n")

    # Create object_classes.txt
    object_classes = config["object_classes"]
    with open(os.path.join(ANNO_OUTPUT_DIR, "object_classes.txt"), "w") as f:
        for obj_class in object_classes:
            f.write(f"{obj_class}\n")

    # Create relationship_classes.txt
    relationship_classes = config["relationship_classes"]  # Example relationship classes
    with open(os.path.join(ANNO_OUTPUT_DIR, "relationship_classes.txt"), "w") as f:
        for rel_class in relationship_classes:
            f.write(f"{rel_class}\n")

    # Create action_name_classes.txt
    action_name_classes = config["action_name_classes"]  # Example relationship classes
    with open(os.path.join(ANNO_OUTPUT_DIR, "action_name_classes.txt"), "w") as f:
        for acn_class in action_name_classes:
            f.write(f"{acn_class}\n")


def write_bounding_boxes(config, scene_information):
    """
    Creates empty bounding box and relationship files for the dataset.
    Args:
        config (dict): A dictionary containing the configuration parameters.
        scene_information (list): A list of dictionaries containing the scene data.
    Returns:
        None
    """
    DOMAIN_NAME = config["domain_name"]
    OUTPUT_DIR = config["output_dir"]
    NAME_EXTENSION = config["name_extension"]
    if NAME_EXTENSION != "":
        OUTPUT_DIR = f"{OUTPUT_DIR}/{DOMAIN_NAME}_{NAME_EXTENSION}"
    else:
        OUTPUT_DIR = f"{OUTPUT_DIR}/{DOMAIN_NAME}"
    ANNO_OUTPUT_DIR = f"{OUTPUT_DIR}/annotations/"


    import pickle
    import os
    import json

    os.makedirs(ANNO_OUTPUT_DIR, exist_ok=True)
    # create format for serialization
    frame_list = {}
    for frame_information in scene_information:
        frame = []
        # extract information about frame
        for obj in frame_information['objects']:
            object_info = {
                'class': obj['object_class'],
                'identifier': obj['name'],
                'bbox': obj['bbox_pixel'],
                'binary_relationships': obj['binary_relationships'],
                'unary_relationships': obj['unary_relationships'],
                'metadata': {
                    #'tag': f"{frame_information['frame_name'].split('/')[0]}/{obj['object_class']}/{frame_information['frame_name'].split('/')[1]}",
                    'set': frame_information['set'],
                    'color': obj['color'],
                },
                'visible': "true"
            }
            frame.append(object_info)

        # add frame information to list under frame name
        frame_list[frame_information['frame_name']] = frame

    # Save frame_list to a JSON file
    with open(os.path.join(ANNO_OUTPUT_DIR, "object_bbox_and_relationship.json"), "w") as json_file:
        json.dump(frame_list, json_file, indent=4)

    # Create empty object_bbox_and_relationship.pkl
    object_bbox_and_relationship = frame_list
    with open(os.path.join(ANNO_OUTPUT_DIR, "object_bbox_and_relationship.pkl"), "wb") as f:
        pickle.dump(object_bbox_and_relationship, f)


def write_actions(config, scene_information):
    """
    Creates action description files for the dataset.
    Args:
        config (dict): A dictionary containing the configuration parameters.
        scene_information (list): A list of dictionaries containing the scene data.
    Returns:
        None
    """
    DOMAIN_NAME = config["domain_name"]
    OUTPUT_DIR = config["output_dir"]
    NAME_EXTENSION = config["name_extension"]
    if NAME_EXTENSION != "":
        OUTPUT_DIR = f"{OUTPUT_DIR}/{DOMAIN_NAME}_{NAME_EXTENSION}"
    else:
        OUTPUT_DIR = f"{OUTPUT_DIR}/{DOMAIN_NAME}"
    ANNO_OUTPUT_DIR = f"{OUTPUT_DIR}/annotations/"

    import pickle
    import os
    import json

    os.makedirs(ANNO_OUTPUT_DIR, exist_ok=True)
    # create format for serialization
    frame_list = {}
    for frame_information in scene_information:
        if 'action' not in frame_information or frame_information['action'] == "": # first frame
            frame_list[frame_information['frame_name']] = {'predicate': "", 'args': [], 'metadata': {'set': frame_information['set']}}
            continue 

        frame = {
            'predicate': frame_information['action']['predicate'],
            'args': [
                {
                'class': arg['class'],
                'identifier': arg['identifier']
                } for arg in frame_information['action']['args']
            ],
            'metadata': {
                'set': frame_information['set']
            },
        }

        # add frame information to list under frame name
        frame_list[frame_information['frame_name']] = frame

    # Save frame_list to a JSON file
    with open(os.path.join(ANNO_OUTPUT_DIR, "action_descriptions.json"), "w") as json_file:
        json.dump(frame_list, json_file, indent=4)

    # Create action_descriptions.pkl
    with open(os.path.join(ANNO_OUTPUT_DIR, "action_descriptions.pkl"), "wb") as f:
        pickle.dump(frame_list, f)