from lib.rendering.utils import fig2data, PileTracker

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import random

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
from config.config_loader import CONFIG


def get_objects_from_obs(obs, pile_tracker = None):
    on_links = {}
    pile_bottoms = set()
    all_objs = set()
    holding = None
    for lit in obs:
        if lit.predicate.name.lower() == "ontable":
            pile_bottoms.add(lit.variables[0])
            all_objs.add(lit.variables[0])
        elif lit.predicate.name.lower() == "on":
            on_links[lit.variables[1]] = lit.variables[0]
            all_objs.update(lit.variables)
        elif lit.predicate.name.lower() == "holding":
            holding = lit.variables[0]
            all_objs.add(lit.variables[0])
    all_objs = sorted(all_objs)

    bottom_to_pile = {}
    for obj in pile_bottoms:
        bottom_to_pile[obj] = [obj]
        key = obj
        while key in on_links:
            assert on_links[key] not in bottom_to_pile[obj]
            bottom_to_pile[obj].append(on_links[key])
            key = on_links[key]
    # check for number of piles
    if pile_tracker == None or len(all_objs) <= pile_tracker.max_piles: 
        piles = []
        for pile_base in all_objs:
            if pile_base in bottom_to_pile:
                piles.append(bottom_to_pile[pile_base])
            else:
                piles.append([])

        # add piles to pile_tracker
        for p_i, pile in enumerate(piles):
            if len(pile) == 0:
                continue
            for block in pile:
                pile_tracker.pile_mapping[block.name] = p_i
    else: # overflow in pile positions possible -> track pile positions
        piles = [[]] * pile_tracker.max_piles

        # clear / remove non-existing piles
        for pile_base in list(pile_tracker.pile_mapping):
            if pile_base not in [ele.name for ele in bottom_to_pile]:
                del pile_tracker.pile_mapping[pile_base]
        for pile_base in bottom_to_pile:
            if pile_base.name in pile_tracker.pile_mapping: # already assigned position
                piles[pile_tracker.pile_mapping[pile_base.name]]= bottom_to_pile[pile_base]
            else:  # randomly pick new position for the pile
                # index = random.randint(0, pile_tracker.max_piles-1)
                def rotate(l, n):
                    return l[-n:] + l[:-n]
                indices = list(range(0, pile_tracker.max_piles))
                pseudo_random_indices = {
                    'a': rotate(indices, 0),
                    'b': rotate(indices, 1),
                    'c': rotate(indices, 2),
                    'd': rotate(indices, 3),
                    'e': rotate(indices, 4),
                    'f': rotate(indices, 5),
                    'g': rotate(indices, 6),
                    'h': rotate(indices, 7),
                    'i': rotate(indices, 8),
                    'j': rotate(indices, 9),
                    'k': rotate(indices, 10),
                    'l': rotate(indices, 11)
                }
                i = 0
                index = pseudo_random_indices[pile_base.name][i]
                while piles[index] != [] or index in pile_tracker.pile_mapping.values():
                    #index = random.randint(0, pile_tracker.max_piles-1)
                    index = pseudo_random_indices[pile_base.name][i]
                    i += 1
                pile_tracker.pile_mapping[pile_base.name] = index
                piles[index]= bottom_to_pile[pile_base]
                
        # Check for duplicate values in pile_mapping
        if len(pile_tracker.pile_mapping.values()) != len(set(pile_tracker.pile_mapping.values())):
            raise ValueError("Duplicate pile positions detected in pile_mapping")
        
    return piles, holding

def get_block_params(piles, width, height, table_height, robot_height, num_blocks, pile_tracker):
    # Load configuration from JSON file
    config = CONFIG.dset

    import json
    try:
        with open(config['domain_file'], "r") as file:
            config = json.load(file)
    except:
        print("Error, domain not supported")
        import sys
        sys.exit(0)

    if config['fixed_size']:
        num_blocks = 7

    # Use the object classes from the config
    horizontal_padding = 0.025 * width
    block_width = width / num_blocks - 2*horizontal_padding
    block_height = (height - table_height - robot_height) / num_blocks - 0.05 * height

    block_positions = {}
    for pile_i, pile in enumerate(piles):
        if pile == []:
            continue
        x = horizontal_padding + pile_i * (block_width + 2*horizontal_padding) # old version
        if pile_tracker != None:
            if len(pile_tracker.x_positions) == 0:
                pile_tracker.generate_x_positions(horizontal_padding, block_width, width)
            try:
                x = pile_tracker.x_positions[pile_tracker.pile_mapping[pile[0].name]]
            except:
                import pdb;pdb.set_trace()
        for block_i, name in enumerate(pile):
            y = table_height + block_i * block_height
            block_positions[name] = (x, y)

    return block_width, block_height, block_positions

def draw_table(ax, width, table_height, add_label=False):
    rect = patches.Rectangle((0,0), width, table_height, 
        linewidth=1, edgecolor=(0.2,0.2,0.2), facecolor=(0.5,0.2,0.0))
    ax.add_patch(rect)

    if add_label:
        ax.text(width/10, table_height/4, "table", 
            ha='center', va='center', fontsize=10, weight='bold', color='black')

    x1, y1 = data_to_pixel(0,0)
    x2, y2 = data_to_pixel(width, table_height)
    y1, y2 = y2, y1
    return{
        'name': 'table',
        'object_class': 'table',
        'color': (0.5, 0.2, 0.0),
        'bbox_pixel': (x1, y1, x2, y2)
    }

def draw_robot(ax, robot_width, robot_height, midx, midy, holding, block_width, block_height, add_label=False, train_set=True):
    x = midx - robot_width / 2
    y = midy - robot_height / 2

    x1, y1 = data_to_pixel(x, y)
    x2, y2 = data_to_pixel(x + robot_width, y + robot_height)
    y1, y2 = y2, y1
    x1 = max(5, x1)
    x2 = min(475, x2)
    y1 = max(5, y1)
    y2 = min(475, y2)
    robot_data = {
        'name': 'robot',
        'object_class': 'robot',
        'color': (0.0, 0.0, 0.0),
        'bbox_pixel': (x1, y1, x2, y2)
    }
    # Draw robot base
    base_width = robot_width * 0.8  # Adjusted to fit inside the bounding box
    base_height = robot_height * 0.2  # Extend slightly over the block
    base_x = x + (robot_width - base_width) / 2  # Center the base within the bounding box
    base_y = y + robot_height - base_height  # Position the base at the top of the robot
    rect = patches.Rectangle((base_x, base_y), base_width, base_height, 
                             linewidth=1, edgecolor='black', facecolor=(0.0, 0.0, 0.0))
    ax.add_patch(rect)

    # Draw Gripper Fingers
    finger_width = robot_width * 0.1  # Narrow gripper fingers
    finger_height = robot_height  # Adjusted to fit inside the bounding box

    left_finger_x = x + (robot_width - base_width) / 2 - finger_width  # Adjusted to fit inside the bounding box
    right_finger_x = x + (robot_width + base_width) / 2  # Adjusted to fit inside the bounding box
    finger_y = y + (robot_height - finger_height) / 2  # Center the fingers vertically

    # Left Finger
    left_finger = patches.Rectangle((left_finger_x, finger_y), finger_width, finger_height, 
                                    linewidth=1, edgecolor='black', facecolor=(0.0, 0.0, 0.0))
    ax.add_patch(left_finger)

    # Right Finger
    right_finger = patches.Rectangle((right_finger_x, finger_y), finger_width, finger_height, 
                                     linewidth=1, edgecolor='black', facecolor=(0.0, 0.0, 0.0))
    ax.add_patch(right_finger)

    if add_label:
        ax.text(x - 0.6 * robot_width, y + 0.75 * robot_height, "robot", 
                ha='center', va='center', fontsize=10, weight='bold', color='black')

    block_data = None
    if holding is not None:
        is_block = str(holding).endswith('block')    
        
        holding_color = block_name_to_color(holding, train_set=train_set)
        ec = (0.2, 0.2, 0.2)
        holding_x = midx - block_width / 2
        holding_y = y - robot_height/2 - block_height / 2  # Positioned above the robot

        x1, y1 = data_to_pixel(holding_x, holding_y)
        x2, y2 = data_to_pixel(holding_x + block_width, holding_y + block_height)
        y1, y2 = y2, y1
        bbox_pixel = (x1, y1, x2, y2)
        if not is_block:
            wh_diff = int(abs((x2-x1) - (y2-y1))/2)
            bbox_pixel = (x1+wh_diff, y1, x2-wh_diff, y2)
        block_data = {
            'name': holding[:1],
            'object_class': 'block' if is_block else 'ball',
            'color': holding_color,
            'bbox_pixel': bbox_pixel,
        }

        # Draw the held object
        if is_block:
            rect = patches.Rectangle((holding_x, holding_y), block_width, block_height, 
                                linewidth=1, edgecolor=ec, facecolor=holding_color)
            ax.add_patch(rect)
        else:
            circle = patches.Circle((holding_x + block_width / 2, holding_y + block_height / 2),
                                    radius=block_height / 2, linewidth=1, edgecolor=ec, facecolor=holding_color)
            ax.add_patch(circle)

        # Add block label
        if add_label:
            ax.text(holding_x + block_width / 2, holding_y + block_height / 2, holding[0], 
                    ha='center', va='center', fontsize=10, weight='bold', color='black')

    return robot_data, block_data


_block_name_to_color = {}
def block_name_to_color(block_name, train_set=True):
    color_range = [0., 1.]

    # read color information config from config file
    config = CONFIG.dset

    import json    
    try:
        with open(config['domain_file'], "r") as file:
            config = json.load(file)
    except:
        print("Error, domain not supported")
        import sys
        sys.exit(0)
    if config['training_static_color']:
        if train_set:
            return (0.9, 0.1, 0.1)
    if train_set:
        color_range = config['train_color_range']
    else:
        color_range = config['test_color_range']
    
    # sample random color
    import random
    _rng = np.random.RandomState(random.randint(0, 1000))
    if block_name not in _block_name_to_color:
        if len(_block_name_to_color) == 0:
            if color_range[0] <= 0.1 and 0.9 <= color_range[1]:
                best_color = (0.9, 0.1, 0.1)
            else:
                best_color = _rng.uniform(color_range[0], color_range[1], size=3)
        else:
            # Generate 20 random colors and keep the one most different from prior colors
            best_color = None
            max_min_color_diff = 0.
            for _ in range(200):
                color = _rng.uniform(color_range[0], color_range[1], size=3)
                min_color_diff = np.inf
                for existing_color in _block_name_to_color.values():
                    diff = np.sum(np.subtract(color, existing_color)**2)
                    min_color_diff = min(diff, min_color_diff)
                if min_color_diff > max_min_color_diff:
                    best_color = color
                    max_min_color_diff = min_color_diff
        _block_name_to_color[block_name] = best_color
    return _block_name_to_color[block_name]

def draw_blocks(ax, block_width, block_height, block_positions, add_label=False, jitter_pos_xy_vec=[0.0, 0.0], jitter_size_xy_vec=[0.0, 0.0], train_set=True):
    block_data = []
    for block_name, (x, y) in block_positions.items():
        color = block_name_to_color(block_name, train_set=train_set)

        is_block = str(block_name).endswith('block')        
        
        done = False
        while not done:
            
            # add noise to block position
            x += np.random.uniform(-jitter_pos_xy_vec[0], jitter_pos_xy_vec[0])
            y += np.random.uniform(-jitter_pos_xy_vec[1], jitter_pos_xy_vec[1])

            # add noise to block size
            width_add = np.random.uniform(-jitter_size_xy_vec[0], jitter_size_xy_vec[0])
            height_add = np.random.uniform(-jitter_size_xy_vec[1], jitter_size_xy_vec[1])
            block_width = max(block_width/2, block_width + width_add)
            block_height = max(block_height/2, block_height + height_add)

            # Collect block data
            x1, y1 = data_to_pixel(x, y)
            x2, y2 = data_to_pixel(x+block_width, y+block_height)
            y1, y2 = y2, y1

            # check if block is outside of the scene
            if x1 >= x2 or y1 >= y2:
                continue

            # Add block data to the scene
            bbox_pixel = (x1, y1, x2, y2)
            if not is_block:
                wh_diff = int(abs((x2-x1) - (y2-y1))/2)
                bbox_pixel = (x1+wh_diff, y1, x2-wh_diff, y2)
            block_data.append({
                'name': block_name[:1],
                'object_class': 'block' if is_block else 'ball',
                'color': color,
                'bbox_pixel': bbox_pixel,
            })

            if x1 < 0 or x2 > 480.0 or y1 < 0 or y2 > 480.0:
                print("ERROR")
                import pdb;pdb.set_trace()
            if x1 >= x2 or y1 >= y2:
                print("ERROR")
                import pdb;pdb.set_trace()

            # Draw block rectangle
            if is_block:
                rect = patches.Rectangle((x, y), block_width, block_height, 
                                    linewidth=1, edgecolor=(0.2, 0.2, 0.2), facecolor=color)
                ax.add_patch(rect)
            else:
                circle = patches.Circle((x + block_width / 2, y + block_height / 2),
                                        radius=block_height / 2, linewidth=1, edgecolor=(0.2, 0.2, 0.2), facecolor=color)
                ax.add_patch(circle)

            # Add block label (centered in the rectangle)
            if add_label:
                ax.text(x + block_width / 2, y + block_height / 2, block_name[0], 
                    ha='center', va='center', fontsize=10, weight='bold', color='black')
            
            done = True
    
    return block_data

def data_to_pixel(x, y, width=3.2, height=3.2, dpi=150):
    """
    Convert data coordinates (x, y) to pixel coordinates.
    
    Args:
        x (float): X-coordinate in data space.
        y (float): Y-coordinate in data space.
        width (float): Width of the figure in inches.
        height (float): Height of the figure in inches.
        dpi (int): Dots per inch (DPI) of the figure.
        
    Returns:
        (int, int): Pixel coordinates (px_x, px_y).
    """
    # Define the matplotlib limits used in render()
    x_min, x_max = -0.05, width + 0.05
    y_min, y_max = -0.05, height + 0.05

    # Compute figure size in pixels
    img_width = int(width * dpi)
    img_height = int(height * dpi)

    # Convert to pixel coordinates
    px_x = int(((x - x_min) / (x_max - x_min)) * img_width)
    px_y = int(((y - y_min) / (y_max - y_min)) * img_height)  
    
    px_y = img_height - px_y# Invert Y-axis

    return px_x, px_y

def pixel_to_data(px_x, px_y, width=3.2, height=3.2, dpi=150):
    """
    Convert pixel coordinates (px_x, px_y) to data coordinates.
    
    Args:
        px_x (int): X-coordinate in pixel space.
        px_y (int): Y-coordinate in pixel space.
        width (float): Width of the figure in inches.
        height (float): Height of the figure in inches.
        dpi (int): Dots per inch (DPI) of the figure.
        
    Returns:
        (float, float): Data coordinates (x, y).
    """
    # Define the matplotlib limits used in render()
    x_min, x_max = -0.05, width + 0.05
    y_min, y_max = -0.05, height + 0.05

    # Compute figure size in pixels
    img_width = int(width * dpi)
    img_height = int(height * dpi)

    # Invert Y-axis
    px_y = img_height - px_y

    # Convert to data coordinates
    x = x_min + (px_x / img_width) * (x_max - x_min)
    y = y_min + (px_y / img_height) * (y_max - y_min)

    return x, y


def render(obs, add_object_labels=False, env=None, interpolation=None):

    # Compute the interpolation
    if interpolation is not None:
        # Compute next state
        import copy
        pile_tracker_copy = copy.deepcopy(env.pile_tracker)
        new_img, new_scene_data = render(obs, add_object_labels=add_object_labels, env=env)
        if len(env.pile_tracker.pile_mapping) < len(pile_tracker_copy.pile_mapping): # Do not let the new scene remove old pile 
            env.pile_tracker = pile_tracker_copy 


        # Get pre- and after-position of robot and held block
        block = None
        robot = None

        # Iterate through the new scene data and compare with the last state
        for new_item in new_scene_data:
            old_item = next((ele for ele in interpolation['last_state'] if ele['name'] == new_item['name']), None)

            # Check if the item is new or has changed
            add = False
            if set(old_item['unary_relationships']) != set(new_item['unary_relationships']):
                add = True
            elif set(old_item['binary_relationships']) != set(new_item['binary_relationships']):
                add = True
            elif old_item['bbox_pixel'] != new_item['bbox_pixel']:
                add = True
            
            if not add:
                continue

            # Remove item again if it is not the object that is interacted with
            if old_item['name'] != 'robot' and 'holding' not in old_item['unary_relationships'] and 'holding' not in new_item['unary_relationships']:
                add = False


            if add:
                if old_item['name'] == 'robot':
                    robot = {
                                'before': old_item,
                                'after': new_item
                            }
                else:
                    block = {
                                'before': old_item,
                                'after': new_item
                            }
                    
        # COMPUTE THE INTERPOLATION

        # Get important points of the trajectory
        intermediate_goal = data_to_pixel(0, 0)
        bbox = block['after']['bbox_pixel'] if 'holding' not in block['after']['unary_relationships'] else block['before']['bbox_pixel']
        intermediate_goal = bbox[0] - (bbox[2] - bbox[0]) / 2 + (bbox[2] - bbox[0])*.1, bbox[1] - (bbox[3] - bbox[1]) / 1
        offset = (0, 0)
        intermediate_goal = (intermediate_goal[0] + offset[0], intermediate_goal[1] + offset[1])
        
        start = robot['before']['bbox_pixel'][:2]
        start_x, start_y = pixel_to_data(start[0], start[1])
        goal_x, goal_y = pixel_to_data(intermediate_goal[0], intermediate_goal[1])
        
        trajectory = [
            #[start_x, start_y], # Skip this since the first frame is already rendered initially
            [goal_x, start_y],
            [goal_x, goal_y],
            [goal_x, start_y],
            [start_x, start_y]
        ]


        # Compute num of frames between points
        if interpolation['framerate'] < len(trajectory):
            raise ValueError("Interpolation framerate is too low for the number of points in the trajectory.")
        
        # Do the interpolation
        import copy
        bf = copy.deepcopy(trajectory)
        while len(trajectory) < interpolation['framerate']:
            new_traj = list()
            curr = trajectory[0]
            new_traj.append(curr)
            for ind, nxt in enumerate(trajectory[1:]):
                if len(trajectory) + ind < interpolation['framerate']:
                    new_traj.append([(curr[0] + nxt[0]) / 2, (curr[1] + nxt[1]) / 2])
                new_traj.append(nxt)
                curr = nxt
            trajectory = new_traj
        action_frame_index = trajectory.index([goal_x, goal_y])

        # Debug: visualize trajs
        visualize_trajs = False
        import matplotlib.pyplot as plt
        if visualize_trajs:
            import matplotlib.pyplot as plt

            # Visualize the trajectory before and after interpolation
            fig, ax = plt.subplots(figsize=(6, 6))

            # Plot the original trajectory points
            trajectory_np = np.array(trajectory)
            ax.plot(trajectory_np[:, 0], trajectory_np[:, 1], 'o-', label='Interpolated Trajectory', color='blue')

            # Plot the "before" trajectory points
            bf_np = np.array(bf)
            ax.plot(bf_np[:, 0], bf_np[:, 1], 'x--', label='Original Trajectory (Before)', color='orange')

            # Annotate the start and end points for both trajectories
            ax.annotate('Start', (trajectory_np[0, 0], trajectory_np[0, 1]), textcoords="offset points", xytext=(-10, 10), ha='center', color='green')
            ax.annotate('End', (trajectory_np[-1, 0], trajectory_np[-1, 1]), textcoords="offset points", xytext=(-10, -15), ha='center', color='red')

            ax.annotate('Start (Before)', (bf_np[0, 0], bf_np[0, 1]), textcoords="offset points", xytext=(-10, 10), ha='center', color='darkgreen')
            ax.annotate('End (Before)', (bf_np[-1, 0], bf_np[-1, 1]), textcoords="offset points", xytext=(-10, -15), ha='center', color='darkred')

            # Set plot properties
            ax.set_title("Trajectory Visualization")
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")
            ax.legend()
            ax.grid(True)
            # Save the plot to a file
            plt.savefig("trajectory_visualization.png")

            import pprint
            print("old")
            pprint.pprint(bf)
            print()
            print("interpolated")
            pprint.pprint(trajectory)
            print()
            import pdb;pdb.set_trace()
        

        # Execute current interpolation frame
        frame = interpolation['frame_current']
        interpol_robot_pos = trajectory[frame]
        
        # Change state after action
        if frame < action_frame_index:
            obs = interpolation['last_obs']
        elif frame >= action_frame_index:
            obs = obs



    # START RENDERING ----------
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg') # memory optimization

    width, height = 3.2, 3.2
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0),
                                aspect='equal', frameon=False,
                                xlim=(-0.05, width + 0.05),
                                ylim=(-0.05, height + 0.05))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plt.NullFormatter())
        axis.set_major_locator(plt.NullLocator())

    table_height = height * 0.15
    robot_height = height * 0.06

    try:
        jitter_pos_xy_vec = env.jitter_pos_xy_vec
        jitter_size_xy_vec = env.jitter_size_xy_vec
        train_set = env.train_set
        num_blocks = env.num_blocks
        RENDER_REALISTICALLY = env.render_realistically
        if num_blocks < 4:
            num_blocks = 4
        elif num_blocks > 7:
            num_blocks = 7
    except:
        jitter_pos_xy_vec = [0.0, 0.0]
        jitter_size_xy_vec = [0.0, 0.0]
        train_set = True


    piles, holding = get_objects_from_obs(obs, pile_tracker=env.pile_tracker) # if pile tracker is set, pass it to get_objects_from_obs function
    #piles, holding = get_objects_from_obs(obs) # old version
    #raise ValueError("Error in get_objects_from_obs function. Please check the input data.")

    block_width, block_height, block_positions = get_block_params(piles, width, height, 
        table_height, robot_height, num_blocks, env.pile_tracker)
    robot_width = block_width * 1.8
    robot_midx = width / 2
    robot_midy = height - robot_height/2

    #synthesize scene data 
    obs_list = list(obs)
    scene_data = draw_blocks(ax, block_width, block_height, block_positions, add_object_labels, jitter_pos_xy_vec=jitter_pos_xy_vec, jitter_size_xy_vec=jitter_size_xy_vec, train_set=train_set) # get block information
    # add relationships information
    for i, typed_entity in enumerate(list(block_positions.keys())):
        scene_data[i]['unary_relationships'] = [] # format: list of predicates
        scene_data[i]['binary_relationships'] = {} # format: dict of predicates with the according list of other elements

        # traverse all literals to scan for relations
        for literal in obs_list: 
            if typed_entity not in literal.variables:
                continue

            # unary relationships:
            if len(literal.variables) <= 1:
                scene_data[i]['unary_relationships'].append(literal.predicate.name)
                continue
            
            # binary relationships:
            if typed_entity != literal.variables[0]: # skip if entity is not the first argument
                continue
            if literal.predicate in scene_data[i]['binary_relationships']:
                scene_data[i]['binary_relationships'][literal.predicate.name].extend(
                    [var.name for var in literal.variables[1:]]
                )
            else:
                scene_data[i]['binary_relationships'][literal.predicate.name] = [
                    var.name for var in literal.variables[1:]
                ]
    # other objects
    table_data = draw_table(ax, width, table_height, add_object_labels)


    if interpolation is not None:
        # Change positions of robot (interpolated)
        robot_midx = interpol_robot_pos[0] + robot_width / 2
        robot_midy = interpol_robot_pos[1] - robot_height / 2 


    robot_data, block_data = draw_robot(ax, robot_width, robot_height, robot_midx, robot_midy, holding,
        block_width, block_height, add_object_labels, train_set=train_set)
    table_data['unary_relationships'] = []
    table_data['binary_relationships'] = {} 
    robot_data['unary_relationships'] = [i.predicate.name for i in obs_list if i.predicate in ["handfull", "handempty"]]
    robot_data['binary_relationships'] = {}
    
    if block_data != None:
        block_data['unary_relationships'] = []
        block_data['binary_relationships'] = {} 
        # traverse all literals to scan for relations
        for literal in obs_list: 
            if holding not in literal.variables:
                continue

            # unary relationships:
            if len(literal.variables) <= 1:
                block_data['unary_relationships'].append(literal.predicate.name)
                continue
            
            # binary relationships:
            if holding != literal.variables[0]: # skip if entity is not the first argument
                continue
            if literal.predicate in block_data['binary_relationships']:
                block_data['binary_relationships'][literal.predicate.name].extend(
                    [var.name for var in literal.variables[1:]]
                )
            else:
                block_data['binary_relationships'][literal.predicate.name] = [
                    var.name for var in literal.variables[1:]
                ]
        scene_data.append(block_data)
    scene_data.append(table_data)
    scene_data.append(robot_data)

    data = fig2data(fig)
    plt.close(fig)


    # RENDER THE SCENE REALISTICALLY
    if not RENDER_REALISTICALLY:
        if interpolation is not None:
            return (data, (scene_data, action_frame_index))
        else:
            return data, scene_data    
    # Get the scene data
    blocks_data = []
    for block in scene_data:
        if block['object_class'] == 'block':
            blocks_data.append(block)
    # Get Pile Information
    pile_data = piles
    for i, pile in enumerate(pile_data):
        if pile == []:
            continue
        for j, block in enumerate(pile):
            pile_data[i][j] = block.name
    # Convert ndarray values to lists for JSON serialization
    import json
    for block in blocks_data:
        if isinstance(block['color'], np.ndarray):
            block['color'] = block['color'].tolist()
    with open("lib/rendering/blocks/realistic/input.json", "w") as json_file:
        json.dump(
            [blocks_data, pile_data, [holding], robot_data],
            json_file,
            indent=4
        )
    # Run the realistic image generator using the run.sh script
    output_file = "lib/rendering/blocks/realistic/output.png"
    run_script = "lib/rendering/blocks/realistic/run.sh"
    try:
        import subprocess
        with open(os.devnull, 'w') as devnull:
            subprocess.run(["bash", run_script, output_file], check=True, stdout=devnull, stderr=devnull)
    except subprocess.CalledProcessError as e:
        pass # This is just a warning, we can ignore it
        #print(f"Error occurred while running the realistic image generator: {e}")

    # Replace data and scene_data with the realistic image
    from PIL import Image
    try:
        data = np.array(Image.open(output_file))
    except Exception as e:
        print(f"Error occurred while reading the realistic image: {e}")
        raise ValueError("Error in reading the realistic image. Please check the output file.")
    

    # Read updated coordinates of the blocks
    with open('lib/rendering/blocks/realistic/output.json', 'r') as file:
        block_data = json.load(file)
    scene_data_upated = []
    for i, item in enumerate(scene_data):
        updated_item = next((ele for ele in block_data if ele['name'] == item['name']), None)
        item['bbox_pixel'] = updated_item['bbox_pixel']
        scene_data_upated.append(item)
    scene_data = scene_data_upated


    if interpolation is not None:
        return (data, (scene_data, action_frame_index))
    else: 
        return data, scene_data