from lib.rendering.utils import fig2data

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
from config.config_loader import CONFIG


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



def get_objects_from_obs(obs):
    all_objs = set()
    discs = set()
    disc_pair_order = set()
    holding = None

    # A peg is a largest object
    for lit in obs:
        if lit.predicate.name == 'smaller':
            smaller, larger = lit.variables
            all_objs.update({larger, smaller})
            discs.add(smaller)
            disc_pair_order.add((larger, smaller))
        elif lit.predicate.name.lower() == "holding":
            holding = lit.variables[0]
            all_objs.add(lit.variables[0])

    pegs = sorted(all_objs - discs)

    # Get discs ordered by size
    discs_ordered_by_size = []
    while len(discs):
        for d1 in discs:
            is_next = True
            for d2 in discs:
                if d1 == d2:
                    continue
                if (d2, d1) in disc_pair_order:
                    is_next = False
                    break
            if is_next:
                break
        else:
            import ipdb; ipdb.set_trace()
        discs_ordered_by_size.append(d1)
        discs.remove(d1)

    # Get peg_to_disc_list
    on_links = {}
    for lit in obs:
        if lit.predicate.name == 'on':
            on_links[lit.variables[1]] = lit.variables[0]

    peg_to_disc_list = {}
    for peg in pegs:
        disc_list = []
        key = peg
        while key in on_links:
            disc_list.append(on_links[key])
            key = on_links[key]
        peg_to_disc_list[peg] = disc_list

    return pegs, discs_ordered_by_size, peg_to_disc_list, holding

def get_peg_params(pegs, width, height):
    peg_width = (width / 10.) / len(pegs)
    vertical_padding = height * 0.35
    peg_height = height - vertical_padding
    boundaries = np.linspace(0, width, len(pegs)+1)
    interval = (boundaries[1] - boundaries[0]) / 2
    peg_midpoints = boundaries[:-1] + interval
    peg_to_hor_midpoints = dict(zip(pegs, peg_midpoints))
    return peg_width, peg_height, peg_to_hor_midpoints

def get_disc_params(discs_ordered_by_size, peg_to_disc_list, peg_to_hor_midpoints, width, peg_height):
    num_pegs = len(peg_to_hor_midpoints)
    num_discs = len(discs_ordered_by_size)
    disc_height = (peg_height * 0.75) / num_discs

    horizontal_padding = width * 0.1
    max_disc_width = width / num_pegs - horizontal_padding
    min_disc_width = max_disc_width / 3
    all_disc_widths = np.linspace(max_disc_width, min_disc_width, num_discs)
    disc_widths = dict(zip(discs_ordered_by_size, all_disc_widths))

    disc_midpoints = {}
    for peg, discs in peg_to_disc_list.items():
        x = peg_to_hor_midpoints[peg]
        for i, disc in enumerate(discs):
            y = i * disc_height + disc_height / 2
            disc_midpoints[disc] = (x, y)

    return disc_height, disc_midpoints, disc_widths

def draw_pegs(ax, peg_width, peg_height, peg_to_hor_midpoints, height):
    peg_data = []
    for peg_name, midx in peg_to_hor_midpoints.items():
        x = midx - peg_width / 2
        y = 0
        rect = patches.Rectangle((x, y), peg_width, peg_height, 
            linewidth=1, edgecolor=(0.2, 0.2, 0.2), facecolor=(0.5, 0.5, 0.5))
        ax.add_patch(rect)

        # Collect peg data
        x1, y1 = data_to_pixel(x, y)
        x2, y2 = data_to_pixel(x + peg_width, y + peg_height)
        y1, y2 = y2, y1
        peg_data.append({
            'name': peg_name.split(':')[0],
            'object_class': 'peg',
            'color': (0.5, 0.5, 0.5),
            'bbox_pixel': (x1, y1, x2, y2)
        })
    
    return peg_data

def draw_discs(ax, disc_height, disc_midpoints, disc_widths, jitter_pos_xy_vec=[0.0, 0.0], jitter_size_xy_vec=[0.0, 0.0], train_set=True):
    disc_data = []
    for disc, (midx, midy) in disc_midpoints.items():
        color = block_name_to_color(disc.name, train_set=train_set)
        disc_width = disc_widths[disc]
        
        done = False
        while not done:
            x = midx - disc_width / 2
            y = midy - disc_height / 2

            # add noise to block position
            x += np.random.uniform(-jitter_pos_xy_vec[0], jitter_pos_xy_vec[0])
            y += np.random.uniform(-jitter_pos_xy_vec[1], jitter_pos_xy_vec[1])

            # add noise to disc size
            width_add = np.random.uniform(-jitter_size_xy_vec[0], jitter_size_xy_vec[0])
            height_add = np.random.uniform(-jitter_size_xy_vec[1], jitter_size_xy_vec[1])
            disc_width_new = max(disc_width/2, disc_width + width_add)
            disc_height_new = max(disc_height/2, disc_height + height_add)



            rect = patches.Rectangle((x,y), disc_width_new, disc_height_new, 
                linewidth=1, edgecolor=(0.2,0.2,0.2), facecolor=color)#(0.8,0.1,0.1))
            ax.add_patch(rect)

            # Collect disc data
            x1, y1 = data_to_pixel(x, y)
            x2, y2 = data_to_pixel(x + disc_width_new, y + disc_height_new)
            y1, y2 = y2, y1


            # check if block is outside of the scene
            if x1 >= x2 or y1 >= y2:
                continue

            disc_data.append({
                'name': disc.split(':')[0],
                'object_class': 'disc',
                'color': (0.8, 0.1, 0.1),
                'bbox_pixel': (x1, y1, x2, y2)
            })

            done = True
    
    return disc_data



def draw_robot(ax, robot_width, robot_height, midx, midy, holding, disc_width, disc_height, add_label=False, train_set=True):
    x = midx - robot_width / 2
    y = midy - robot_height / 2

    x1, y1 = data_to_pixel(x, y)
    x2, y2 = data_to_pixel(x + robot_width, y + robot_height)
    y1, y2 = y2, y1
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

    block_data = None
    if holding is not None:
        color = block_name_to_color(holding.name, train_set=train_set)
        x = midx - disc_width / 2
        y = (midy - robot_height / 2) - robot_height /2 - disc_height /2

        # Draw the held block
        rect = patches.Rectangle((x,y), disc_width, disc_height, 
            linewidth=1, edgecolor=(0.2,0.2,0.2), facecolor=color)#(0.8,0.1,0.1))
        ax.add_patch(rect)

        # Collect disc data
        x1, y1 = data_to_pixel(x, y)
        x2, y2 = data_to_pixel(x + disc_width, y + disc_height)
        y1, y2 = y2, y1

        block_data = ({
            'name': holding.name.split(':')[0],
            'object_class': 'disc',
            'color': (0.8, 0.1, 0.1),
            'bbox_pixel': (x1, y1, x2, y2)
        })



    return robot_data, block_data


def render(obs, mode='human', close=False, add_object_labels=False, env=None):
    import matplotlib
    matplotlib.use('Agg') # memory optimization


    # get rendering parameters from the environment
    try:
        jitter_pos_xy_vec = env.jitter_pos_xy_vec
        jitter_size_xy_vec = env.jitter_size_xy_vec
        train_set = env.train_set
    except:
        jitter_pos_xy_vec = [0.0, 0.0]
        jitter_size_xy_vec = [0.0, 0.0]
        train_set = True


    width, height = 4.2, 1.5
    width, height = 3.2, 3.2 #TODO: make fine
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0),
                                aspect='equal', frameon=False,
                                xlim=(-0.05, width + 0.05),
                                ylim=(-0.05, height + 0.05))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plt.NullFormatter())
        axis.set_major_locator(plt.NullLocator())

    pegs, discs_ordered_by_size, peg_to_disc_list, holding = get_objects_from_obs(obs)
    peg_width, peg_height, peg_to_hor_midpoints = get_peg_params(pegs, width, height)
    disc_height, disc_midpoints, disc_widths = get_disc_params(discs_ordered_by_size, 
        peg_to_disc_list, peg_to_hor_midpoints, width, peg_height)

    pegs_data = draw_pegs(ax, peg_width, peg_height, peg_to_hor_midpoints, height)
    discs_data = draw_discs(ax, disc_height, disc_midpoints, disc_widths, jitter_pos_xy_vec=jitter_pos_xy_vec, jitter_size_xy_vec=jitter_size_xy_vec, train_set=train_set) # get block information
    
    #synthesize scene data 
    obs_list = list(obs)
    scene_data = pegs_data + discs_data
    for i in range(len(scene_data)):
        scene_data[i]['unary_relationships'] = [] # format: list of predicates
        scene_data[i]['binary_relationships'] = {} # format: dict of predicates with the according list of other elements


        # traverse all literals to scan for relations
        for literal in obs_list: 
            vars = [var.name for var in literal.variables]
            entity = scene_data[i]['name']
            if entity not in vars:
                continue

            # unary relationships:
            if len(vars) <= 1:
                scene_data[i]['unary_relationships'].append(literal.predicate.name)
                continue
            
            # binary relationships:
            if entity != vars[0]: # skip if entity is not the first argument
                continue
            if literal.predicate in scene_data[i]['binary_relationships']:
                scene_data[i]['binary_relationships'][literal.predicate.name].extend(
                    vars[1:]
                )
            else:
                scene_data[i]['binary_relationships'][literal.predicate.name] = vars[1:]

    # render the robot
    block_width, block_height = disc_widths[holding].item() if holding else max([item[1].item() for item in disc_widths.items()]), disc_height
    robot_width = block_width * 1.8
    robot_midx = width / 2
    robot_height = height * 0.06
    robot_midy = height - robot_height/2
    robot_data, block_data = draw_robot(ax, robot_width, robot_height, robot_midx, robot_midy, holding,
        block_width, block_height, add_object_labels, train_set=train_set)
                
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
    scene_data.append(robot_data)

    data = fig2data(fig)
    plt.close(fig)
    return data, scene_data