from lib.rendering.utils import fig2data, PileTracker

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import random

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
from config.config_loader import CONFIG

from collections import defaultdict, deque
def layout_scene_from_base_layer(base_layer, obs, x_spacing=0.42, y_spacing_horizontal=0.42, y_spacing_vertical=0.12):
    from collections import defaultdict

    # Step 1: Build child → parents and parent → children maps
    parents = defaultdict(set)
    children = defaultdict(set)

    for lit in obs:
        if hasattr(lit, 'predicate') and lit.predicate.name == 'on':
            above = lit.variables[0].name
            below = lit.variables[1].name
            parents[above].add(below)
            children[below].add(above)

    # Step 2: Position base layer
    placed = {}
    layer_y = 0.5
    offset = 3.2/2 - len(base_layer) * (x_spacing-0.1) / 2
    for i, name in enumerate(base_layer):
        placed[name] = {'name': name, 'x': offset + i * x_spacing, 'y': layer_y}

    # Step 3: Recursively build layers above
    current_layer = set(base_layer)
    visited = set(base_layer)

    horizontally = True
    while True:
        next_layer = set()
        layer_y += y_spacing_horizontal if horizontally else y_spacing_vertical
        horizontally = not horizontally

        for parent, below_set in parents.items():
            if parent in visited:
                continue
            if below_set.issubset(current_layer):
                # All supports have been placed
                xs = [placed[child]['x'] for child in below_set if child in placed]
                if not xs:
                    continue
                avg_x = sum(xs) / len(xs)
                placed[parent] = {'name': parent, 'x': avg_x, 'y': layer_y}
                next_layer.add(parent)
                visited.add(parent)

        if not next_layer:
            break
        current_layer = next_layer

    return list(placed.values())






import re
from collections import defaultdict, deque

def parse_predicates(predicates):
    left_edges = []
    on_table = set()

    for pred in predicates:
        pred= str(pred)
        if pred.startswith("left("):
            match = re.match(r"left\((\w+):domino,(\w+):domino\)", pred)
            if match:
                left_edges.append((match.group(1), match.group(2)))
        elif pred.startswith("ontable("):
            match = re.match(r"ontable\((\w+):domino\)", pred)
            if match:
                on_table.add(match.group(1))
    
    return left_edges, on_table

def sort_ontable_dominoes(predicates):
    left_edges, on_table = parse_predicates(predicates)

    # Build graph from left predicates
    graph = defaultdict(list)
    in_degree = defaultdict(int)

    for a, b in left_edges:
        graph[a].append(b)
        in_degree[b] += 1
        if a not in in_degree:
            in_degree[a] = 0

    # Topological sort on the on-table dominoes only
    queue = deque([node for node in on_table if in_degree[node] == 0])
    result = []

    while queue:
        node = queue.popleft()
        if node in on_table:
            result.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return result




import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_domino_positions(domino_positions, filename="test.png", width=1.0, height=0.4):
    fig, ax = plt.subplots(figsize=(10, 6))

    for domino in domino_positions:
        x = domino['x']
        y = domino['y']
        name = domino['name']

        # Draw rectangle for the domino
        rect = patches.Rectangle((x, y), width, height, linewidth=1.5,
                                 edgecolor='black', facecolor='lightblue')
        ax.add_patch(rect)

        # Add label
        ax.text(x + width/2, y + height/2, name,
                ha='center', va='center', fontsize=12, weight='bold')

    # Adjust plot limits
    all_x = [d['x'] for d in domino_positions]
    all_y = [d['y'] for d in domino_positions]
    ax.set_xlim(min(all_x) - 1, max(all_x) + 2)
    ax.set_ylim(min(all_y) - 1, max(all_y) + 2)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved to {filename}")


_domino_name_to_x_pos = {}
def get_objects_from_obs(obs):
    global _domino_name_to_x_pos
    ontable = sort_ontable_dominoes(obs)
    dominoes = layout_scene_from_base_layer(ontable, obs)
    #plot_domino_positions(dominoes, filename="test.png", width=0.4, height=0.2)

    # Handle shift in x-axis
    if len(_domino_name_to_x_pos.items()) > 0:
        shift = 0

        # Find difference
        for dom in dominoes:
            if dom['name'] in _domino_name_to_x_pos:
                x, y = _domino_name_to_x_pos[dom['name']]
                if dom['y'] != y:
                    continue
                if dom['x'] != x:
                    shift = x - dom['x']
                    break
        
        # Shift all blocks
        for dom in dominoes:
            dom['x'] += shift

    # Reset
    _domino_name_to_x_pos = {}
    for dom in dominoes:
        _domino_name_to_x_pos[dom['name']] = (dom['x'], dom['y']) # x position for tracking, y for making sure it didn't change

    

    # add orientation information
    for i, ele in enumerate(dominoes):
        ele['orientation'] = 'vertical' if f"vertically({dominoes[i]['name']}:domino" in str(obs) else 'horizontal'


    # Add typed entities
    entities = set()
    for literal in list(obs):
        for arg in literal.variables:
            entities.add(arg)
    for i, ele in enumerate(dominoes):
        for entity in entities:
            if str(entity).startswith(ele['name']):
                ele['typed_entity'] = entity
                break

    # Get the robot's holding domino
    holding = None
    for literal in obs:
        if literal.predicate.name == "holding":
            holding = literal.variables[0]
            break
    from pddlgym.structs import TypedEntity, Literal, Predicate
    if holding is not None:
        vertically = Literal(Predicate("vertically", 1), [holding]) in obs
        x = 1.0
        y = 1.0
        holding = {'name': holding.name, 'x': x, "y": y,'orientation': 'vertical' if vertically else 'horizontal', 'typed_entity': holding}

    return dominoes, holding


def get_params(width, height, table_height, robot_height, num_dominoes):
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

    # Use the object classes from the config
    horizontal_padding = 0.025 * width
    block_width = 0.1#width / num_dominoes - 2*horizontal_padding
    block_height = 0.4#(height - table_height - robot_height) / num_dominoes - 0.05 * height

    return block_width, block_height

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
    holding_name = holding['typed_entity'].name if holding is not None else None
    holding, holding_dict = holding_name, holding
    if holding is not None:
        holding_color = block_name_to_color(holding, train_set=train_set)
        ec = (0.2, 0.2, 0.2)
        if holding_dict['orientation'] == 'vertical':
            holding_x = midx - block_width / 2
            holding_y = y - robot_height/2 - block_height / 1.5  # Positioned above the robot
        else:
            holding_x = midx - block_height / 2
            holding_y = y - robot_height/2 - block_width / 2


        x1, y1 = data_to_pixel(holding_x, holding_y)
        if holding_dict['orientation'] == 'vertical':
            x2, y2 = data_to_pixel(holding_x + block_width, holding_y + block_height)
        else:
            x2, y2 = data_to_pixel(holding_x + block_height, holding_y + block_width)
        y1, y2 = y2, y1
        block_data = {
            'name': holding[:1],
            'object_class': 'domino',
            'color': holding_color,
            'bbox_pixel': (x1, y1, x2, y2)
        }

        # Draw the held block
        if holding_dict['orientation'] == 'vertical':
            rect = patches.Rectangle((holding_x, holding_y), block_width, block_height, 
                                 linewidth=1, edgecolor=ec, facecolor=holding_color)
        else:
            rect = patches.Rectangle((holding_x, holding_y), block_height, block_width, 
                                 linewidth=1, edgecolor=ec, facecolor=holding_color)
        ax.add_patch(rect)

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

def draw_dominoes(ax, block_width, block_height, block_positions, add_label=False, jitter_pos_xy_vec=[0.0, 0.0], jitter_size_xy_vec=[0.0, 0.0], train_set=True):
    block_data = []
    for ele in block_positions:
        block_name = ele['name']
        orientation = ele['orientation']
        x = ele['x']
        y = ele['y']
        color = block_name_to_color(block_name, train_set=train_set)

        done = False
        while not done:
            
            # add noise to block position
            x += np.random.uniform(-jitter_pos_xy_vec[0], jitter_pos_xy_vec[0])
            y += np.random.uniform(-jitter_pos_xy_vec[1], jitter_pos_xy_vec[1])

            # account for rotation
            if orientation == 'horizontal':
                x -= block_height / 2 - block_width / 2

            # add noise to block size
            width_add = np.random.uniform(-jitter_size_xy_vec[0], jitter_size_xy_vec[0])
            height_add = np.random.uniform(-jitter_size_xy_vec[1], jitter_size_xy_vec[1])
            block_width = max(block_width/2, block_width + width_add)
            block_height = max(block_height/2, block_height + height_add)

            # Collect block data
            x1, y1 = data_to_pixel(x, y)
            if orientation == 'vertical':
                x2, y2 = data_to_pixel(x + block_width, y + block_height)
            else:
                x2, y2 = data_to_pixel(x + block_height, y + block_width)


            y1, y2 = y2, y1

            # check if block is outside of the scene
            if x1 >= x2 or y1 >= y2:
                continue

            # Add block data to the scene
            block_data.append({
                'name': block_name[:1],
                'object_class': 'domino',
                'color': color,
                'bbox_pixel': (x1, y1, x2, y2)
            })

            if x1 < 0 or x2 > 480.0 or y1 < 0 or y2 > 480.0:
                raise RuntimeError("Domino is outside of the scene.")
            if x1 >= x2 or y1 >= y2:
                raise RuntimeError("Domino's x and y coordinates are not valid.")

            # Draw block rectangle
            if orientation == 'vertical':
                rect = patches.Rectangle((x, y), block_width, block_height, 
                                        linewidth=1, edgecolor=(0.2, 0.2, 0.2), facecolor=color)
            else:
                rect = patches.Rectangle((x, y), block_height, block_width, 
                                    linewidth=1, edgecolor=(0.2, 0.2, 0.2), facecolor=color)
            ax.add_patch(rect)

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


def render(obs, add_object_labels=False, env=None):
    import matplotlib
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
        num_dominoes = env.num_dominoes
        RENDER_REALISTICALLY = env.render_realistically
    except:
        jitter_pos_xy_vec = [0.0, 0.0]
        jitter_size_xy_vec = [0.0, 0.0]
        train_set = True
    

    # Get coordinates of the objects
    forest_coords, holding = get_objects_from_obs(obs)

    

    block_width, block_height = get_params(width, height, table_height, robot_height, num_dominoes)
    if holding is None or holding['orientation'] == "vertical":
        robot_width = block_width * 2.0
    else:
        robot_width = block_height * 1.6
    robot_midx = width / 2
    robot_midy = height - robot_height/2



    #synthesize scene data 
    obs_list = list(obs)
    scene_data = draw_dominoes(ax, block_width, block_height, forest_coords, add_object_labels, jitter_pos_xy_vec=jitter_pos_xy_vec, jitter_size_xy_vec=jitter_size_xy_vec, train_set=train_set) # get block information


    # add relationships information
    for i, ele in enumerate(list(forest_coords)):
        typed_entity = ele['typed_entity']
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
    return data, scene_data