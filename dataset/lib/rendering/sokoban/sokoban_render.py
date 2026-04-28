from .utils import get_asset_path, render_from_layout, render_from_layout_crisp

import matplotlib.pyplot as plt
import numpy as np

NUM_OBJECTS = 6
CLEAR, PLAYER, STONE, STONE_AT_GOAL, GOAL, WALL = range(NUM_OBJECTS)

TOKEN_IMAGES = {
    PLAYER : plt.imread(get_asset_path('sokoban_player.png')),
    STONE : plt.imread(get_asset_path('sokoban_stone.png')),
    STONE_AT_GOAL : plt.imread(get_asset_path('sokoban_stone_at_goal.png')),
    GOAL : plt.imread(get_asset_path('sokoban_goal.png')),
    WALL : plt.imread(get_asset_path('sokoban_wall.png')),
    CLEAR : plt.imread(get_asset_path('sokoban_clear.png')),
}

def loc_str_to_loc(loc_str):
    _, r, c = loc_str.split('-')
    return (int(r), int(c))

def get_locations(obs, thing):
    locs = []
    for lit in obs:
        if lit.predicate.name != 'at':
            continue
        if thing in lit.variables[0]:
            locs.append(loc_str_to_loc(lit.variables[1]))
    return locs

def get_values(obs, name):
    values = []
    for lit in obs:
        if lit.predicate.name == name:
            values.append(lit.variables)
    return values

def build_layout(obs):
    # Get location boundaries
    max_r, max_c = -np.inf, -np.inf
    for lit in obs:
        for v in lit.variables:
            if v.startswith('pos-'):
                r, c = loc_str_to_loc(v)
                max_r = max(max_r, r)
                max_c = max(max_c, c)
    layout = CLEAR * np.ones((max_r+1, max_c+1), dtype=np.uint8)

    # Put things in the layout
    # Also track seen locs and goal locs
    seen_locs = set()
    goal_locs = set()

    for v in get_values(obs, 'is-goal'):
        r, c = loc_str_to_loc(v[0])
        layout[r, c] = GOAL
        seen_locs.add((r, c))
        goal_locs.add((r, c))

    for r, c in get_locations(obs, 'stone'):
        if (r, c) in goal_locs:
            layout[r, c] = STONE_AT_GOAL
        else:
            layout[r, c] = STONE
        seen_locs.add((r, c))

    for r, c in get_locations(obs, 'player'):
        layout[r, c] = PLAYER
        seen_locs.add((r, c))

    for v in get_values(obs, 'clear'):
        r, c = loc_str_to_loc(v[0])
        if (r, c) in goal_locs:
            continue
        layout[r, c] = CLEAR
        seen_locs.add((r, c))

    # Add walls
    for v in get_values(obs, 'is-nongoal'):
        r, c = loc_str_to_loc(v[0])
        if (r, c) in seen_locs:
            continue
        layout[r, c] = WALL

    # 1 indexing
    layout = layout[1:, 1:]

    # r-c flip
    layout = np.transpose(layout)

    # print("layout:")
    # print(layout)
    # import ipdb; ipdb.set_trace()
    return layout

def build_layout_egocentric(obs,size=5):
    if (size % 2) == 0:
        size += 1
    width = (size-1)//2

    layout = CLEAR * np.ones((size, size), dtype=np.uint8)

    # Put things in the layout
    # Also track seen locs and goal locs
    seen_locs = set()
    goal_locs = set()

    for r, c in get_locations(obs, 'player'):
        player_r, player_c = r, c
        offset_r, offset_c = r - width, c - width

    def within_view(r,c):
        return (abs(r - player_r) <= width) and (abs(c - player_c) <= width)

    for v in get_values(obs, 'is-goal'):
        r, c = loc_str_to_loc(v[0])
        if within_view(r,c):
            layout[r-offset_r, c-offset_c] = GOAL
            seen_locs.add((r, c))
            goal_locs.add((r, c))

    for r, c in get_locations(obs, 'stone'):
        if within_view(r,c):
            if (r, c) in goal_locs:
                layout[r-offset_r, c-offset_c] = STONE_AT_GOAL
            else:
                layout[r-offset_r, c-offset_c] = STONE
            seen_locs.add((r, c))

    for r, c in get_locations(obs, 'player'):
        layout[width, width] = PLAYER
        seen_locs.add((r, c))

    for v in get_values(obs, 'clear'):
        r, c = loc_str_to_loc(v[0])
        if within_view(r,c):
            if (r, c) in goal_locs:
                continue
            layout[r-offset_r, c-offset_c] = CLEAR
            seen_locs.add((r, c))

    # Add walls
    for v in get_values(obs, 'is-nongoal'):
        r, c = loc_str_to_loc(v[0])
        if within_view(r,c):
            if (r, c) in seen_locs:
                continue
            layout[r-offset_r, c-offset_c] = WALL

    # r-c flip
    layout = np.transpose(layout)

    # print("layout:")
    # print(layout)
    # import ipdb; ipdb.set_trace()
    return layout

def get_token_images(obs_cell):
    return [TOKEN_IMAGES[obs_cell]]


def render(obs, mode='human', close=False, add_object_labels=False, env=None, draw_bboxes=False):
    

    if mode == "human":
        layout = build_layout(obs)
        #import pdb;pdb.set_trace()

        # resize to 480x480
        import cv2
        TARGET_SIZE = (480, 480)
        def resize_image(img):
            return cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        
        out_img_raw = render_from_layout(layout, get_token_images)
        out_img = resize_image(out_img_raw)


        # synthesize scene_data
        scene_data = layout_to_scene_data(layout, image_size=480)

        # term = False
        # import pdb;pdb.set_trace()
        # if term:
        #     import sys; sys.exit(0)



        return out_img, scene_data
    


    # other cases are irrelevant for sokoban
    '''
    elif mode == "egocentric":
        layout = build_layout_egocentric(obs)
        return render_from_layout(layout, get_token_images), []
    elif mode == "egocentric_crisp":
        layout = build_layout_egocentric(obs)
        return render_from_layout_crisp(layout, get_token_images), []
    elif mode == "human_crisp":
        layout = build_layout(obs)
        return render_from_layout_crisp(layout, get_token_images), []
    elif mode == "layout":
        return build_layout(obs)
    elif mode == "egocentric_layout":
        return build_layout_egocentric(obs), []
    '''








import numpy as np

def layout_to_scene_data(layout, image_size=480):
    """
    layout: np.ndarray of shape (n, n), values:
        0 = clear
        1 = player
        2 = box
        3 = box+goal (STONE_AT_GOAL, optional)
        4 = goal
        5 = wall
    image_size: full image is image_size x image_size (default 480).
    """
    layout = np.asarray(layout)
    n_rows, n_cols = layout.shape
    assert n_rows == n_cols, "layout must be square (n x n)"
    n = n_rows

    tile_size = image_size / n  # use float to avoid accumulating rounding errors

    # First, create a name grid with canonical ordering p1, p2, ...
    name_grid = np.empty_like(layout, dtype=object)
    idx = 1
    for r in range(n):
        for c in range(n):
            name_grid[r, c] = f"p{idx}"
            idx += 1

    scene_data = []


    VERSION_TO_CHOOSE = "NEW" # OLD or NEW

    if VERSION_TO_CHOOSE == "OLD":
        for r in range(n):
            for c in range(n):
                name = name_grid[r, c]
                val = int(layout[r, c])
                # Compute bbox: (x_min, y_min, x_max, y_max)
                x_min = int(round(c * tile_size))
                y_min = int(round(r * tile_size))
                x_max = int(round((c + 1) * tile_size))
                y_max = int(round((r + 1) * tile_size))
                bbox_pixel = (x_min+3, y_min+3, x_max-3, y_max-3)

                # Unary relationships
                unary = []

                if val == 5:
                    unary.append("is_wall")
                else:
                    unary.append("is_clear")

                # box / goal / player
                if val in (2, 3):  # STONE or STONE_AT_GOAL
                    unary.append("has_box")
                if val in (4, 3):  # GOAL or STONE_AT_GOAL
                    unary.append("has_goal")
                if val == 1:       # PLAYER
                    unary.append("has_player")

                # Binary relationships (neighbors)
                binary = {}

                # left neighbor
                if c > 0:
                    binary.setdefault("adj-left", []).append(name_grid[r, c - 1])
                # right neighbor
                if c < n - 1:
                    binary.setdefault("adj-right", []).append(name_grid[r, c + 1])
                # above neighbor
                if r > 0:
                    binary.setdefault("adj-above", []).append(name_grid[r - 1, c])
                # below neighbor
                if r < n - 1:
                    binary.setdefault("adj-below", []).append(name_grid[r + 1, c])

                obj = {
                    "name": name,
                    "object_class": "position",
                    "color": (90, 140, 220),  # same as your example
                    "bbox_pixel": bbox_pixel,
                    "unary_relationships": unary,
                    "binary_relationships": binary,
                }
                scene_data.append(obj)

    
    elif VERSION_TO_CHOOSE == "NEW": # TODO: adapt this one 
        for r in range(n):
            for c in range(n):
                name = name_grid[r, c]
                val = int(layout[r, c])
                # Compute bbox: (x_min, y_min, x_max, y_max)
                x_min = int(round(c * tile_size))
                y_min = int(round(r * tile_size))
                x_max = int(round((c + 1) * tile_size))
                y_max = int(round((r + 1) * tile_size))
                bbox_pixel = (x_min+3, y_min+3, x_max-3, y_max-3)

                # Unary relationships
                unary = []

                # if val == 5:
                #     unary.append("is_wall")
                # else:
                #     unary.append("is_clear")

                # box / goal / player
                if val in (2, 3):  # STONE or STONE_AT_GOAL
                    unary.append("has_box")
                # if val in (4, 3):  # GOAL or STONE_AT_GOAL
                #     unary.append("has_goal")
                if val == 1:       # PLAYER
                    unary.append("has_player")

                # Binary relationships (neighbors)
                binary = {}

                # left neighbor
                if c > 0:
                    binary.setdefault("adj", []).append(name_grid[r, c - 1])
                # right neighbor
                if c < n - 1:
                    binary.setdefault("adj", []).append(name_grid[r, c + 1])
                # above neighbor
                if r > 0:
                    binary.setdefault("adj", []).append(name_grid[r - 1, c])
                # below neighbor
                if r < n - 1:
                    binary.setdefault("adj", []).append(name_grid[r + 1, c])

                # left neighbor 2
                if c > 1:
                    binary.setdefault("adj_2", []).append(name_grid[r, c - 2])
                # right neighbor 2
                if c < n - 2:
                    binary.setdefault("adj_2", []).append(name_grid[r, c + 2])
                # above neighbor 2
                if r > 1:
                    binary.setdefault("adj_2", []).append(name_grid[r - 2, c])
                # below neighbor 2
                if r < n - 2:
                    binary.setdefault("adj_2", []).append(name_grid[r + 2, c])

                obj = {
                    "name": name,
                    "object_class": "position",
                    "color": (90, 140, 220),  # same as your example
                    "bbox_pixel": bbox_pixel,
                    "unary_relationships": unary,
                    "binary_relationships": binary,
                }
                scene_data.append(obj)

    return scene_data
