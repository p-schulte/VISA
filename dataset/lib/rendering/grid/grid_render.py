from .utils import render_grid_layout
from .layout_builder import build_layout

import matplotlib.pyplot as plt
import numpy as np





def render(obs, mode='human', close=False, add_object_labels=False, env=None, draw_bboxes=False):
    if mode != "human": import pdb;pdb.set_trace()  # only human mode supported



    # build layout
    layout = build_layout(obs)
    size_per_tile = 480 // layout["walls"].shape[0]

    # render image
    img = render_grid_layout(layout, tile_size=size_per_tile)

    # synthesize scene_data
    scene_data = layout_to_scene_data(layout, image_size=480, obs=obs)

    return img, scene_data





    return out_img, scene_data


    







import numpy as np

def layout_to_scene_data(layout, image_size=480, obs=None):
    

    # get tile size
    n = layout['player'].shape[0]
    tile_size = image_size / n


    # First, create a name grid with canonical ordering p1, p2, ...
    name_grid = np.empty_like(layout['keys'], dtype=object)
    idx = 1
    for r in range(n):
        for c in range(n):
            name_grid[r, c] = f"p{idx}"
            idx += 1

    # create emtpy list to hold scene data
    scene_data = []
    for r in range(n):
        for c in range(n):
            name = name_grid[r, c]


            # Compute bbox: (x_min, y_min, x_max, y_max)
            x_min = int(round(c * tile_size))
            y_min = int(round(r * tile_size))
            x_max = int(round((c + 1) * tile_size))
            y_max = int(round((r + 1) * tile_size))
            bbox_pixel = (x_min+3, y_min+3, x_max-3, y_max-3)


            # Unary relationships
            unary = []

            # wall
            if layout['walls'][r, c]:
                unary.append("is_wall")

            # floor
            if layout['floor'][r, c]:
                unary.append("is_floor")
            
            # keys
            val = layout['keys'][r, c]
            if val > 0:
                unary.append(f"has_key_type_{val}")

            # player/robot
            if layout['player'][r, c]:
                unary.append("has_robot")

                # hand-empty
                if 'empty' in str(obs):
                    unary.append("is_robot_hand_empty")

                # find argument # holding(key4:default)
                if 'holding' in str(obs):
                    for lit in obs:
                        if lit.predicate.name == "holding":
                            arg = lit.variables[0]
                            kname = arg.name.split('key')[-1]
                            unary.append(f"has_robot_holding_key_type_{int(kname)+1}")
            
            # doors
            val = layout['doors'][r, c]
            if val > 0:
                
                if layout['door_open'][r, c]:
                    unary.append(f"has_open_door_type_{val}")
                else:
                    unary.append(f"has_locked_door_type_{val}")

            

            # Binary relationships (neighbors)
            binary = {}

            # directions
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
                "color": (90, 140, 220),
                "bbox_pixel": bbox_pixel,
                "unary_relationships": unary,
                "binary_relationships": binary,
            }
            scene_data.append(obj)

    return scene_data
