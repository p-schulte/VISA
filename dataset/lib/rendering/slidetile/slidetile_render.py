from .utils import get_asset_path, render_from_layout

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def overlay_bboxes(image_array, scene_data, color_map=None, thickness=3, draw_labels=True):
    if color_map is None:
        color_map = {
            "tile": (0, 120, 255),   # blue
            "blank": (0, 200, 100),  # green
            "board": (180, 180, 180)
        }

    im = Image.fromarray(image_array.copy())
    draw = ImageDraw.Draw(im)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for obj in scene_data:
        if "bbox" in obj:
            x1, y1, x2, y2 = obj["bbox"]
            cls = obj.get("class") or obj.get("object_class")
            ident = obj.get("identifier", "")
        elif "bbox_pixel" in obj:
            x1, y1, x2, y2 = obj["bbox_pixel"]
            cls = obj.get("object_class")
            ident = obj.get("name", "")
        else:
            continue

        color = color_map.get(cls, (255, 0, 0))
        for i in range(thickness):
            draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline=color)

        if draw_labels:
            label = f"{ident}:{cls}" if ident else cls
            tw, th = draw.textsize(label, font=font)
            draw.rectangle([x1, y1 - th, x1 + tw + 4, y1], fill=color)
            draw.text((x1 + 2, y1 - th), label, fill=(255, 255, 255), font=font)

    return np.array(im)


# ==========================
# Slide-tile definitions
# ==========================

def generate_tile_token(tile_num):
    if tile_num is None:
        return plt.imread(get_asset_path("slidetile_empty.png"))
    return plt.imread(get_asset_path(f"slidetile_{tile_num}.png"))


def make_token_images(size=3):
    """Build TOKEN_IMAGES dict for a given size."""
    num_tiles = size * size - 1
    token_images = {0: generate_tile_token(None)}  # blank
    for t in range(1, num_tiles + 1):
        token_images[t] = generate_tile_token(t)
    return token_images


def build_layout(obs):
    # infer grid size from obs (xN, yN)
    xs, ys = set(), set()
    for lit in obs:
        if lit.predicate.name == "at":
            _, x, y = lit.variables
            xs.add(int(x[1:]))
            ys.add(int(y[1:]))
        elif lit.predicate.name == "blank":
            x, y = lit.variables
            xs.add(int(x[1:]))
            ys.add(int(y[1:]))

    n = max(max(xs), max(ys))
    layout = np.zeros((n, n), dtype=int)

    for lit in obs:
        if lit.predicate.name == "at":
            tile, x, y = lit.variables
            tile_num = int(tile[1:])
            c = int(x[1:]) - 1
            r = int(y[1:]) - 1
            layout[r, c] = tile_num
        elif lit.predicate.name == "blank":
            x, y = lit.variables
            c = int(x[1:]) - 1
            r = int(y[1:]) - 1
            layout[r, c] = 0  # blank

    return layout


def render(obs, mode="human", close=False, add_object_labels=False, env=None, draw_bboxes=False):
    layout = build_layout(obs)
    n = layout.shape[0]
    TOKEN_IMAGES = make_token_images(n)

    def get_token_images(obs_cell):
        return [TOKEN_IMAGES[obs_cell]]

    from .utils import build_scene_data_slidetile_from_obs
    scene_data = build_scene_data_slidetile_from_obs(obs)

    img = render_from_layout(layout, get_token_images)

    draw_bboxes = False
    if draw_bboxes:
        img = overlay_bboxes(img, scene_data)

    return img, scene_data
