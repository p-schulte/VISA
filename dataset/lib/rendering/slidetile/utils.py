from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import RegularPolygon
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import os


from skimage.transform import resize

def normalize_token_image(im, target_size=480/3):
    """
    Resize any token image to a fixed square size.
    Args:
        im (ndarray): H×W×3 or H×W×4 image array
        target_size (int): output size in pixels (height=width)
    Returns:
        ndarray: resized image, dtype same as input
    """
    im_resized = resize(im, (target_size, target_size, im.shape[-1]),
                        preserve_range=True, anti_aliasing=True)
    return im_resized.astype(im.dtype)



IM_SCALE = 1.0

def get_asset_path(asset_name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    asset_dir_path = os.path.join(dir_path, 'assets')
    return os.path.join(asset_dir_path, asset_name)

def fig2data(fig, dpi: int = 150):
    fig.set_dpi(dpi)
    fig.canvas.draw()

    # copy image data from buffer
    data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).copy()

    # get the dpi adjusted figure dimensions
    width, height = map(int, fig.get_size_inches() * fig.get_dpi())
    data = data.reshape(height, width, 4)
    
    data[..., [0, 1, 2, 3]] = data[..., [1, 2, 3, 0]]

    return data

def initialize_figure(height, width, fig_scale=1., grid_colors=None):
    # Fix figure size so 3.2 inches × 150 dpi = 480 px
    fig = plt.figure(figsize=(3.2, 3.2))


    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0),
                                aspect='equal', frameon=False,
                                xlim=(-0.05, width + 0.05),
                                ylim=(-0.05, height + 0.05))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plt.NullFormatter())
        axis.set_major_locator(plt.NullLocator())

    # Draw a grid in the background
    for r in range(height):
        for c in range(width):
            edge_color = '#888888'
            if grid_colors is not None:
                face_color = grid_colors[r, c]
            else:
                face_color = 'white'
            
            drawing = RegularPolygon((c + 0.5, (height - 1 - r) + 0.5),
                                         numVertices=4,
                                         radius=0.5 * np.sqrt(2),
                                         orientation=np.pi / 4,
                                         ec=edge_color,
                                         fc=face_color)
            ax.add_patch(drawing)

    return fig, ax

def render_from_layout(layout, get_token_images, dpi=150, grid_colors=None):
    height, width = layout.shape[:2]

    fig, ax = initialize_figure(height, width, grid_colors=grid_colors)

    for r in range(height):
        for c in range(width):
            token_images = get_token_images(layout[r, c])
            for im in token_images:
                im = normalize_token_image(im, target_size=64)   # <--- normalize here
                draw_token(im, r, c, ax, height, width)


    im = fig2data(fig, dpi=dpi)
    plt.close(fig)

    # No downscaling – directly return 480×480
    im = np.array(im)
    return im


def draw_token(token_image, r, c, ax, height, width, token_scale=1.0, fig_scale=1.0):
    oi = OffsetImage(token_image, zoom = fig_scale * (token_scale / max(height, width)**0.5))
    box = AnnotationBbox(oi, (c + 0.5, (height - 1 - r) + 0.5), frameon=False)
    ax.add_artist(box)
    return box


################################################################
# no margins

from skimage.transform import resize

def render_from_layout_crisp(layout, get_token_images, tilesize=16):
    height, width = layout.shape[:2]
    canvas = np.zeros((height*tilesize,width*tilesize,3))

    for r in range(height):
        for c in range(width):
            token_images = get_token_images(layout[r, c])
            for im in token_images:
                canvas[r*tilesize:(r+1)*tilesize, c*tilesize:(c+1)*tilesize] = \
                    resize(im[:,:,:3], (tilesize,tilesize,3), preserve_range=True)

    return canvas

























# Own part:
import re

def _var_name(v):
    # PDDLGym objects have .name; fall back to str for safety
    return getattr(v, "name", str(v))

def _idx(pos_name):
    # "x1" -> 1, "y3" -> 3
    m = re.search(r"(\d+)$", pos_name)
    return int(m.group(1)) if m else None

def _cell_bbox(ix, iy, n, img=480, margin=0.2):
    """
    Bounding Box für ein Tile im n×n Slidetile Grid.

    ix, iy: 1-basierte Zellindizes
    img: Bildgröße in Pixeln
    margin: Anteil des Zellraums, der leer bleibt (0.2 => Tile ist 80% der Zelle)
    """
    # exakte Zellgröße
    cell = img / n
    margin = max(0.0, min(margin, 0.45))
    tile = cell * (1 - margin)
    offset = (cell - tile) / 2

    # global_offset sorgt für symmetrische Einbettung des Boards
    global_offset = (img - cell * n) / 2  # bei 480/3=160 ist das 0, aber generisch korrekt

    # Koordinaten: beide Achsen gleich behandeln!
    x1 = global_offset + (ix - 1) * cell + offset
    y1 = global_offset + (n - iy) * cell + offset
    x2 = x1 + tile
    y2 = y1 + tile

    offset = 5
    match ix:
        case 1:
            x1 += offset
            x2 += offset
        case 3:
            x1 -= offset
            x2 -= offset

    match iy:
        case 1:
            y1 -= offset
            y2 -= offset
        case 3:
            y1 += offset
            y2 += offset

    if True: # Invert y-axis
        y1, y2 = img - y2, img - y1

    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))







def build_scene_data_slidetile_from_obs(obs, img=480, pad=6, add_board=True):
    """
    obs: frozenset of PDDLGym literals (slidetile)
    returns: scene_data (list[dict])
    """
    tile_pos = {}     # name -> (ix,iy)
    tile_mat = [[None for _ in range(4)] for _ in range(4)]    # (ix, iy) -> name
    blank_pos = None  # (ix,iy)
    xs, ys = set(), set()
    ADD_POSITION_MARKER = True

    # 1) Parse the literals we care about
    for lit in obs:
        pred = getattr(lit.predicate, "name", str(lit.predicate))
        vars_ = getattr(lit, "variables", ())
        if pred == "at":  # at(tile, x, y)
            t, x, y = map(_var_name, vars_)
            ix, iy = _idx(x), _idx(y)
            tile_pos[t] = (ix, iy)
            tile_mat[ix-1][iy-1] = t
            xs.add(ix); ys.add(iy)
        elif pred == "blank":  # blank(x, y)
            x, y = map(_var_name, vars_)
            ix, iy = _idx(x), _idx(y)
            blank_pos = (ix, iy)
            tile_mat[ix-1][iy-1] = "blank"
            xs.add(ix); ys.add(iy)

    # 2) Infer grid size N
    if not xs or not ys:
        raise ValueError("Could not infer grid size from obs.")
    n = max(max(xs), max(ys))

    # 3) Build scene_data entries
    scene_data = []

    # tiles
    for t, (ix, iy) in sorted(tile_pos.items()):        
        relationships = {}
        if ix < n:
            relationships["adj-right"] = [tile_mat[ix][iy-1]]
        if ix > 1:
            relationships["adj-left"] = [tile_mat[ix-2][iy-1]]
        if iy < n:
            relationships["adj-below"] = [tile_mat[ix-1][iy]]
        if iy > 1:
            relationships["adj-above"] = [tile_mat[ix-1][iy-2]]


        scene_data.append({
            "name": t,
            "object_class": "tile",
            "color": (90, 140, 220),                # any RGB tuple is fine
            "bbox_pixel": _cell_bbox(ix, iy, n, img, pad),
            "unary_relationships": [],
            # keep coordinates as strings to mirror your blocks pipeline style
            "binary_relationships": { "at": [f"x{ix}_y{iy}"] } if ADD_POSITION_MARKER else relationships,  #{ "at": [f"x{ix}_y{iy}"] }
        })

    # blank
    if blank_pos is not None:
        ix, iy = blank_pos        
        relationships = {}
        if ix < n:
            relationships["adj-right"] = [tile_mat[ix][iy-1]]
        if ix > 1:
            relationships["adj-left"] = [tile_mat[ix-2][iy-1]]
        if iy < n:
            relationships["adj-below"] = [tile_mat[ix-1][iy]]
        if iy > 1:
            relationships["adj-above"] = [tile_mat[ix-1][iy-2]]


        scene_data.append({
            "name": "blank",
            "object_class": "blank",
            "color": (245, 245, 245),
            "bbox_pixel": _cell_bbox(ix, iy, n, img, pad),
            "unary_relationships": [],
            "binary_relationships": { "at": [f"x{ix}_y{iy}"] } if ADD_POSITION_MARKER else relationships,  #{ "at": [f"x{ix}_y{iy}"] }
        })

    
    # 4) alle Positionen als Objekte hinzufügen
    if ADD_POSITION_MARKER:
        for ix in range(1, n+1):
            for iy in range(1, n+1):
                new_box = _cell_bbox(ix, iy, n, img, margin=0.0)
                y_start = new_box[3]
                new_box = (new_box[0]+15, max(3, y_start-25) , new_box[2]-15, min(477,y_start))
                relationships = {}
                if ix < n:
                    relationships["adj-right"] = [f"x{ix+1}_y{iy}"]
                if ix > 1:
                    relationships["adj-left"] = [f"x{ix-1}_y{iy}"]
                if iy < n:
                    relationships["adj-below"] = [f"x{ix}_y{iy+1}"]
                if iy > 1:
                    relationships["adj-above"] = [f"x{ix}_y{iy-1}"]

                scene_data.append({
                    "name": f"x{ix}_y{iy}",
                    "object_class": "position",
                    "color": (200, 200, 200),  # optional
                    "bbox_pixel": new_box,  # ganze Zelle
                    "unary_relationships": [],
                    "binary_relationships": relationships,
                })

    return scene_data
