import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import numpy as np


from PIL import Image
import os

ASSETS_DIR = "lib/rendering/logistics/assets"
ASSET_MAP = {
    "airplane": "plane.png",  # put your airplane file name here
    "truck": "bus.png",          # use the bus sprite for trucks
}
_SPRITE_CACHE = {}

def _load_sprite(object_class):
    """Load and cache RGBA sprites by logical class name."""
    if object_class in _SPRITE_CACHE:
        return _SPRITE_CACHE[object_class]

    fname = ASSET_MAP.get(object_class)
    if not fname:
        _SPRITE_CACHE[object_class] = None
        return None

    path = os.path.join(ASSETS_DIR, fname)
    if not os.path.exists(path):
        _SPRITE_CACHE[object_class] = None
        return None

    img = Image.open(path).convert("RGBA")
    _SPRITE_CACHE[object_class] = img
    return img

def _draw_sprite(ax, sprite_img, x, y, w, h, z=10):
    """
    Draw a PIL RGBA sprite into Matplotlib coordinates using extent.
    (x,y) is lower-left corner in data coords, w/h are data sizes.
    """
    ax.imshow(sprite_img, extent=(x, x + w, y, y + h), zorder=z, interpolation="nearest")




# === same helper import as gripper_render ===
from lib.rendering.utils import fig2data

# If your utils.py is in lib/rendering/gripper/utils.py (as in gripper_render.py), keep this:
from lib.rendering.gripper.utils import add_relationships
# If you actually placed it in lib/rendering/logistics/utils.py, change the line above accordingly.

# ---------------------------------------------------------------------
# Coordinate helpers (kept identical so 3.2in × 150 dpi => 480×480 px)
# ---------------------------------------------------------------------
def data_to_pixel(x, y, width=3.2, height=3.2, dpi=150):
    x_min, x_max = -0.05, width + 0.05
    y_min, y_max = -0.05, height + 0.05
    img_w = int(width * dpi)
    img_h = int(height * dpi)

    px_x = int(((x - x_min) / (x_max - x_min)) * img_w)
    px_y = int(((y - y_min) / (y_max - y_min)) * img_h)
    px_y = img_h - px_y
    return px_x, px_y

def _add_bbox(scene_data, name, cls, x, y, w, h):
    """x,y is lower-left in data coords; w,h are data-size."""
    x1, y1 = data_to_pixel(x, y)
    x2, y2 = data_to_pixel(x + w, y + h)
    # swap y so bbox is (xmin,ymin,xmax,ymax) in pixel space
    y1, y2 = y2, y1
    scene_data.append({
        "name": name,
        "object_class": cls,
        "bbox_pixel": (x1, y1, x2, y2)
    })

# ---------------------------------------------------------------------
# Simple sprite drawers (rectangles for now)
# ---------------------------------------------------------------------
def draw_truck(ax, name, x, y, scene_data, color=None):
    # logical size kept consistent with the previous placeholder
    w, h = 0.9, 0.275
    sprite = _load_sprite("truck")
    if sprite is not None:
        _draw_sprite(ax, sprite, x, y, w, h, z=6)
    else:
        # fallback: simple rectangle if asset missing
        ax.add_patch(patches.Rectangle((x, y), w, h, facecolor="limegreen", edgecolor="black", lw=1.5, zorder=5))
    _add_bbox(scene_data, name, "truck", x, y, w, h)

def draw_plane(ax, name, x, y, scene_data, color=None):
    w, h = 1.0, 0.25
    sprite = _load_sprite("airplane")
    if sprite is not None:
        _draw_sprite(ax, sprite, x, y, w, h, z=6)
    else:
        ax.add_patch(patches.Rectangle((x, y), w, h, facecolor="gold", edgecolor="black", lw=1.5, zorder=5))
    _add_bbox(scene_data, name, "airplane", x, y+0.075, w, h-0.075)


def draw_package(ax, name, x, y, scene_data):
    w, h = 0.28, 0.28
    ax.add_patch(patches.Rectangle((x, y), w, h, facecolor="white", edgecolor="black", lw=1.2))
    _add_bbox(scene_data, name, "package", x, y, w, h)
    # draw ID number (last char) centered
    cx, cy = x + w/2, y + h/2
    ax.text(cx, cy, name[-1], ha="center", va="center", fontsize=10)

def overlay_bboxes(image_array, scene_data, color_map=None, thickness=3, draw_labels=True):
    if color_map is None:
        color_map = {"package": (0,120,255), "truck": (200,0,0), "airplane": (255,165,0), "cell": (0,200,0)}
    im = Image.fromarray(image_array.copy())
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for obj in scene_data:
        if "bbox_pixel" not in obj:
            continue
        x1, y1, x2, y2 = obj["bbox_pixel"]
        cls = obj.get("object_class")
        ident = obj.get("name", "")
        color = color_map.get(cls, (255, 0, 0))
        for i in range(thickness):
            draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)
        if draw_labels:
            label = f"{ident}:{cls}" if ident else cls
            tw, th = draw.textsize(label, font=font)
            draw.rectangle([x1, y1 - th, x1 + tw + 4, y1], fill=color)
            draw.text((x1 + 2, y1 - th), label, fill=(255, 255, 255), font=font)
    return np.array(im)

# ---------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------



# --- keep your imports as before ---

# unchanged: data_to_pixel(), _add_bbox(), draw_truck(), draw_plane(), draw_package(), overlay_bboxes()

def render(obs, add_object_labels=False, env=None, draw_bboxes=False):
    """
    480×480 renderer for the logistics domain.
    - Unloaded packages are stacked on the RIGHT side of each tile.
    - Loaded packages are placed on the vehicle row.
    - Tile labels moved slightly upward.
    """
    # ---- Parse literals ----
    at_map, in_map, place_type = {}, {}, {}
    for lit in obs:
        pred = lit.predicate.name.lower()
        vars_ = [v.name for v in lit.variables]
        if pred == "at" and len(vars_) == 2:
            at_map[vars_[0]] = vars_[1]
        elif pred == "in" and len(vars_) == 2:
            in_map[vars_[0]] = vars_[1]
        # harvest types we might see (not strictly required for layout)
        for v in lit.variables:
            raw = str(v)
            if ":" in raw:
                name, typ = raw.split(":", 1)
                if typ in ("airport", "location"):
                    place_type[name] = typ

    # ---- Figure / axes ----
    width, height = 3.2, 3.2
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), aspect='equal', frameon=False,
                      xlim=(-0.05, width + 0.05), ylim=(-0.05, height + 0.05))
    ax.axis("off")

    scene_data = []

    # ---- Cell bounds + labels ----
    # tiles are 1.6 × 1.6; we keep these bounds to place stacks at the right edge
    cell_bounds = {
        "city1-1": (0.0, 1.6, 1.6, 1.6),
        "city1-2": (1.6, 1.6, 1.6, 1.6),
        "city2-1": (0.0, 0.0, 1.6, 1.6),
        "city2-2": (1.6, 0.0, 1.6, 1.6),
    }

    # grid lines
    ax.plot([1.6, 1.6], [0.0, 3.2], color="black", lw=1)
    ax.plot([0.0, 3.2], [1.6, 1.6], color="black", lw=1)

    def reg_cell(name, label):
        x, y, w, h = cell_bounds[name]
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, lw=1))
        # moved label UP slightly: was y + h - 0.20 with va='top'
        ax.text(x + 0.08, y + h - 0.10, label, fontsize=12, weight="bold", va="top")
        _add_bbox(scene_data, name, "cell", x, y+1.3, 1, .3)

    reg_cell("city1-1", "Loc 1-1")
    reg_cell("city1-2", "Loc 1-2")
    reg_cell("city2-1", "Loc 2-1")
    reg_cell("city2-2", "Loc 2-2")

    # ---- Placement anchors (vehicle rows) ----
    anchors = {
        "city1-1": {"vehicle_row": 2.05},
        "city1-2": {"vehicle_row": 2.05},
        "city2-1": {"vehicle_row": 0.45},
        "city2-2": {"vehicle_row": 0.45},
    }

    # ---- Entities by type ----
    trucks = [n for n in at_map if ":truck" in n or "truck" in n]
    planes = [n for n in at_map if ":airplane" in n or "plane" in n or "air" in n]
    packages = [n for n in set(list(at_map.keys()) + list(in_map.keys()))
                if "package" in n or ":obj" in n]

    # ---- Draw vehicles ----
    for t in trucks:
        cell = at_map.get(t)
        if cell not in anchors:
            continue
        y = anchors[cell]["vehicle_row"]
        y -= 0.15  # slight vertical nudge
        base_x = 0.05 if cell in ("city1-1", "city2-1") else 1.65
        draw_truck(ax, t, base_x, y - 0.25, scene_data, color="red" if "red" in t else "limegreen")

    for p in planes:
        cell = at_map.get(p)
        if cell not in anchors:
            continue
        y = anchors[cell]["vehicle_row"]
        y += 0.4  # slight vertical nudge
        base_x = 0.05 if cell in ("city1-1", "city2-1") else 1.65
        draw_plane(ax, p, base_x, y + 0.15, scene_data, color="gold" if "yellow" in p else "deepskyblue")

    # ---- Packages: loaded vs unloaded ----
    # loaded -> on vehicle row; unloaded -> RIGHT-SIDE VERTICAL STACK
    veh_cell = {**{t: at_map[t] for t in trucks}, **{p: at_map[p] for p in planes}}

    # per-cell offsets
    # loaded_offset = {c: 0 for c in cell_bounds}   # horizontal spread on row
    # stack_offset = {c: 0 for c in cell_bounds}    # vertical stack count

    pkg_w = 0.28
    pkg_h = 0.28
    loaded_dx = 0.32         # spacing for loaded packages on the row
    stack_dy = 0.32          # vertical spacing for stacks on the right


    # Build unique positioning 
    for pkg in in_map.keys(): # place packages in vehicles
        veh = in_map[pkg]
        if pkg in env.load_order[veh]:
            continue # already placed at the right position
        else:
            # remove from other places
            for v in env.load_order.keys():
                for i in range(len(env.load_order[v])):
                    if env.load_order[v][i] == pkg:
                        env.load_order[v][i] = None

            # place at the first available slot on new vehicle
            for i in range(len(env.load_order[veh])):
                if env.load_order[veh][i] is None:
                    env.load_order[veh][i] = pkg
                    break
    for pkg in at_map.keys(): # place packages at places
        if not pkg.startswith("package"): # skip non-package objects
            continue
        city = at_map[pkg]
        if pkg in env.load_order[city]:
            continue # already placed at the right position
        else:
            # remove from other places
            for v in env.load_order.keys():
                for i in range(len(env.load_order[v])):
                    if env.load_order[v][i] == pkg:
                        env.load_order[v][i] = None

            # place at the first available slot at the new city
            for i in range(len(env.load_order[city])):
                if env.load_order[city][i] is None:
                    env.load_order[city][i] = pkg
                    break
                

    for pkg in packages:
        if pkg in in_map:
            veh = in_map[pkg]
            cell = veh_cell.get(veh)
            if cell in anchors:
                x_offset = env.load_order[veh].index(pkg)
                x0 = (0.05 if cell in ("city1-1", "city2-1") else 1.65) + loaded_dx * x_offset#loaded_offset[cell]
                if veh.startswith("plane"):
                    y0 = anchors[cell]["vehicle_row"] + 0.3
                else:
                    y0 = anchors[cell]["vehicle_row"] - 0.075
                draw_package(ax, pkg, x0, y0, scene_data)
                #loaded_offset[cell] += 1
            continue

        # Unloaded: at(package, cell) -> stack at right edge
        cell = at_map.get(pkg)
        if cell in cell_bounds:
            cx, cy, cw, ch = cell_bounds[cell]
            # right edge minus a small margin and the package width
            right_x = cx + cw - 0.16 - pkg_w + 0.1
            # start near bottom of tile
            base_y = cy + 0.08
            y_offset = env.load_order[cell].index(pkg)
            y0 = base_y + stack_dy * y_offset# stack_offset[cell]
            draw_package(ax, pkg, right_x, y0, scene_data)
            #stack_offset[cell] += 1

    # ---- Rasterize ----
    img = fig2data(fig)
    plt.close(fig)

    if draw_bboxes:
        img = overlay_bboxes(img, scene_data, draw_labels=add_object_labels)

    # attach relationships as in gripper renderer
    add_relationships(scene_data, obs)
    return img, scene_data
