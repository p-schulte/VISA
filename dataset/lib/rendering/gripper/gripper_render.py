import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from lib.rendering.utils import fig2data




def data_to_pixel(x, y, width=3.2, height=3.2, dpi=150):  # was 4.30,4.30
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




# === Drawing ===
def draw_single_gripper(ax, room, side="left", carried=None, scene_data=None):
    base_y = 2.2 if room == "rooma" else 0.6
    body_x = 0.2 if side == "left" else 1.0
    body_y = base_y + 0.7
    body_w, body_h = 0.4, 0.2

    # Body
    rect = patches.Rectangle((body_x, body_y), body_w, body_h,
                             facecolor="black", edgecolor="black")
    ax.add_patch(rect)

    # bbox
    x1, y1 = data_to_pixel(body_x, body_y)
    x2, y2 = data_to_pixel(body_x + body_w, body_y + body_h)
    y1, y2 = y2, y1
    scene_data.append({
        "name": side,
        "object_class": "gripper",
        "bbox_pixel": (x1, y1, x2, y2)
    })

    # Fingers
    finger_w, finger_h = 0.08, 0.20
    for i in [0, 1]:
        gx = body_x + i * (body_w - finger_w)
        gy = body_y - finger_h
        ax.add_patch(patches.Rectangle((gx, gy), finger_w, finger_h,
                                       facecolor="black", edgecolor="black"))

    # Ball if carried
    if carried is not None:
        cx, cy = body_x + body_w / 2, body_y - 0.35
        circ = plt.Circle((cx, cy), 0.2, fc="white", ec="black", lw=1.5)
        ax.add_patch(circ)

        bx1, by1 = data_to_pixel(cx - 0.2, cy - 0.2)
        bx2, by2 = data_to_pixel(cx + 0.2, cy + 0.2)
        by1, by2 = by2, by1
        scene_data.append({
            "name": carried,
            "object_class": "ball",
            "bbox_pixel": (bx1, by1, bx2, by2)
        })

        ax.text(cx, cy, carried[-1], ha="center", va="center", fontsize=12)


def overlay_bboxes(image_array, scene_data, color_map=None, thickness=3, draw_labels=True):
    if color_map is None:
        color_map = {"ball": (0, 120, 255), "gripper": (200, 0, 0), "room": (0, 200, 0)}

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


# === Rendering ===
def render(obs, add_object_labels=False, env=None, draw_bboxes=False):
    balls = []
    robby_room = None
    gripper_states = {"left": None, "right": None}
    scene_data = []

    for lit in obs:
        name = lit.predicate.name.lower()
        vars = [v.name for v in lit.variables]

        if name == "at-robby":
            robby_room = vars[0]
        elif name == "at":
            if "ball" in vars[0]:
                balls.append((vars[0], vars[1]))
        elif name == "carry":
            ball, grip = vars[0], vars[1]
            gripper_states[grip] = ball

    
    width, height = 3.2, 3.2
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0),
                                aspect='equal', frameon=False,
                                xlim=(-0.05, width + 0.05),
                                ylim=(-0.05, height + 0.05))
    ax.axis("off")

    # Rooms
    # Room A (top half)
    ax.add_patch(patches.Rectangle((0, 1.6), 3.2, 1.6, fill=False, lw=2))
    ax.annotate("Room A",
                xy=(2.6, 3.0), xycoords="data",
                ha="left", va="top", fontsize=8, weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.8))
    # bbox in pixel space
    x1, y1 = data_to_pixel(2.2, 2.4)
    x2, y2 = data_to_pixel(3.2, 3.2)
    y1, y2 = y2, y1
    scene_data.append({
        "name": "rooma",
        "object_class": "room",
        "bbox_pixel": (x1, y1, x2, y2),
        "color": (0, 200, 0),
        "unary_relationships": [],
        "binary_relationships": {}
    })

    # Room B (bottom half)
    ax.add_patch(patches.Rectangle((0, 0), 3.2, 1.6, fill=False, lw=2))
    ax.annotate("Room B",
                xy=(2.6, 1.4), xycoords="data",
                ha="left", va="top", fontsize=8, weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.8))
    x1, y1 = data_to_pixel(2.2, 0.8)
    x2, y2 = data_to_pixel(3.2, 1.4)
    y1, y2 = y2, y1
    scene_data.append({
        "name": "roomb",
        "object_class": "room",
        "bbox_pixel": (x1, y1, x2, y2),
        "color": (0, 200, 0),
        "unary_relationships": [],
        "binary_relationships": {}
    })


    # Balls
    ball_positions = {"rooma": (0.2, 1.8), "roomb": (0.2, 0.2)}
    carried_balls = {b for b in gripper_states.values() if b is not None}

    for ball, room in balls:
        if ball in carried_balls:
            continue
        base_x, base_y = ball_positions[room]
        ballind = int(ball[-1]) - 1
        x = base_x + env.obj_order[ballind] * 0.5
        y = base_y
        circ = plt.Circle((x, y), 0.2, fc="white", ec="black", lw=1)
        ax.add_patch(circ)

        x1, y1 = data_to_pixel(x - 0.2, y - 0.2)
        x2, y2 = data_to_pixel(x + 0.2, y + 0.2)
        y1, y2 = y2, y1
        scene_data.append({
            "name": ball,
            "object_class": "ball",
            "bbox_pixel": (x1, y1, x2, y2)
        })

        ax.text(x, y, ball[-1], ha="center", va="center", fontsize=12)

    # Grippers
    if robby_room is not None:
        draw_single_gripper(ax, robby_room, side="left", carried=gripper_states["left"], scene_data=scene_data)
        draw_single_gripper(ax, robby_room, side="right", carried=gripper_states["right"], scene_data=scene_data)

    img = fig2data(fig)
    plt.close(fig)

    # draw_bboxes = True
    if draw_bboxes:
        img = overlay_bboxes(img, scene_data)

    from lib.rendering.gripper.utils import add_relationships

    add_relationships(scene_data, obs)



    return img, scene_data