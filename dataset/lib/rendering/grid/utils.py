import numpy as np
import cv2

def draw_key(img, x0, y0, tile_size, color, bg_color=(255, 255, 255)):
    """Draw a key inside one tile with top-left corner (x0, y0)."""
    outline = (0, 0, 0)

    cx = x0 + tile_size // 2          # center x
    cy = y0 + tile_size // 3          # center y of the head
    head_r = tile_size // 4           # radius of the head

    # --- round head ---
    # outer filled circle
    cv2.circle(img, (cx, cy), head_r, color, -1)
    cv2.circle(img, (cx, cy), head_r, outline, 1)

    # inner hole
    cv2.circle(img, (cx, cy), head_r // 2, bg_color, -1)
    cv2.circle(img, (cx, cy), head_r // 2, outline, 1)

    # --- shaft ---
    shaft_w = max(2, tile_size // 6)
    shaft_top = cy + head_r // 2
    shaft_bot = y0 + int(tile_size * 0.85)

    cv2.rectangle(
        img,
        (cx - shaft_w // 2, shaft_top),
        (cx + shaft_w // 2, shaft_bot),
        color,
        -1
    )
    cv2.rectangle(
        img,
        (cx - shaft_w // 2, shaft_top),
        (cx + shaft_w // 2, shaft_bot),
        outline,
        1
    )

    # --- teeth ---
    tooth_h = max(2, tile_size // 8)
    tooth_w = max(2, tile_size // 5)

    # one tooth to the right
    cv2.rectangle(
        img,
        (cx, shaft_bot - tooth_h),
        (cx + tooth_w, shaft_bot),
        color,
        -1
    )
    cv2.rectangle(
        img,
        (cx, shaft_bot - tooth_h),
        (cx + tooth_w, shaft_bot),
        outline,
        1
    )


def render_grid_layout(layers, tile_size=32, line_thickness=1):
    walls     = layers["walls"]
    floor     = layers["floor"]
    doors     = layers["doors"]
    door_open = layers["door_open"]
    keys      = layers["keys"]
    player    = layers["player"]

    H, W = walls.shape
    img = np.ones((H * tile_size, W * tile_size, 3), dtype=np.uint8) * 40  # dark bg

    # Colors
    COLOR_CLEAR  = (30, 30, 30)
    COLOR_WALL   = (160, 160, 160)
    COLOR_PLAYER = (0,   0, 255)

    KEY_COLORS = [
        (0, 200, 200),
        (0, 0, 180),
        (0, 180, 0),
        (180, 0, 0),
        (180, 0, 180),
    ]
    DOOR_COLORS = [
        (0, 200, 200),
        (0, 0, 180),
        (0, 180, 0),
        (180, 0, 0),
        (180, 0, 180),
    ]

    for y in range(H):
        for x in range(W):
            y0 = y * tile_size
            x0 = x * tile_size
            y1 = y0 + tile_size
            x1 = x0 + tile_size

            # ---- base: wall / floor ----
            if walls[y, x]:
                cv2.rectangle(img, (x0, y0), (x1, y1), COLOR_WALL, -1)
            elif floor[y, x]:
                cv2.rectangle(img, (x0, y0), (x1, y1), COLOR_CLEAR, -1)
            else:
                # outside world (shouldn't happen if walls is consistent)
                cv2.rectangle(img, (x0, y0), (x1, y1), (20, 20, 20), -1)

            # ---- door (on top of floor) ----
            door_sid = doors[y, x]  # 0 = no door, k>0 shape_id+1
            if door_sid > 0:
                color = DOOR_COLORS[(door_sid - 1) % len(DOOR_COLORS)]
                if door_open[y, x]:
                    # open door: outline
                    cv2.rectangle(
                        img,
                        (x0 + 6, y0 + 6),
                        (x1 - 6, y1 - 6),
                        color,
                        2
                    )
                else:
                    # locked door: filled block
                    cv2.rectangle(
                        img,
                        (x0 + 6, y0 + 6),
                        (x1 - 6, y1 - 6),
                        color,
                        -1
                    )
                    cv2.rectangle(
                        img,
                        (x0 + 6, y0 + 6),
                        (x1 - 6, y1 - 6),
                        (0, 0, 0),
                        1
                    )

            # ---- key ----
            key_sid = keys[y, x]  # 0 = no key, k>0 shape_id+1
            if key_sid > 0:
                color = KEY_COLORS[(key_sid - 1) % len(KEY_COLORS)]
                draw_key(img, x0, y0, tile_size, color, bg_color=COLOR_CLEAR)

            # Old way of drawing keys:
            # key_sid = keys[y, x]  # 0 = no key, k>0 shape_id+1
            # if key_sid > 0:
            #     color = KEY_COLORS[(key_sid - 1) % len(KEY_COLORS)]
            #     cv2.circle(
            #         img,
            #         (x0 + tile_size // 2, y0 + tile_size // 2),
            #         tile_size // 4,
            #         color,
            #         -1
            #     )
            #     cv2.circle(
            #         img,
            #         (x0 + tile_size // 2, y0 + tile_size // 2),
            #         tile_size // 4,
            #         (0, 0, 0),
            #         1
            #     )

            # ---- player (on top of everything) ----
            if player[y, x]:
                pts = np.array([
                    [x0 + tile_size // 2, y0 + 4],
                    [x0 + 4,              y1 - 4],
                    [x1 - 4,              y1 - 4]
                ])
                cv2.drawContours(img, [pts], 0, COLOR_PLAYER, -1)

            # grid lines
            cv2.rectangle(img, (x0, y0), (x1, y1), (80, 80, 80), line_thickness)

    return img
