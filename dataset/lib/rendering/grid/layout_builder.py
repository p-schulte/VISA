import numpy as np

def obj_name(o) -> str:
    return str(o).split(":")[0]

def shape_id(shape_obj) -> int:
    s = obj_name(shape_obj)
    assert s.startswith("shape")
    return int(s[len("shape"):])  # shape3 -> 3


def build_layout(obs):
    """
    Build separate arrays:

      walls      : bool [H,W]
      floor      : bool [H,W]
      doors      : uint8 [H,W]  (0 = none, k = shape_id+1)
      door_open  : bool [H,W]
      keys       : uint8 [H,W]  (0 = none, k = shape_id+1)
      player     : bool [H,W]
    """

    # ---------- 1) Collect places and coordinates ----------
    place_to_coord = {}
    for lit in obs:
        if lit.predicate.name == "place":
            raw = obj_name(lit.variables[0])  # e.g. "p-3-4"
            parts = raw.split("-")
            if len(parts) != 3:
                continue
            _, cx, cy = parts
            x = int(cx)
            y = int(cy)
            place_to_coord[raw] = (x, y)

    if not place_to_coord:
        H = W = 1
        return {
            "walls": np.ones((H, W), dtype=bool),
            "floor": np.zeros((H, W), dtype=bool),
            "doors": np.zeros((H, W), dtype=np.uint8),
            "door_open": np.zeros((H, W), dtype=bool),
            "keys": np.zeros((H, W), dtype=np.uint8),
            "player": np.zeros((H, W), dtype=bool),
        }

    coords = list(place_to_coord.values())
    max_x = max(c for c, r in coords)
    max_y = max(r for c, r in coords)
    W = max_x + 1
    H = max_y + 1

    # ---------- 2) Init arrays ----------
    walls     = np.ones((H, W), dtype=bool)
    floor     = np.zeros((H, W), dtype=bool)
    doors     = np.zeros((H, W), dtype=np.uint8)
    door_open = np.zeros((H, W), dtype=bool)
    keys      = np.zeros((H, W), dtype=np.uint8)
    player    = np.zeros((H, W), dtype=bool)

    # mark all known places as floor (non-wall)
    for p_name, (x, y) in place_to_coord.items():
        floor[y, x] = True
        walls[y, x] = False

    # ---------- 3) Collect door info (open/locked + shape) ----------
    open_places   = set()
    locked_places = set()
    door_shapes   = {}  # place_name -> shape_id

    for lit in obs:
        name = lit.predicate.name
        if name == "open":
            p = obj_name(lit.variables[0])
            open_places.add(p)
        elif name == "locked":
            p = obj_name(lit.variables[0])
            locked_places.add(p)
        elif name == "lock-shape":
            p   = obj_name(lit.variables[0])
            sid = shape_id(lit.variables[1])
            door_shapes[p] = sid

    # ---------- 4) Keys ----------
    key_shapes = {}  # key_name -> shape_id
    keys_at    = {}  # place_name -> [shape_id]

    for lit in obs:
        if lit.predicate.name == "key-shape":
            kname = obj_name(lit.variables[0])
            sid   = shape_id(lit.variables[1])
            key_shapes[kname] = sid

    for lit in obs:
        if lit.predicate.name == "at":
            kname = obj_name(lit.variables[0])
            pname = obj_name(lit.variables[1])
            sid   = key_shapes[kname]
            keys_at.setdefault(pname, []).append(sid)

    # ---------- 5) Robot ----------
    robot_place = None
    for lit in obs:
        if lit.predicate.name == "at-robot":
            robot_place = obj_name(lit.variables[0])

    # ---------- 6) Fill doors / door_open / keys ----------
    for p_name, (x, y) in place_to_coord.items():
        # doors
        if p_name in door_shapes:
            sid = door_shapes[p_name]
            doors[y, x] = sid + 1  # encode 1..N instead of 0..N-1
            if p_name in open_places:
                door_open[y, x] = True
            elif p_name in locked_places:
                door_open[y, x] = False
            # if neither open nor locked is present, you can decide a default

        # keys
        if p_name in keys_at:
            # if multiple keys in one cell: choose first or sum, your call
            sid = keys_at[p_name][0]
            keys[y, x] = sid + 1

    # ---------- 7) Player ----------
    if robot_place is not None and robot_place in place_to_coord:
        rx, ry = place_to_coord[robot_place]
        player[ry, rx] = True

    return {
        "walls": walls,
        "floor": floor,
        "doors": doors,
        "door_open": door_open,
        "keys": keys,
        "player": player,
    }
