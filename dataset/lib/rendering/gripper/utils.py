def add_relationships(scene_data, obs):
    """
    Attach unary and binary relationships to scene_data objects based on literals in obs.
    Handles 'at', 'carry', 'free', 'at-robby', and 'different'.
    Ensures no duplicates in relationships.
    """
    # default color mapping
    color_map = {"ball": (0, 120, 255), "gripper": (200, 0, 0), "room": (0, 200, 0)}

    obs_list = list(obs)
    name_to_obj = {obj["name"]: obj for obj in scene_data}

    # init
    for obj in scene_data:
        obj.setdefault("unary_relationships", [])
        obj.setdefault("binary_relationships", {})
        obj.setdefault("color", color_map.get(obj["object_class"], (128, 128, 128)))
        # use sets internally to avoid duplicates
        for k in obj["binary_relationships"]:
            obj["binary_relationships"][k] = set(obj["binary_relationships"][k])

    for lit in obs_list:
        pred = lit.predicate.name.lower()
        vars = [v.name for v in lit.variables]

        # === Unary ===
        if len(vars) == 1:
            target = name_to_obj.get(vars[0])
            if target:
                if pred not in target["unary_relationships"]:
                    target["unary_relationships"].append(pred)

        # === Binary ===
        elif len(vars) == 2:
            subj, obj = vars
            subj_entry = name_to_obj.get(subj)
            obj_entry = name_to_obj.get(obj)
            if not subj_entry or not obj_entry:
                continue

            subj_entry["binary_relationships"].setdefault(pred, set()).add(obj)

            # symmetric case
            if pred == "different":
                obj_entry["binary_relationships"].setdefault(pred, set()).add(subj)

    # convert sets back to lists
    for obj in scene_data:
        for k, v in obj["binary_relationships"].items():
            if isinstance(v, set):
                obj["binary_relationships"][k] = list(v)
