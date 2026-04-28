
import json

input_file = "/Users/ulzhalgasrakhman/graph-separator/json_files/grids_fixed.json"
output_file = "/Users/ulzhalgasrakhman/graph-separator/json_files/grids_fixed2.json"

with open(input_file) as f:
    data = json.load(f)

for step in data:
    if "action-name" in step:
        step["action_name"] = step.pop("action-name")

    if "action-objects" in step:
        step["action_objects"] = step.pop("action-objects")

with open(output_file, "w") as f:
    json.dump(data, f, indent=4)

print("Fixed JSON written to", output_file)






'''



import json

INPUT_FILE = "/Users/ulzhalgasrakhman/graph-separator/json_files/grids_new.json"
OUTPUT_FILE = "/Users/ulzhalgasrakhman/graph-separator/json_files/grids_fixed.json"

# predicates that must stay binary
KEEP_BINARY = {"adj-right", "adj-left", "adj-above", "adj-below"}

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

for step in data:
    state = step.get("state", {})
    
    for predicate, values in state.items():
        
        # skip adjacency predicates
        if predicate in KEEP_BINARY:
            continue
        
        # skip meta fields
        if predicate in {"detected_objects", "detected_object_names"}:
            continue
        
        new_values = []
        
        for grounding in values:
            # convert ["p19","p19"] → ["p19"]
            if isinstance(grounding, list) and len(grounding) == 2 and grounding[0] == grounding[1]:
                new_values.append([grounding[0]])
            else:
                new_values.append(grounding)
        
        state[predicate] = new_values

with open(OUTPUT_FILE, "w") as f:
    json.dump(data, f, indent=4)

print("Conversion finished. Output saved to:", OUTPUT_FILE)
'''