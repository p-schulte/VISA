import bpy
import random

# Parameters
REALISTIC_RENDERING = False # Uses cycles for rendering. Is more realisitc but also takes more computation time
RENDER_ANNOTATIONS = True # If true, the generated scene will be annotated with bounding boxes around the blocks and the gripper

# Read the output file path from command-line arguments
import sys
if len(sys.argv) > 1:
    output_filepath = sys.argv[-1]  # Assuming the last argument is the file path
else:
    output_filepath = "generated_scene.png"  # Default value




# Clear existing objects
bpy.ops.wm.read_factory_settings(use_empty=True)

# Ensure GPU usage
prefs = bpy.context.preferences
cprefs = prefs.addons['cycles'].preferences
cprefs.compute_device_type = 'CUDA'
for device in cprefs.devices:
    if not device.use:
        device.use = True
bpy.context.scene.cycles.device = 'GPU'

if REALISTIC_RENDERING:
    bpy.context.scene.render.engine = 'CYCLES'
prefs = bpy.context.preferences
cprefs = prefs.addons['cycles'].preferences


# Set output resolution to 480x480
bpy.context.scene.render.resolution_x = 480
bpy.context.scene.render.resolution_y = 480
bpy.context.scene.render.resolution_percentage = 100

# Improve efficiency
scene = bpy.context.scene
scene.cycles.samples = 16  # Try 16–64 for drafts
scene.cycles.max_bounces = 3
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transmission_bounces = 2
scene.cycles.use_denoising = True


# Add background
if bpy.context.scene.world is None:
    bpy.context.scene.world = bpy.data.worlds.new("World")
bpy.context.scene.world.use_nodes = True
bg = bpy.context.scene.world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (0.2, 0.2, 0.2, .6)  # Soft gray
bg.inputs[1].default_value = 1.0  # Strength


# Add camera
x_jitter = random.uniform(-0.1, 0.1)
y_jitter = random.uniform(-0.1, 0.1)
z_jitter = random.uniform(-0.05, 0.05)
bpy.ops.object.camera_add(location=(0 + x_jitter, -17 + y_jitter, 4 + z_jitter), rotation=(1.7, 0, 0))
bpy.context.scene.camera = bpy.context.object
camera = bpy.context.scene.camera
camera.data.lens = 35


# Add light
import math
bpy.ops.object.light_add(type='SUN')
sun1 = bpy.context.object
sun1.rotation_euler = (math.radians(60), 0, math.radians(45))
sun1.data.energy = 5.0  # Default is 1.0 — try 5.0, 10.0, or even 20.0+
sun1.data.angle = 0.5  # For soft shadows
bpy.ops.object.light_add(type='SUN')
sun2 = bpy.context.object
sun2.rotation_euler = (math.radians(30), math.radians(45), math.radians(90))
sun2.data.energy = 3.0  # Adjust energy for balance
sun2.data.angle = 0.7  # Slightly softer shadows

# Add ground plane
bpy.ops.mesh.primitive_plane_add(size=17, location=(0, 2, 0))
plane = bpy.context.active_object
plane.cycles.is_shadow_catcher = False
plane_mat = bpy.data.materials.new(name="GroundMaterial")
plane_mat.use_nodes = True
nodes = plane_mat.node_tree.nodes
bsdf = nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.36, 0.22, 0.12, 1.0)  # Darker brown
bsdf.inputs["Roughness"].default_value = 0.6
plane.data.materials.append(plane_mat)




# Function to add a colored cube
def add_colored_cube(identifier, location, color_rgb, size=1.0):
    bpy.ops.mesh.primitive_cube_add(size=size, location=location)
    obj = bpy.context.active_object
    obj.name = f"Block_{color_rgb}"

    # Store identifier as a custom property
    obj["identifier"] = identifier
    obj["color"] = color_rgb


    # Add bevel for rounded edges
    bevel = obj.modifiers.new(name="Bevel", type='BEVEL')
    bevel.width = 0.05
    bevel.segments = 4
    bevel.profile = 0.7

    # Smooth shading
    bpy.ops.object.shade_smooth()

    # Add material
    mat = bpy.data.materials.new(name="Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color_rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.1
    bsdf.inputs["Metallic"].default_value = 0.8
    obj.data.materials.append(mat)


# Define stacks
# min_x = -6.5, max_x = 6.5
# min_y = 0.5, max_y = 10.5
# Load stacks from a JSON file
import json
with open('lib/rendering/blocks/realistic/input.json', 'r') as file:
    input_data = json.load(file)
    block_data = input_data[0]
    pile_data = input_data[1]
    holding_data = input_data[2]
    robot_data = input_data[3]
default_gripper_location = (0, 0, 14)
x = (robot_data['bbox_pixel'][0] + robot_data['bbox_pixel'][2]) / 2
y = (robot_data['bbox_pixel'][1] + robot_data['bbox_pixel'][3]) / 2
x = (x - 240) / 240 * 8
y = (240 - y) / 240 * 8 + 6
default_gripper_location = (x, 0, y)

# Synthesize stacks_data
stacks = []
for pile in pile_data:
    if pile == []:
        continue
    # Get the pile's position
    ele = next((block for block in block_data if block['name'] == pile[0]), None)
    pile_pos = (ele['bbox_pixel'][0] + ele['bbox_pixel'][2]) / 2
    pile_pos = (pile_pos - 240) / 240 * 6.5
    if abs(x-pile_pos) < 0.5:
        x = pile_pos
        default_gripper_location = (x, 0, y)
    for i, block in enumerate(pile):
        ele = next((block for block in block_data if block['name'] == pile[i]), None)
        stacks.append((ele['name'], (pile_pos, 0, 0.5 + 1 * i), ele['color']))
if holding_data[0] != None:
    ele = next((block for block in block_data if block['name'] == holding_data[0]), None)
    stacks.append((ele['name'], (x, 0, y-.8), ele['color']))
# Add all blocks
for identifier, pos, color in stacks:
    add_colored_cube(identifier, pos, color)




def add_gripper(location=(0, 0, 6), spacing=0.7, finger_size=(0.15, 0.15, 0.7), bridge_thickness=0.15):
    color = (0.1, 0.1, 0.1)

    def create_finger(offset_x):
        bpy.ops.mesh.primitive_cube_add(size=2.3, location=(location[0] + offset_x, location[1], location[2]))
        finger = bpy.context.active_object
        finger.scale = finger_size
        bpy.ops.object.shade_smooth()

        mat = bpy.data.materials.new(name="GripperMaterial")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.5
        finger.data.materials.append(mat)
        return finger

    # Fingers
    left = create_finger(-spacing)
    right = create_finger(spacing)

    # Top connector
    bridge_z = location[2] + finger_size[2] + bridge_thickness / 2
    bpy.ops.mesh.primitive_cube_add(size=2.5, location=(location[0], location[1], bridge_z))
    bridge = bpy.context.active_object
    bridge.scale = (spacing + finger_size[0], finger_size[1], bridge_thickness)
    bpy.ops.object.shade_smooth()

    mat = bpy.data.materials.new(name="GripperBridgeMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.5
    bridge.data.materials.append(mat)
    bridge.name = f"Robot"

# Add gripper to the scene
gripper_fingers = add_gripper(location=default_gripper_location)

# Get reference to the scene and camera
scene = bpy.context.scene
camera = scene.camera

# Get image size
render = scene.render
res_x = render.resolution_x * render.resolution_percentage / 100.0
res_y = render.resolution_y * render.resolution_percentage / 100.0

# Render image
bpy.context.scene.render.filepath = output_filepath
bpy.ops.render.render(write_still=True)






if RENDER_ANNOTATIONS:

    import bpy_extras
    from mathutils import Vector
    from PIL import Image, ImageDraw

    scene = bpy.context.scene
    camera = scene.camera

    # Render resolution
    res_x = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
    res_y = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)

    bbox_data = []
    scene_data = []

    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.name.startswith("Block_"):
            # Get 8 corners of bounding box in world coords
            bbox_world = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
            # Project each corner to 2D
            coords_2d = [bpy_extras.object_utils.world_to_camera_view(scene, camera, coord) for coord in bbox_world]

            # Convert to pixel space
            coords_px = [(int(c.x * res_x), int((1 - c.y) * res_y)) for c in coords_2d]
            xs, ys = zip(*coords_px)
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            bbox_data.append(((min_x, min_y), (max_x, max_y)))
            obj_data = {
                'name': obj["identifier"],
                'color': tuple(obj["color"]),
                'bbox_pixel': (min_x, min_y, max_x, max_y)
            }
            scene_data.append(obj_data)
        elif obj.type == 'MESH' and obj.name.startswith("Robot"):
            # Get 8 corners of bounding box in world coords
            bbox_world = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
            # Project each corner to 2D
            coords_2d = [bpy_extras.object_utils.world_to_camera_view(scene, camera, coord) for coord in bbox_world]

            # Convert to pixel space
            coords_px = [(int(c.x * res_x), int((1 - c.y) * res_y)) for c in coords_2d]
            xs, ys = zip(*coords_px)
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            max_y += 10 # Adjust for the height of the blocks

            bbox_data.append(((min_x, min_y), (max_x, max_y)))
            obj_data = {
                'name': "robot",
                'color': (0.1, 0.1, 0.1),
                'bbox_pixel': (min_x, min_y, max_x, max_y)
            }
            scene_data.append(obj_data)

    # Add table and robot to bboxes:
    table_bbox = ((10, 430), (470, 460))
    bbox_data.append(table_bbox)
    scene_data.append({
        'name': "table",
        'color': (0.36, 0.22, 0.12),
        'bbox_pixel': (10, 430, 470, 460)
    })

    '''
    robot_bbox = ((210, 12), (270, 42))
    bbox_data.append(robot_bbox)
    scene_data.append({
        'name': "robot",
        'color': (0.1, 0.1, 0.1),
        'bbox_pixel': (210, 12, 270, 42)
    })
    '''

    img = Image.open(output_filepath)
    draw = ImageDraw.Draw(img)

    for bbox in bbox_data:
        draw.rectangle(bbox, outline="red", width=2)

    with open("lib/rendering/blocks/realistic/output.json", "w") as json_file:
        json.dump(
            scene_data,
            json_file,
            indent=4
        )

    img.save(output_filepath.replace(".png", "_annotated.png"))
