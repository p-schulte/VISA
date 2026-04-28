import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

def encode_scene_xml(scene_data, output_dir, filename, img_size=(480, 480)):
    # Create XML structure
    annotation = ET.Element("annotation")

    # Add folder and filename
    folder = ET.SubElement(annotation, "folder")
    folder.text = output_dir
    filename_elem = ET.SubElement(annotation, "filename")
    filename_elem.text = filename

    # Add source information
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "..."
    ET.SubElement(source, "annotation").text = "..."
    #ET.SubElement(source, "image").text = "flickr"
    #ET.SubElement(source, "flickrid").text = "341012865"

    # Add owner information
    owner = ET.SubElement(annotation, "owner")
    #ET.SubElement(owner, "flickrid").text = "Fried Camels"
    ET.SubElement(owner, "name").text = "Paul Schulte"

    # Add size information
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(img_size[0])
    ET.SubElement(size, "height").text = str(img_size[1])
    ET.SubElement(size, "depth").text = "4" # 4 channels (RGBA)

    # Add segmented information
    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0" # 1 if image segmentation is present, 0 otherwise

    # Add objects
    for obj in scene_data:
        object_elem = ET.SubElement(annotation, "object")
        ET.SubElement(object_elem, "name").text = obj["object_class"]
        ET.SubElement(object_elem, "truncated").text = "0" # 1 if object is partially visible, 0 if fully visible
        ET.SubElement(object_elem, "pose").text = "Unspecified" # Object pose (Unspecified, Left, Right, Frontal, Rear)
        ET.SubElement(object_elem, "difficult").text = "0" # 1 if object is difficult to recognize, 0 otherwise
        
        bndbox = ET.SubElement(object_elem, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(obj["bbox_pixel"][0])
        ET.SubElement(bndbox, "ymin").text = str(obj["bbox_pixel"][1])
        ET.SubElement(bndbox, "xmax").text = str(obj["bbox_pixel"][2])
        ET.SubElement(bndbox, "ymax").text = str(obj["bbox_pixel"][3])

    # Convert to string and beautify
    rough_string = ET.tostring(annotation, encoding="utf-8")
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")  # 2-space indentation

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save to file
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)

    #print(f"XML file generated successfully at {output_path}")


if __name__ == "__main__":
    # Example usage
    scene_data = [
        {'name': 'b', 'object_class': 'block', 'color': [0.9473705904889242, 0.7308558067701578, 0.25394164259502583], 'bbox_pixel': [85, 403, 128, 377]},
        {'name': 'a', 'object_class': 'block', 'color': [0.272821902424467, 0.3708527992178887, 0.19705428018563964], 'bbox_pixel': [85, 377, 128, 350]},
        {'name': 'f', 'object_class': 'block', 'color': [0.9, 0.1, 0.1], 'bbox_pixel': [351, 403, 394, 377]},
        {'name': 'e', 'object_class': 'block', 'color': [0.4375872112626925, 0.8917730007820798, 0.9636627605010293], 'bbox_pixel': [351, 377, 394, 350]},
        {'name': 'd', 'object_class': 'block', 'color': [0.15896958364551972, 0.11037514116430513, 0.6563295894652734], 'bbox_pixel': [351, 350, 394, 324]},
        {'name': 'c', 'object_class': 'block', 'color': [0.14944830465799375, 0.8681260573682142, 0.16249293467637482], 'bbox_pixel': [351, 324, 394, 297]},
        {'name': 'g', 'object_class': 'block', 'color': [0.9615701545414985, 0.2317016264712045, 0.9493188224156814], 'bbox_pixel': [417, 403, 461, 377]},
        {'name': 'table', 'object_class': 'table', 'color': [0.5, 0.2, 0.0], 'bbox_pixel': [7, 473, 472, 403]},
        {'name': 'robot', 'object_class': 'robot', 'color': [0.4, 0.4, 0.4], 'bbox_pixel': [209, 54, 270, 8]}
    ]

    output_dir = "/path/to/output"
    filename = "scene_annotation.xml"
    encode_scene_xml(scene_data, output_dir, filename)
