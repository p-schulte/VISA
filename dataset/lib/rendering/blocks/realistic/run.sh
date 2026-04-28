#!/bin/bash

# Define Blender download URL and target folder
blender_url="https://ftp.nluug.nl/pub/graphics/blender/release/Blender4.4/blender-4.4.0-linux-x64.tar.xz"
blender_folder="blender-4.4.0-linux-x64"

# Check if Blender folder exists
if [ ! -d "$blender_folder" ]; then
    echo "Blender folder not found. Downloading Blender..."
    wget "$blender_url" -O blender.tar.xz
    echo "Extracting Blender..."
    tar -xf blender.tar.xz
    rm blender.tar.xz
    blender-4.4.0-linux-x64/4.4/python/bin/python* -m pip install Pillow
    echo "Blender downloaded and extracted."
fi

# Check if output file argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <output_file>"
    exit 1
fi

output_file="$1"

# Run Blender
$blender_folder/blender -noaudio --background --python lib/rendering/blocks/realistic/generate.py "$output_file"
