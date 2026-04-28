#!/bin/bash

# Define paths to be cleaned
CACHE_DIRS=(
    "$(pwd)/fasterRCNN/data/scene_understanding/VOCdevkit2025/results"
    "$(pwd)/fasterRCNN/data/scene_understanding/VOCdevkit2025/annotations_cache"
    "$(pwd)/fasterRCNN/data/scene_understanding/cache"
)

# Loop through directories and remove them if they exist
for dir in "${CACHE_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "Removing $dir"
        rm -rf "$dir"
    else
        echo "Skipping $dir (not found)"
    fi
done

echo "Cache cleaning completed."

