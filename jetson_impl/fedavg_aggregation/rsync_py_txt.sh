#!/bin/bash

# Your remote password
PASS="123mxenos123"

# Remote user and IP addresses
BOARDS=("board1" "board2" "board3")

# Folders to sync
FOLDERS=("fedavg" "fl" "fl_adaptive_freeze" "fl_swap")

# Remote destination base path
DEST_BASE="./jetson_impl/fedavg_aggregation"

for board in "${BOARDS[@]}"; do
    echo "------------------------------------------------"
    echo "Syncing to $board..."
    
    for folder in "${FOLDERS[@]}"; do
        echo "Updating $folder..."
        
        # EXPLAINING THE FLAGS:
        # -v: verbose, -z: compress, --progress: show speed
        # --include="*.py" --include="*.txt": only these extensions
        # --exclude="*/": DO NOT go into subdirectories (Crucial for your request)
        # --exclude="*": ignore everything else
        
        sshpass -p "$PASS" rsync -vz --progress \
        --include="*.py" \
        --include="*.txt" \
        --exclude="*/" \
        --exclude="*" \
        "./$folder/" "$board:$DEST_BASE/$folder/"
    done
done

echo "------------------------------------------------"
echo "✅ All boards are up to date!"
