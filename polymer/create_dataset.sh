#!/bin/bash

set -e  # Exit on any error

echo "=== Dataset Generation Script ==="
echo

# Step 1: Create config
echo "1. Creating initial config..."
python generate_dataset.py --create-config
echo "✓ Config template created"
echo

# Step 2: Get user input for paths
echo "2. Configure paths:"
read -p "Input folder path [/data/sample_pe]: " input_path
read -p "Output folder path [/data/processed_pe]: " output_path

# Use defaults if empty
input_path=${input_path:-"/data/sample_pe"}
output_path=${output_path:-"/data/processed_pe"}

# Step 3: Update config.json
echo "3. Updating config.json..."
python3 -c "
import json
with open('config.json', 'r') as f:
    config = json.load(f)
config['input_folder'] = '$input_path'
config['output_folder'] = '$output_path'
with open('config.json', 'w') as f:
    json.dump(config, f, indent=2)
"
echo "✓ Config updated with your paths"
echo

# Step 4: Show config and confirm
echo "Current configuration:"
cat config.json
echo
read -p "Proceed with generation? [y/N]: " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Step 5: Generate dataset
echo "4. Generating dataset..."
python generate_dataset.py config.json
echo "✓ Dataset generated"
echo

# Step 6: Generate TFRecords using the processed data
echo "5. Creating TFRecords..."
read -p "TFRecords directory path: " tfrecord_dir
# Remove leading /data/ if user includes it
tfrecord_dir=${tfrecord_dir#/data/}
tfrecord_path="/data/$tfrecord_dir/dataset.tfrecord"

echo "Using processed data from: $output_path"
echo "Creating TFRecords at: $tfrecord_path"

# Create directory if needed
mkdir -p "$(dirname "$tfrecord_path")"

python generate_tfrecords.py "$output_path" "$tfrecord_path" 2
echo "✓ TFRecords created at: $tfrecord_path"
echo

echo "🎉 Dataset generation complete!"
