#!/bin/bash
# Docker run script with interactive options
# Usage: ./run_docker.sh

echo "=== Docker Container Setup ==="
echo

# Image Selection
echo "Available Docker Images:"
echo "========================"

# Get list of available images and format them nicely
images=($(docker images --format "table {{.Repository}}:{{.Tag}}" | tail -n +2 | grep -v "<none>" | sort))

if [ ${#images[@]} -eq 0 ]; then
    echo "❌ No Docker images found!"
    echo "Please build or pull some Docker images first."
    exit 1
fi

# Display images with numbers
for i in "${!images[@]}"; do
    printf "%2d) %s\n" $((i+1)) "${images[$i]}"
done

echo
read -p "Select image number (1-${#images[@]}): " image_selection

# Validate selection
if ! [[ "$image_selection" =~ ^[0-9]+$ ]] || [ "$image_selection" -lt 1 ] || [ "$image_selection" -gt ${#images[@]} ]; then
    echo "❌ Invalid selection. Exiting."
    exit 1
fi

# Get selected image
SELECTED_IMAGE="${images[$((image_selection-1))]}"
echo "✓ Selected: $SELECTED_IMAGE"
echo

# Get port configuration
echo "Port Configuration:"
echo "Current default: 8889:8889"
read -p "Enter port mapping (default: 8889:8889): " port_input
PORT=${port_input:-"8889:8889"}

# Get container name with smart default
echo
echo "Container Name:"
# Generate a default name based on the selected image with timestamp for uniqueness
BASE_NAME=$(echo "$SELECTED_IMAGE" | sed 's/:/_/g' | sed 's/\//_/g')
TIMESTAMP=$(date +"%m%d_%H%M")
DEFAULT_NAME="${BASE_NAME}_${TIMESTAMP}"
echo "Suggested: $DEFAULT_NAME"
read -p "Enter container name (press Enter for suggested, 'none' to skip): " container_name_input

case $container_name_input in
    "none"|"NONE")
        container_name=""
        echo "No container name will be used (Docker will auto-generate)"
        ;;
    "")
        container_name="$DEFAULT_NAME"
        echo "✓ Using: $container_name"
        ;;
    *)
        # If user provides custom name, add timestamp to make it unique
        CUSTOM_NAME="${container_name_input}_${TIMESTAMP}"
        container_name="$CUSTOM_NAME"
        echo "✓ Using: $container_name (timestamp added for uniqueness)"
        ;;
esac

# Volume configuration - always mount current directory to /code
VOLUME_ARG="-v `pwd`:/code"
echo
echo "Volume Configuration:"
echo "✓ Current directory will be mounted to /code"

# Get optional /data mount
echo
read -p "Do you want to mount a directory to /data? (y/N): " mount_data
case $mount_data in
    [Yy]*)
        read -p "Enter the directory path to mount to /data: " data_dir
        if [ -d "$data_dir" ]; then
            VOLUME_ARG="$VOLUME_ARG -v $data_dir:/data"
            echo "✓ $data_dir will be mounted to /data"
        else
            echo "⚠ Directory doesn't exist: $data_dir"
            read -p "Create directory and continue? (y/N): " create_dir
            case $create_dir in
                [Yy]*)
                    mkdir -p "$data_dir"
                    VOLUME_ARG="$VOLUME_ARG -v $data_dir:/data"
                    echo "✓ Created and mounted: $data_dir → /data"
                    ;;
                *)
                    echo "Skipping /data mount"
                    ;;
            esac
        fi
        ;;
    *)
        echo "No /data mount"
        ;;
esac

# GPU Configuration
echo
read -p "Enable GPU support? (Y/n): " gpu_support
case $gpu_support in
    [Nn]*)
        GPU_ARG=""
        echo "GPU support disabled"
        ;;
    *)
        GPU_ARG="--gpus all"
        echo "✓ GPU support enabled"
        ;;
esac

# Build the docker command
DOCKER_CMD="docker run -it"

# Add GPU support if enabled
if [ ! -z "$GPU_ARG" ]; then
    DOCKER_CMD="$DOCKER_CMD $GPU_ARG"
fi

# Add port mapping
DOCKER_CMD="$DOCKER_CMD -p $PORT"

# Add container name if provided
if [ ! -z "$container_name" ]; then
    DOCKER_CMD="$DOCKER_CMD --name $container_name"
fi

# Add volume argument
DOCKER_CMD="$DOCKER_CMD $VOLUME_ARG"

# Add the selected image
DOCKER_CMD="$DOCKER_CMD $SELECTED_IMAGE"

# Show the final command
echo
echo "=== Final Command ==="
echo "$DOCKER_CMD"
echo

# Confirm before running
read -p "Run this command? (Y/n): " confirm
case $confirm in
    [Nn]*)
        echo "Command not executed."
        echo "You can run it manually:"
        echo "$DOCKER_CMD"
        ;;
    *)
        echo "Running Docker container..."
        eval $DOCKER_CMD
        ;;
esac