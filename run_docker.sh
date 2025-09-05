#!/bin/bash

# Docker run script with interactive options
# Usage: ./run_docker.sh

echo "=== Docker Container Setup ==="
echo

# Get port configuration
echo "Port Configuration:"
echo "Current default: 8889:8889"
read -p "Enter port mapping (default: 8889:8889): " port_input
PORT=${port_input:-"8889:8889"}

# Get container name
echo
echo "Container Name:"
read -p "Enter container name (optional, press Enter to skip): " container_name

# Volume configuration - always mount current directory to /code
VOLUME_ARG="-v `pwd`:/code"
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

# Build the docker command
DOCKER_CMD="docker run -it --gpus all -p $PORT"

# Add container name if provided
if [ ! -z "$container_name" ]; then
    DOCKER_CMD="$DOCKER_CMD --name $container_name"
fi

# Add volume argument if set
if [ ! -z "$VOLUME_ARG" ]; then
    DOCKER_CMD="$DOCKER_CMD $VOLUME_ARG"
fi

# Add the image name
DOCKER_CMD="$DOCKER_CMD optical"

# Show the final command
echo
echo "=== Final Command ==="
echo "$DOCKER_CMD"
echo

# Confirm before running
read -p "Run this command? (y/N): " confirm
case $confirm in
    [Yy]*)
        echo "Running Docker container..."
        eval $DOCKER_CMD
        ;;
    *)
        echo "Command not executed."
        echo "You can run it manually:"
        echo "$DOCKER_CMD"
        ;;
esac
