#!/bin/bash

# Jupyter notebook startup script with auto port detection
# Usage: ./start_jupyter.sh

echo "=== Starting Jupyter Notebook ==="
echo

# Try to detect the port from various sources
DETECTED_PORT=""

# Method 1: Check if we're in a container and look for exposed ports
if [ -f /.dockerenv ]; then
    echo "Detected running inside Docker container"
    
    # Try to get port from environment or network info
    # Check common environment variables that might contain port info
    if [ ! -z "$JUPYTER_PORT" ]; then
        DETECTED_PORT=$JUPYTER_PORT
        echo "Found port from JUPYTER_PORT environment variable: $DETECTED_PORT"
    elif [ ! -z "$PORT" ]; then
        DETECTED_PORT=$PORT
        echo "Found port from PORT environment variable: $DETECTED_PORT"
    else
        # Try to detect from netstat or ss if available
        if command -v netstat >/dev/null 2>&1; then
            # Look for commonly mapped ports
            COMMON_PORTS="8889 8888 8890 8891 8892"
            for port in $COMMON_PORTS; do
                if netstat -ln 2>/dev/null | grep -q ":$port "; then
                    continue  # Port is already in use
                else
                    DETECTED_PORT=$port
                    echo "Selected available port: $DETECTED_PORT"
                    break
                fi
            done
        fi
        
        # If still no port detected, default to 8889
        if [ -z "$DETECTED_PORT" ]; then
            DETECTED_PORT=8889
            echo "Using default port: $DETECTED_PORT"
        fi
    fi
else
    echo "Not running in Docker container"
    DETECTED_PORT=8888
    echo "Using standard Jupyter port: $DETECTED_PORT"
fi

# Allow user to override the detected port
echo
read -p "Use port $DETECTED_PORT? Press Enter to confirm or type a different port: " user_port
if [ ! -z "$user_port" ]; then
    DETECTED_PORT=$user_port
    echo "Using user-specified port: $DETECTED_PORT"
fi

# Additional Jupyter options
echo
echo "Additional Options:"
read -p "Set a custom notebook directory? (press Enter to use current): " notebook_dir

# Build the Jupyter command
JUPYTER_CMD="jupyter notebook --no-browser --port=$DETECTED_PORT --ip=0.0.0.0 --allow-root"

# Add notebook directory if specified
if [ ! -z "$notebook_dir" ]; then
    if [ -d "$notebook_dir" ]; then
        JUPYTER_CMD="$JUPYTER_CMD --notebook-dir='$notebook_dir'"
        echo "Using notebook directory: $notebook_dir"
    else
        echo "Directory doesn't exist: $notebook_dir"
        read -p "Create directory and continue? (y/N): " create_nb_dir
        case $create_nb_dir in
            [Yy]*)
                mkdir -p "$notebook_dir"
                JUPYTER_CMD="$JUPYTER_CMD --notebook-dir='$notebook_dir'"
                echo "Created and using directory: $notebook_dir"
                ;;
            *)
                echo "Using current directory"
                ;;
        esac
    fi
fi

# Show the final command
echo
echo "=== Final Command ==="
echo "$JUPYTER_CMD"
echo

# Show connection info
echo "=== Connection Information ==="
if [ -f /.dockerenv ]; then
    echo "Once started, access Jupyter at: http://localhost:$DETECTED_PORT"
    echo "If running on a remote server, use: http://YOUR_SERVER_IP:$DETECTED_PORT"
else
    echo "Once started, access Jupyter at: http://localhost:$DETECTED_PORT"
fi
echo

# Confirm and run
read -p "Start Jupyter Notebook? (Y/n): " confirm
case $confirm in
    [Nn]*)
        echo "Jupyter not started."
        echo "You can run it manually with:"
        echo "$JUPYTER_CMD"
        ;;
    *)
        echo "Starting Jupyter Notebook..."
        echo "Use Ctrl+C to stop the server"
        echo
        eval $JUPYTER_CMD
        ;;
esac
