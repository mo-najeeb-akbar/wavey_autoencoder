#!/bin/bash

set -e  # Exit on any error

echo "=== Starting VAE Model Build Process ==="

echo "Step 1: Moving vae.onnx to cvt_models..."
mv /code/vae.onnx /code/cvt_models/src/model/
echo "✓ Moved vae.onnx to /code/cvt_models/src/model/"

echo "Step 2: Cleaning cvt_models and building..."
cd /code/cvt_models
rm -f /code/cvt_models/src/model/vae.bin
rm -f /code/cvt_models/src/model/vae.rs
# cargo clean
echo "✓ Cleaned cvt_models directory"

OUT_DIR=. cargo run
echo "✓ Built cvt_models with cargo run"

echo "Step 3: Copying model files to browser_models..."
cp src/model/vae.onnx /code/browser_models
echo "✓ Copied vae.onnx to browser_models"

cp src/model/vae.bin /code/browser_models
echo "✓ Copied vae.bin to browser_models"

cp src/model/vae.rs /code/browser_models/src
echo "✓ Copied vae.rs to browser_models/src"

echo "Step 4: Updating vae.rs file path reference..."
cd /code/browser_models/src
sed -i 's|include_bytes!("./src/model/vae.bin")|include_bytes!("../vae.bin")|g' vae.rs
echo "✓ Updated vae.rs file path reference"

echo "Step 5: Checking and removing existing pkg directory..."
cd /code/browser_models/
if [ -d "./pkg" ]; then
    echo "Found existing pkg directory, removing..."
    rm -rf ./pkg
    echo "✓ Removed existing pkg directory"
else
    echo "✓ No existing pkg directory found"
fi

echo "Step 6: Cleaning browser_models and building for web..."
# cargo clean
echo "✓ Cleaned browser_models directory"

./build_for_web.sh wgpu
echo "✓ Built for web with wgpu"

echo "Step 7: Creating tar.gz archive..."
tar -zcvf pkg.tar.gz pkg/
echo "✓ Created pkg.tar.gz archive"

echo "=== VAE Model Build Process Complete ==="
