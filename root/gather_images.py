#!/usr/bin/env python3
import os
import sys
import shutil
import re
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import time

def clean_filename(filename):
    """
    Clean filename by removing spaces, parentheses, and standardizing
    """
    # Remove file extension
    name, ext = os.path.splitext(filename)
    
    # Remove spaces and parentheses, replace with underscores
    name = re.sub(r'[\s\(\)]', '_', name)
    
    # Remove multiple consecutive underscores
    name = re.sub(r'_+', '_', name)
    
    # Remove leading/trailing underscores
    name = name.strip('_')
    
    # Return with .jpg extension (standardize to lowercase)
    return f"{name}.jpg"

def find_jpg_files(directory):
    """
    Recursively find all JPG files in directory and subdirectories
    """
    jpg_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                jpg_files.append(os.path.join(root, file))
    return jpg_files

def process_single_image(args):
    """
    Process a single image - designed to be called by worker processes
    Returns (success, original_path, output_path, error_msg)
    """
    input_path, output_dir, original_filename = args
    
    try:
        # Clean the filename
        clean_name = clean_filename(original_filename)
        
        # Handle duplicate names by adding a number
        output_path = os.path.join(output_dir, clean_name)
        counter = 1
        base_name, ext = os.path.splitext(clean_name)
        
        # Thread-safe way to handle duplicates (not perfect but good enough for most cases)
        while os.path.exists(output_path):
            clean_name = f"{base_name}_{counter}{ext}"
            output_path = os.path.join(output_dir, clean_name)
            counter += 1
        
        # Process the image
        with Image.open(input_path) as img:
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to exact dimensions (ignoring aspect ratio)
            resized_img = img.resize((1024, 1024), Image.Resampling.LANCZOS)
            
            # Save as JPEG
            resized_img.save(output_path, 'JPEG', quality=95, optimize=True)
            
        return (True, input_path, output_path, None)
        
    except Exception as e:
        return (False, input_path, None, str(e))

def create_work_batches(jpg_files, output_dir):
    """
    Create work items for parallel processing
    """
    work_items = []
    for jpg_file in jpg_files:
        original_filename = os.path.basename(jpg_file)
        work_items.append((jpg_file, output_dir, original_filename))
    return work_items

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_directory> <output_directory>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Validate input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist")
        sys.exit(1)
    
    if not os.path.isdir(input_dir):
        print(f"Error: '{input_dir}' is not a directory")
        sys.exit(1)
    
    # Create output directory (remove if exists)
    if os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' exists. Removing...")
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")
    
    # Find all JPG files
    print("Scanning for JPG files...")
    jpg_files = find_jpg_files(input_dir)
    
    if not jpg_files:
        print("No JPG files found in the specified directory")
        return
    
    total_files = len(jpg_files)
    print(f"Found {total_files} JPG files")
    
    # Determine number of workers (use all available cores)
    num_workers = min(cpu_count(), 32)  # Cap at 32 since that's what you have
    print(f"Using {num_workers} worker processes")
    
    # Create work batches
    work_items = create_work_batches(jpg_files, output_dir)
    
    # Process files in parallel
    processed_count = 0
    failed_count = 0
    start_time = time.time()
    
    print(f"\nStarting parallel processing...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all jobs
        future_to_work = {executor.submit(process_single_image, work_item): work_item 
                         for work_item in work_items}
        
        # Process completed jobs
        for future in as_completed(future_to_work):
            success, input_path, output_path, error_msg = future.result()
            
            if success:
                processed_count += 1
                if processed_count % 100 == 0:  # Progress update every 100 files
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed
                    eta = (total_files - processed_count) / rate if rate > 0 else 0
                    print(f"Processed {processed_count}/{total_files} files "
                          f"({rate:.1f} files/sec, ETA: {eta:.0f}s)")
            else:
                failed_count += 1
                print(f"Failed to process {os.path.basename(input_path)}: {error_msg}")
    
    # Final statistics
    elapsed_time = time.time() - start_time
    avg_rate = processed_count / elapsed_time if elapsed_time > 0 else 0
    
    print(f"\nProcessing complete!")
    print(f"Total time: {elapsed_time:.1f} seconds")
    print(f"Average rate: {avg_rate:.1f} files/second")
    print(f"Successfully processed: {processed_count} files")
    print(f"Failed: {failed_count} files")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()
