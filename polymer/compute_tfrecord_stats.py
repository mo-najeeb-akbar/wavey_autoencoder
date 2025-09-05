#!/usr/bin/env python3
"""
Script to compute global statistics for DATX TFRecord dataset
Saves statistics to JSON for use in dataloader normalization
"""

import tensorflow as tf
import numpy as np
import json
import glob
import argparse
from pathlib import Path
from tqdm import tqdm

def parse_tfrecord_example(example_proto):
    """Parse a single TFRecord example - same as your dataloader"""
    feature_description = {
        'intensity': tf.io.FixedLenFeature([], tf.string),
        'quality': tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'pixel_size_nm': tf.io.FixedLenFeature([], tf.float32),
        'magnification': tf.io.FixedLenFeature([], tf.float32),
        'light_level_pct': tf.io.FixedLenFeature([], tf.float32),
        'numerical_aperture': tf.io.FixedLenFeature([], tf.float32),
    }
    
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    # Deserialize intensity array
    intensity = tf.io.parse_tensor(parsed['intensity'], out_type=tf.float32)
    intensity = tf.reshape(intensity, [parsed['height'], parsed['width']])
    
    return intensity

def compute_global_statistics(tfrecord_pattern, max_samples=None, output_file='global_stats.json'):
    """
    Compute global statistics across all TFRecord files
    
    Args:
        tfrecord_pattern: Glob pattern for TFRecord files
        max_samples: Maximum number of images to process (None = all)
        output_file: Output JSON file path
    """
    
    # Find TFRecord files
    tfrecord_files = sorted(glob.glob(tfrecord_pattern))
    if not tfrecord_files:
        raise ValueError(f"No TFRecord files found matching: {tfrecord_pattern}")
    
    print(f"Found {len(tfrecord_files)} TFRecord files")
    
    # Create dataset
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parse_tfrecord_example, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Limit samples if specified
    if max_samples:
        dataset = dataset.take(max_samples)
        print(f"Processing max {max_samples} samples")
    
    # Initialize accumulators
    count = 0
    sum_intensity = 0.0
    sum_squared = 0.0
    min_intensity = float('inf')
    max_intensity = float('-inf')
    
    # For percentile computation - collect samples
    intensity_samples = []
    sample_every_n = max(1, (max_samples or 1000) // 100)  # Sample ~100 images for percentiles
    
    print("Computing statistics...")
    for i, intensity in enumerate(tqdm(dataset)):
        # Convert to numpy for easier computation
        intensity_np = intensity.numpy()
        
        # Remove NaNs
        valid_mask = ~np.isnan(intensity_np)
        if not np.any(valid_mask):
            continue  # Skip images with all NaNs
            
        valid_intensity = intensity_np[valid_mask]
        
        # Update statistics
        count += valid_intensity.size
        sum_intensity += np.sum(valid_intensity)
        sum_squared += np.sum(valid_intensity ** 2)
        min_intensity = min(min_intensity, np.min(valid_intensity))
        max_intensity = max(max_intensity, np.max(valid_intensity))
        
        # Collect samples for percentile computation
        if i % sample_every_n == 0:
            # Subsample the image to avoid memory issues
            subsample = valid_intensity[::10]  # Every 10th pixel
            intensity_samples.extend(subsample.tolist())
    
    if count == 0:
        raise ValueError("No valid (non-NaN) data found!")
    
    # Compute final statistics
    mean = sum_intensity / count
    variance = (sum_squared / count) - (mean ** 2)
    std = np.sqrt(max(variance, 0))
    
    # Compute percentiles
    print("Computing percentiles...")
    intensity_samples = np.array(intensity_samples)
    percentiles = {
        'p1': np.percentile(intensity_samples, 1),
        'p2': np.percentile(intensity_samples, 2),
        'p5': np.percentile(intensity_samples, 5),
        'p95': np.percentile(intensity_samples, 95),
        'p98': np.percentile(intensity_samples, 98),
        'p99': np.percentile(intensity_samples, 99),
    }
    
    # Create statistics dictionary
    stats = {
        'count': int(count),
        'mean': float(mean),
        'std': float(std),
        'min': float(min_intensity),
        'max': float(max_intensity),
        'percentiles': {k: float(v) for k, v in percentiles.items()},
        'computed_from': tfrecord_pattern,
        'max_samples_used': max_samples,
        'total_files': len(tfrecord_files)
    }
    
    # Print summary
    print(f"\nGlobal Statistics Summary:")
    print(f"Total valid pixels: {count:,}")
    print(f"Mean intensity: {mean:.2f}")
    print(f"Std intensity: {std:.2f}")
    print(f"Min intensity: {min_intensity:.2f}")
    print(f"Max intensity: {max_intensity:.2f}")
    print(f"P2-P98 range: {percentiles['p2']:.2f} - {percentiles['p98']:.2f}")
    
    # Save to JSON
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nStatistics saved to: {output_path}")
    
    # Suggest normalization strategies
    print(f"\nRecommended normalization strategies:")
    print(f"1. Global min-max: [{min_intensity:.1f}, {max_intensity:.1f}] -> [0, 1]")
    print(f"2. Percentile clipping: [{percentiles['p2']:.1f}, {percentiles['p98']:.1f}] -> [0, 1]")
    print(f"3. Z-score: (x - {mean:.1f}) / {std:.1f}")
    
    return stats

def load_global_stats(stats_file='global_stats.json'):
    """Load precomputed global statistics"""
    with open(stats_file, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute global statistics for TFRecord dataset')
    parser.add_argument('tfrecord_pattern', help='Glob pattern for TFRecord files')
    parser.add_argument('--max_samples', type=int, default=None, 
                       help='Maximum number of images to process (default: all)')
    parser.add_argument('--output', '-o', default='global_stats.json',
                       help='Output JSON file (default: global_stats.json)')
    
    args = parser.parse_args()
    
    # Compute statistics
    stats = compute_global_statistics(
        tfrecord_pattern=args.tfrecord_pattern,
        max_samples=args.max_samples,
        output_file=args.output
    )
    
    print("\nTo use these stats in your dataloader:")
    print(f"stats = load_global_stats('{args.output}')")
    print("# Then use stats['mean'], stats['std'], stats['percentiles']['p2'], etc.")

# Example usage:
# python compute_global_stats.py "/path/to/your/*.tfrecord" --max_samples 1000
