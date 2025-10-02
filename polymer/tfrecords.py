import os
import sys
from pathlib import Path
import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import gc


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def load_sample_data(dataset_folder, sample_name):
    """Load intensity, quality, and metadata for a sample."""
    sample_dir = Path(dataset_folder) / sample_name
    
    # Define required files
    intensity_file = sample_dir / 'intensity.npy'
    quality_file = sample_dir / 'quality.npy'
    params_file = sample_dir / 'instrument_params.json'
    metadata_file = sample_dir / 'dataset_metadata.json'
    
    intensity = np.load(intensity_file)
    quality = np.load(quality_file)
    
    with open(params_file, 'r') as f:
        params = json.load(f)
    
    dataset_metadata = None
    with open(metadata_file, 'r') as f:
        dataset_metadata = json.load(f)
    
    # Handle no-data values
    intensity, quality = handle_no_data_values(intensity, quality, dataset_metadata)

    return {
        'intensity': intensity,
        'quality': quality,
        'params': params,
        'sample_name': sample_name,
        'dataset_metadata': dataset_metadata
    }


def handle_no_data_values(intensity, quality, dataset_metadata):
    # Handle intensity no-data values
    no_data_intensity = dataset_metadata['intensity']['no_data_value']
    
    # Check if no-data value exists in the array
    no_data_mask = (intensity == no_data_intensity)
    no_data_count = np.sum(no_data_mask)
    
    if no_data_count > 0:
        intensity = intensity.copy()
        intensity[no_data_mask] = np.nan

    # Handle quality no-data values
    no_data_quality = dataset_metadata['quality']['no_data_value']
    
    # Check if no-data value exists in the array
    no_data_mask = (quality == no_data_quality)
    no_data_count = np.sum(no_data_mask)
    
    if no_data_count > 0:
        # Replace with NaN
        quality = quality.copy()
        quality[no_data_mask] = np.nan 
        
    return intensity, quality


def create_tfrecord_example(sample_data):
    """Create a single TFRecord example, optionally from cropped data."""

    intensity = sample_data['intensity'].astype(np.float32)
    quality = sample_data['quality'].astype(np.float32)

    params = sample_data['params']

    # Validate array shapes
    if intensity.shape != quality.shape:
        raise ValueError(f"Shape mismatch: intensity {intensity.shape} vs quality {quality.shape}")

    # Serialize arrays
    intensity_bytes = tf.io.serialize_tensor(intensity).numpy()
    quality_bytes = tf.io.serialize_tensor(quality).numpy()

    # Create feature dictionary
    feature = {
        # Raw data
        'intensity': _bytes_feature(intensity_bytes),
        'quality': _bytes_feature(quality_bytes),

        # Array metadata
        'height': _int64_feature(intensity.shape[0]),
        'width': _int64_feature(intensity.shape[1]),

        # Instrument metadata
        'light_level_pct': _float_feature(params.get('measurement', {}).get('light_level_pct', 0.0)),
        'numerical_aperture': _float_feature(params.get('optics', {}).get('numerical_aperture', 0.0)),
        'wavelength_nm': _float_feature(params.get('optics', {}).get('wavelength_nm', 0.0)),

        # Sample identification
        'sample_name': _bytes_feature(sample_data['sample_name'].encode('utf-8')),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def process_shard(args):
    """Process a single shard - for parallel processing."""
    dataset_folder, sample_names, shard_path = args

    successful = 0
    failed = 0

    try:
        with tf.io.TFRecordWriter(str(shard_path)) as writer:
            for sample_name in sample_names:
                try:
                    # Load sample data
                    sample_data = load_sample_data(dataset_folder, sample_name)
                    if sample_data is None:
                        failed += 1
                        continue

                    example = create_tfrecord_example(sample_data)
                    writer.write(example.SerializeToString())
                    successful += 1
                    del sample_data

                except Exception as e:
                    failed += 1
                    continue

    except Exception as e:
        return 0, len(sample_names), str(shard_path)

    # Force garbage collection
    gc.collect()

    return successful, failed, str(shard_path)


def create_tfrecord_dataset(dataset_folder, sample_names, output_file, samples_per_shard=1000):
    """Create TFRecord dataset from processed DATX data with parallel processing and cropping options."""
    
    # Split samples into batches for parallel filtering
    n_cores = min(cpu_count(), 8)
    batch_size = max(1, len(sample_names) // n_cores)
    sample_batches = [sample_names[i:i + batch_size] for i in range(0, len(sample_names), batch_size)]

    filter_args = [(dataset_folder, batch) for batch in sample_batches]

    total_expected_outputs = len(sample_names) 

    # Create output directory
    output_path = Path(os.path.join(output_file, 'record'))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate number of shards
    num_shards = (len(sample_names) + samples_per_shard - 1) // samples_per_shard

    # Prepare shard arguments
    shard_args = []
    for shard_idx in range(num_shards):
        start_idx = shard_idx * samples_per_shard
        end_idx = min(start_idx + samples_per_shard, len(sample_names))
        shard_samples = sample_names[start_idx:end_idx]

        shard_filename = f"{output_path.stem}_{shard_idx:04d}_of_{num_shards:04d}.tfrecord"
        shard_path = output_path.parent / shard_filename

        shard_args.append((dataset_folder, shard_samples, shard_path))

    # Process shards in parallel
    with Pool(n_cores) as pool:
        shard_results = list(tqdm(
            pool.imap(process_shard, shard_args),
            total=len(shard_args),
            desc="Creating shards"
        ))

    # Collect results
    total_successful = sum(r[0] for r in shard_results)
    total_failed = sum(r[1] for r in shard_results)

    print(f"\nShard creation complete:")
    for i, (successful, failed, shard_path) in enumerate(shard_results):
        print(f"  Shard {i+1}: {successful} successful, {failed} failed")

    print(f"\nDataset creation complete!")
    print(f"Input samples processed: {len(sample_names)}")
    print(f"Total output samples: {total_successful}")
    print(f"Failed samples: {total_failed}")
    print(f"Created {num_shards} shards in: {output_path.parent}")
