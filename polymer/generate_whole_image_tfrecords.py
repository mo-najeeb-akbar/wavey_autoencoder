import tensorflow as tf
import numpy as np
import json
from pathlib import Path
import sys
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import gc
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_list_feature(value):
    """Returns a float_list from a list of floats."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def load_dataset_index(dataset_folder):
    """Load the dataset index."""
    index_file = Path(dataset_folder) / 'dataset_index.json'
    if not index_file.exists():
        raise FileNotFoundError(f"No dataset_index.json found in {dataset_folder}")
    
    try:
        with open(index_file, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in dataset_index.json: {e}")


def load_sample_data(dataset_folder, sample_name):
    """Load intensity, quality, and metadata for a sample."""
    sample_dir = Path(dataset_folder) / sample_name
    
    # Define required files
    intensity_file = sample_dir / 'intensity.npy'
    quality_file = sample_dir / 'quality.npy'
    params_file = sample_dir / 'instrument_params.json'
    metadata_file = sample_dir / 'dataset_metadata.json'
    
    # Check if all files exist
    required_files = [intensity_file, quality_file, params_file]
    if not all(f.exists() for f in required_files):
        missing_files = [f.name for f in required_files if not f.exists()]
        logger.warning(f"Missing files for {sample_name}: {missing_files}")
        return None
    
    try:
        intensity = np.load(intensity_file)
        quality = np.load(quality_file)
        
        with open(params_file, 'r') as f:
            params = json.load(f)
        
        # Load dataset metadata if available (contains no_data_value info)
        dataset_metadata = None
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                dataset_metadata = json.load(f)
        
        # Handle no-data values
        intensity, quality = handle_no_data_values(intensity, quality, dataset_metadata, sample_name)
        
        return {
            'intensity': intensity,
            'quality': quality,
            'params': params,
            'sample_name': sample_name,
            'dataset_metadata': dataset_metadata
        }
    except Exception as e:
        logger.error(f"Error loading sample {sample_name}: {e}")
        return None


def handle_no_data_values(intensity, quality, dataset_metadata, sample_name):
    """Handle no-data values in intensity and quality arrays."""
    if dataset_metadata is None:
        logger.debug(f"No dataset metadata for {sample_name}, skipping no-data value handling")
        return intensity, quality
    
    try:
        # Handle intensity no-data values
        if 'intensity' in dataset_metadata and 'no_data_value' in dataset_metadata['intensity']:
            no_data_intensity = dataset_metadata['intensity']['no_data_value']
            
            # Check if no-data value exists in the array
            no_data_mask = (intensity == no_data_intensity)
            no_data_count = np.sum(no_data_mask)
            
            if no_data_count > 0:
                intensity = intensity.copy()
                intensity[no_data_mask] = np.nan
                logger.debug(f"Replaced intensity no-data values ({no_data_intensity}) with 0")
        
        # Handle quality no-data values
        if 'quality' in dataset_metadata and 'no_data_value' in dataset_metadata['quality']:
            no_data_quality = dataset_metadata['quality']['no_data_value']
            
            # Check if no-data value exists in the array
            no_data_mask = (quality == no_data_quality)
            no_data_count = np.sum(no_data_mask)
            
            if no_data_count > 0:
                # Replace with zeros (or could use median/mean imputation)
                quality = quality.copy()
                quality[no_data_mask] = np.nan 
                logger.debug(f"Replaced quality no-data values ({no_data_quality}) with 0.0")
        
    except Exception as e:
        logger.error(f"Error handling no-data values for {sample_name}: {e}")
        # Return original arrays if processing fails
        return intensity, quality
    
    return intensity, quality


def get_magnification_info(params):
    """Extract magnification and pixel resolution info."""
    try:
        mag = params['optics']['magnification']
        pixel_size_nm = params['resolution']['lateral_nm']
        
        # Calculate field of view (assuming square images)
        # Get actual image dimensions from the first available sample
        fov_um = (1024 * pixel_size_nm) / 1000  # Default assumption
        
        return {
            'magnification': mag,
            'pixel_size_nm': pixel_size_nm,
            'field_of_view_um': fov_um
        }
    except KeyError as e:
        raise ValueError(f"Missing required parameter in instrument_params.json: {e}")


def check_sample_magnification(args):
    """Check magnification for a single sample - for parallel processing."""
    dataset_folder, sample_name = args
    
    try:
        sample_data = load_sample_data(dataset_folder, sample_name)
        if sample_data:
            mag_info = get_magnification_info(sample_data['params'])
            return (sample_name, mag_info['magnification'])
        return None
    except Exception as e:
        logger.debug(f"Error checking magnification for {sample_name}: {e}")
        return None


def analyze_magnifications(dataset_folder, index):
    """Analyze magnification distribution across dataset using parallel processing."""
    logger.info("Analyzing magnifications across dataset...")
    
    sample_names = list(index['files'].keys())
    
    # Use reasonable number of cores
    n_cores = min(cpu_count(), 8)
    args_list = [(dataset_folder, name) for name in sample_names]
    
    with Pool(n_cores) as pool:
        results = list(tqdm(
            pool.imap(check_sample_magnification, args_list), 
            total=len(args_list),
            desc="Checking magnifications"
        ))
    
    # Process results
    mag_counts = defaultdict(int)
    mag_info = {}
    
    for result in results:
        if result is not None:
            sample_name, mag = result
            mag_counts[mag] += 1
            
            # Get detailed info for first sample of each magnification
            if mag not in mag_info:
                sample_data = load_sample_data(dataset_folder, sample_name)
                if sample_data:
                    mag_info[mag] = get_magnification_info(sample_data['params'])
    
    print("\nMagnification distribution:")
    for mag, count in sorted(mag_counts.items()):
        if mag in mag_info:
            pixel_size = mag_info[mag]['pixel_size_nm']
            fov = mag_info[mag]['field_of_view_um']
            print(f"  {mag:.1f}X: {count} samples, {pixel_size:.1f} nm/pixel, {fov:.1f} μm FOV")
    
    return mag_info, mag_counts


def decide_resize_strategy(mag_info, mag_counts):
    """Decide how to handle different magnifications."""
    mags = sorted(mag_info.keys())
    
    if len(mags) == 1:
        logger.info(f"All samples at {mags[0]:.1f}X - keeping at full resolution")
        return mags[0]
    
    print(f"\nFound {len(mags)} different magnifications:")
    for mag in mags:
        pixel_size = mag_info[mag]['pixel_size_nm']
        fov = mag_info[mag]['field_of_view_um']
        count = mag_counts[mag]
        print(f"  {mag:.1f}X: {count} samples, {pixel_size:.1f} nm/pixel, {fov:.1f} μm FOV")
    
    print(f"\nFor defect detection, keeping only 20X samples at full resolution")
    print(f"Filtering out other magnifications to maintain consistent resolution")
    
    # Keep only 20X samples
    target_mag = 20.0
    if target_mag not in mags:
        # Find closest to 20X
        target_mag = min(mags, key=lambda x: abs(x - 20.0))
        print(f"No exact 20X found, using closest: {target_mag:.1f}X")
    
    return target_mag


def should_include_sample(params, target_magnification):
    """Check if sample should be included based on magnification."""
    mag = params['optics']['magnification']
    return abs(mag - target_magnification) < 0.1  # Small tolerance for floating point


def filter_samples_by_magnification(args):
    """Filter a batch of samples by magnification - for parallel processing."""
    dataset_folder, sample_names, target_mag = args
    
    filtered_samples = []
    for sample_name in sample_names:
        try:
            sample_data = load_sample_data(dataset_folder, sample_name)
            if sample_data and should_include_sample(sample_data['params'], target_mag):
                filtered_samples.append(sample_name)
        except Exception as e:
            logger.debug(f"Error filtering sample {sample_name}: {e}")
            continue
    
    return filtered_samples


def create_tfrecord_example(sample_data, target_mag):
    """Create a single TFRecord example."""
    intensity = sample_data['intensity'].astype(np.float32)
    quality = sample_data['quality'].astype(np.float32)
    params = sample_data['params']
    
    # Validate array shapes
    if intensity.shape != quality.shape:
        raise ValueError(f"Shape mismatch: intensity {intensity.shape} vs quality {quality.shape}")
    
    # Get magnification info
    mag_info = get_magnification_info(params)
    pixel_size_nm = mag_info['pixel_size_nm']
    
    # Serialize arrays at full resolution
    intensity_bytes = tf.io.serialize_tensor(intensity).numpy()
    quality_bytes = tf.io.serialize_tensor(quality).numpy()
    
    # Create feature dictionary
    feature = {
        # Raw data at full resolution
        'intensity': _bytes_feature(intensity_bytes),
        'quality': _bytes_feature(quality_bytes),
        
        # Array metadata
        'height': _int64_feature(intensity.shape[0]),
        'width': _int64_feature(intensity.shape[1]),
        
        # Physical metadata
        'pixel_size_nm': _float_feature(pixel_size_nm),
        'magnification': _float_feature(target_mag),
        'field_of_view_um': _float_feature((intensity.shape[0] * pixel_size_nm) / 1000),
        
        # Instrument metadata
        'light_level_pct': _float_feature(params.get('measurement', {}).get('light_level_pct', 0.0)),
        'numerical_aperture': _float_feature(params.get('optics', {}).get('numerical_aperture', 0.0)),
        'wavelength_nm': _float_feature(params.get('optics', {}).get('wavelength_nm', 0.0)),
        'frame_count': _int64_feature(params.get('camera', {}).get('frame_count', 1)),
        
        # Sample identification
        'sample_name': _bytes_feature(sample_data['sample_name'].encode('utf-8')),
    }
    
    return tf.train.Example(features=tf.train.Features(feature=feature))


def process_shard(args):
    """Process a single shard - for parallel processing."""
    dataset_folder, sample_names, target_mag, shard_path = args
    
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
                    
                    # Create TFRecord example
                    example = create_tfrecord_example(sample_data, target_mag)
                    
                    # Write to TFRecord
                    writer.write(example.SerializeToString())
                    successful += 1
                    
                    # Clean up memory
                    del sample_data
                    
                except Exception as e:
                    logger.error(f"Error processing sample {sample_name}: {e}")
                    failed += 1
                    continue
    
    except Exception as e:
        logger.error(f"Error creating shard {shard_path}: {e}")
        return 0, len(sample_names), str(shard_path)
    
    # Force garbage collection
    gc.collect()
    
    return successful, failed, str(shard_path)


def create_tfrecord_dataset(dataset_folder, output_file, samples_per_shard=1000):
    """Create TFRecord dataset from processed DATX data with parallel processing."""
    
    # Load index
    try:
        index = load_dataset_index(dataset_folder)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load dataset index: {e}")
        return
    
    sample_names = list(index['files'].keys())
    logger.info(f"Found {len(sample_names)} total samples")
    
    # Analyze magnifications with parallel processing
    mag_info, mag_counts = analyze_magnifications(dataset_folder, index)
    target_mag = decide_resize_strategy(mag_info, mag_counts)
    
    # Filter samples by magnification with parallel processing
    logger.info(f"Filtering samples to keep only {target_mag:.1f}X...")
    
    # Split samples into batches for parallel filtering
    n_cores = min(cpu_count(), 8)
    batch_size = max(1, len(sample_names) // n_cores)
    sample_batches = [sample_names[i:i + batch_size] for i in range(0, len(sample_names), batch_size)]
    
    filter_args = [(dataset_folder, batch, target_mag) for batch in sample_batches]
    
    with Pool(n_cores) as pool:
        filter_results = pool.map(filter_samples_by_magnification, filter_args)
    
    # Combine filtered results
    filtered_samples = []
    for batch_result in filter_results:
        filtered_samples.extend(batch_result)
    
    logger.info(f"Keeping {len(filtered_samples)}/{len(sample_names)} samples "
                f"({len(filtered_samples)/len(sample_names)*100:.1f}%)")
    
    if len(filtered_samples) == 0:
        logger.error("No samples found with target magnification!")
        return
    
    # Confirm strategy
    response = input(f"\nProceed with {len(filtered_samples)} samples at {target_mag:.1f}X? (y/n): ")
    if response.lower() != 'y':
        logger.info("Operation aborted by user")
        return
    
    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate number of shards
    num_shards = (len(filtered_samples) + samples_per_shard - 1) // samples_per_shard
    
    logger.info(f"Creating {num_shards} TFRecord shards with ~{samples_per_shard} samples each")
    logger.info(f"Using {n_cores} parallel processes for shard creation")
    
    # Prepare shard arguments
    shard_args = []
    for shard_idx in range(num_shards):
        start_idx = shard_idx * samples_per_shard
        end_idx = min(start_idx + samples_per_shard, len(filtered_samples))
        shard_samples = filtered_samples[start_idx:end_idx]
        
        shard_filename = f"{output_path.stem}_{shard_idx:04d}_of_{num_shards:04d}.tfrecord"
        shard_path = output_path.parent / shard_filename
        
        shard_args.append((dataset_folder, shard_samples, target_mag, shard_path))
    
    # Process shards in parallel
    logger.info("Processing shards in parallel...")
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
    
    # Save dataset metadata
    metadata = {
        'total_samples': total_successful,
        'failed_samples': total_failed,
        'target_magnification': target_mag,
        'filtered_from_total': len(sample_names),
        'magnifications_available': {str(k): v for k, v in mag_info.items()},  # JSON serializable
        'samples_per_shard': samples_per_shard,
        'num_shards': num_shards,
        'pixel_size_nm': mag_info[target_mag]['pixel_size_nm'],
        'field_of_view_um': mag_info[target_mag]['field_of_view_um'],
        'shard_files': [f"{output_path.stem}_{i:04d}_of_{num_shards:04d}.tfrecord" for i in range(num_shards)]
    }
    
    metadata_file = output_path.parent / f"{output_path.stem}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Dataset creation complete!")
    print(f"Successful samples: {total_successful}")
    print(f"Failed samples: {total_failed}")
    print(f"Magnification: {target_mag:.1f}X ({mag_info[target_mag]['pixel_size_nm']:.1f} nm/pixel)")
    print(f"Created {num_shards} shards in: {output_path.parent}")
    print(f"Metadata saved to: {metadata_file}")


def main():
    if len(sys.argv) not in [3, 4]:
        print("Usage: python generate_tfrecords.py <dataset_folder> <output_tfrecord_path> [samples_per_shard]")
        print("Example: python generate_tfrecords.py ./processed_dataset ./tfrecords/dataset.tfrecord 500")
        sys.exit(1)
    
    dataset_folder = Path(sys.argv[1])
    output_file = sys.argv[2]
    samples_per_shard = int(sys.argv[3]) if len(sys.argv) == 4 else 1000
    
    if not dataset_folder.exists():
        logger.error(f"Dataset folder not found: {dataset_folder}")
        sys.exit(1)
    
    logger.info(f"Creating TFRecord dataset with {samples_per_shard} samples per shard")
    create_tfrecord_dataset(dataset_folder, output_file, samples_per_shard)


if __name__ == "__main__":
    main()
