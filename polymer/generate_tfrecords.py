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


def get_available_crop_sizes(image_size=1024):
    """Get available crop sizes that divide evenly into the image size."""
    available_sizes = []
    for size in [64, 128, 256, 512]:
        if image_size % size == 0:
            available_sizes.append(size)
    return available_sizes


def get_crop_positions(image_size, crop_size, overlap=0):
    """Generate crop positions for a given image and crop size.
    
    Args:
        image_size: Size of the original image (assumes square)
        crop_size: Size of the crop (assumes square)
        overlap: Overlap between crops in pixels (0 for no overlap)
    
    Returns:
        List of (row, col) positions for crop top-left corners
    """
    if overlap >= crop_size:
        raise ValueError("Overlap must be less than crop size")
    
    step = crop_size - overlap
    positions = []
    
    for row in range(0, image_size - crop_size + 1, step):
        for col in range(0, image_size - crop_size + 1, step):
            positions.append((row, col))
    
    return positions


def crop_arrays(intensity, quality, crop_size, overlap=0):
    """Generate crops from intensity and quality arrays.
    
    Args:
        intensity: 2D numpy array
        quality: 2D numpy array  
        crop_size: Size of crops to generate
        overlap: Overlap between crops in pixels
    
    Returns:
        List of dictionaries containing crop data and metadata
    """
    if intensity.shape != quality.shape:
        raise ValueError(f"Array shapes don't match: {intensity.shape} vs {quality.shape}")
    
    image_size = intensity.shape[0]  # Assuming square images
    if intensity.shape[1] != image_size:
        raise ValueError(f"Expected square images, got {intensity.shape}")
    
    positions = get_crop_positions(image_size, crop_size, overlap)
    crops = []
    
    for i, (row, col) in enumerate(positions):
        # Extract crop
        intensity_crop = intensity[row:row+crop_size, col:col+crop_size]
        quality_crop = quality[row:row+crop_size, col:col+crop_size]
        
        # Create crop metadata
        crop_data = {
            'intensity': intensity_crop,
            'quality': quality_crop,
            'crop_id': i,
            'crop_row': row,
            'crop_col': col,
            'crop_size': crop_size,
            'total_crops': len(positions)
        }
        
        crops.append(crop_data)
    
    return crops


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
                logger.debug(f"Replaced intensity no-data values ({no_data_intensity}) with NaN")
        
        # Handle quality no-data values
        if 'quality' in dataset_metadata and 'no_data_value' in dataset_metadata['quality']:
            no_data_quality = dataset_metadata['quality']['no_data_value']
            
            # Check if no-data value exists in the array
            no_data_mask = (quality == no_data_quality)
            no_data_count = np.sum(no_data_mask)
            
            if no_data_count > 0:
                # Replace with NaN
                quality = quality.copy()
                quality[no_data_mask] = np.nan 
                logger.debug(f"Replaced quality no-data values ({no_data_quality}) with NaN")
        
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


def choose_crop_settings():
    """Interactive function to choose cropping settings."""
    available_sizes = get_available_crop_sizes(1024)
    
    print(f"\nCropping options:")
    print(f"0. No cropping (keep full 1024x1024 images)")
    for i, size in enumerate(available_sizes, 1):
        crops_per_image = (1024 // size) ** 2
        print(f"{i}. {size}x{size} crops ({crops_per_image} crops per image)")
    
    while True:
        try:
            choice = input(f"Choose cropping option (0-{len(available_sizes)}): ").strip()
            choice_idx = int(choice)
            
            if choice_idx == 0:
                return None, 0  # No cropping
            elif 1 <= choice_idx <= len(available_sizes):
                crop_size = available_sizes[choice_idx - 1]
                
                # Ask about overlap
                print(f"\nOverlap options for {crop_size}x{crop_size} crops:")
                max_overlap = crop_size // 2
                overlap_options = [0, crop_size // 8, crop_size // 4, max_overlap]
                
                for i, overlap in enumerate(overlap_options):
                    step = crop_size - overlap
                    crops_per_dim = len(range(0, 1024 - crop_size + 1, step))
                    total_crops = crops_per_dim ** 2
                    print(f"  {i}. {overlap}px overlap ({total_crops} crops per image)")
                
                while True:
                    try:
                        overlap_choice = input(f"Choose overlap option (0-{len(overlap_options)-1}): ").strip()
                        overlap_idx = int(overlap_choice)
                        
                        if 0 <= overlap_idx < len(overlap_options):
                            overlap = overlap_options[overlap_idx]
                            return crop_size, overlap
                        else:
                            print("Invalid choice. Please try again.")
                    except ValueError:
                        print("Please enter a number.")
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")


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


def create_tfrecord_example(sample_data, target_mag, crop_data=None):
    """Create a single TFRecord example, optionally from cropped data."""
    
    if crop_data is not None:
        # Use cropped data
        intensity = crop_data['intensity'].astype(np.float32)
        quality = crop_data['quality'].astype(np.float32)
        crop_metadata = {
            'crop_id': crop_data['crop_id'],
            'crop_row': crop_data['crop_row'],
            'crop_col': crop_data['crop_col'],
            'crop_size': crop_data['crop_size'],
            'total_crops': crop_data['total_crops'],
            'is_crop': True
        }
    else:
        # Use full image
        intensity = sample_data['intensity'].astype(np.float32)
        quality = sample_data['quality'].astype(np.float32)
        crop_metadata = {
            'crop_id': -1,  # -1 indicates full image
            'crop_row': 0,
            'crop_col': 0,
            'crop_size': intensity.shape[0],
            'total_crops': 1,
            'is_crop': False
        }
    
    params = sample_data['params']
    
    # Validate array shapes
    if intensity.shape != quality.shape:
        raise ValueError(f"Shape mismatch: intensity {intensity.shape} vs quality {quality.shape}")
    
    # Get magnification info
    mag_info = get_magnification_info(params)
    pixel_size_nm = mag_info['pixel_size_nm']
    
    # Calculate effective pixel size and FOV for crop
    effective_pixel_size = pixel_size_nm
    crop_fov_um = (intensity.shape[0] * effective_pixel_size) / 1000
    
    # Serialize arrays
    intensity_bytes = tf.io.serialize_tensor(intensity).numpy()
    quality_bytes = tf.io.serialize_tensor(quality).numpy()
    
    # Create unique sample ID for crops
    if crop_metadata['is_crop']:
        sample_id = f"{sample_data['sample_name']}_crop_{crop_metadata['crop_id']:03d}"
    else:
        sample_id = sample_data['sample_name']
    
    # Create feature dictionary
    feature = {
        # Raw data
        'intensity': _bytes_feature(intensity_bytes),
        'quality': _bytes_feature(quality_bytes),
        
        # Array metadata
        'height': _int64_feature(intensity.shape[0]),
        'width': _int64_feature(intensity.shape[1]),
        
        # Physical metadata
        'pixel_size_nm': _float_feature(effective_pixel_size),
        'magnification': _float_feature(target_mag),
        'field_of_view_um': _float_feature(crop_fov_um),
        
        # Crop metadata
        'is_crop': _int64_feature(1 if crop_metadata['is_crop'] else 0),
        'crop_id': _int64_feature(crop_metadata['crop_id']),
        'crop_row': _int64_feature(crop_metadata['crop_row']),
        'crop_col': _int64_feature(crop_metadata['crop_col']),
        'crop_size': _int64_feature(crop_metadata['crop_size']),
        'total_crops_from_source': _int64_feature(crop_metadata['total_crops']),
        
        # Instrument metadata
        'light_level_pct': _float_feature(params.get('measurement', {}).get('light_level_pct', 0.0)),
        'numerical_aperture': _float_feature(params.get('optics', {}).get('numerical_aperture', 0.0)),
        'wavelength_nm': _float_feature(params.get('optics', {}).get('wavelength_nm', 0.0)),
        'frame_count': _int64_feature(params.get('camera', {}).get('frame_count', 1)),
        
        # Sample identification
        'sample_name': _bytes_feature(sample_data['sample_name'].encode('utf-8')),
        'sample_id': _bytes_feature(sample_id.encode('utf-8')),
    }
    
    return tf.train.Example(features=tf.train.Features(feature=feature))


def process_shard(args):
    """Process a single shard - for parallel processing."""
    dataset_folder, sample_names, target_mag, shard_path, crop_size, overlap = args
    
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
                    
                    if crop_size is None:
                        # No cropping - write full image
                        example = create_tfrecord_example(sample_data, target_mag, None)
                        writer.write(example.SerializeToString())
                        successful += 1
                    else:
                        # Generate crops
                        crops = crop_arrays(
                            sample_data['intensity'], 
                            sample_data['quality'], 
                            crop_size, 
                            overlap
                        )
                        
                        # Write each crop as a separate example
                        for crop_data in crops:
                            example = create_tfrecord_example(sample_data, target_mag, crop_data)
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
    """Create TFRecord dataset from processed DATX data with parallel processing and cropping options."""
    
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
    
    # Choose cropping settings
    crop_size, overlap = choose_crop_settings()
    
    if crop_size is None:
        logger.info("Processing full 1024x1024 images")
        expected_samples_per_input = 1
    else:
        crops_per_dim = len(range(0, 1024 - crop_size + 1, crop_size - overlap))
        expected_samples_per_input = crops_per_dim ** 2
        logger.info(f"Processing {crop_size}x{crop_size} crops with {overlap}px overlap")
        logger.info(f"Will generate {expected_samples_per_input} crops per input image")
    
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
    
    total_expected_outputs = len(filtered_samples) * expected_samples_per_input
    
    logger.info(f"Keeping {len(filtered_samples)}/{len(sample_names)} input samples "
                f"({len(filtered_samples)/len(sample_names)*100:.1f}%)")
    logger.info(f"Expected total output samples: {total_expected_outputs}")
    
    if len(filtered_samples) == 0:
        logger.error("No samples found with target magnification!")
        return
    
    # Confirm strategy
    if crop_size is None:
        crop_info = "full 1024x1024 images"
    else:
        crop_info = f"{crop_size}x{crop_size} crops with {overlap}px overlap ({expected_samples_per_input} per image)"
    
    response = input(f"\nProceed with {len(filtered_samples)} input samples at {target_mag:.1f}X "
                    f"generating {crop_info}? (y/n): ")
    if response.lower() != 'y':
        logger.info("Operation aborted by user")
        return
    
    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Adjust samples per shard based on cropping
    if crop_size is not None:
        # Adjust samples_per_shard to account for multiple crops per input
        adjusted_samples_per_shard = max(1, samples_per_shard // expected_samples_per_input)
    else:
        adjusted_samples_per_shard = samples_per_shard
    
    # Calculate number of shards
    num_shards = (len(filtered_samples) + adjusted_samples_per_shard - 1) // adjusted_samples_per_shard
    
    logger.info(f"Creating {num_shards} TFRecord shards with ~{adjusted_samples_per_shard} input samples each")
    logger.info(f"Using {n_cores} parallel processes for shard creation")
    
    # Prepare shard arguments
    shard_args = []
    for shard_idx in range(num_shards):
        start_idx = shard_idx * adjusted_samples_per_shard
        end_idx = min(start_idx + adjusted_samples_per_shard, len(filtered_samples))
        shard_samples = filtered_samples[start_idx:end_idx]
        
        shard_filename = f"{output_path.stem}_{shard_idx:04d}_of_{num_shards:04d}.tfrecord"
        shard_path = output_path.parent / shard_filename
        
        shard_args.append((dataset_folder, shard_samples, target_mag, shard_path, crop_size, overlap))
    
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
        'total_output_samples': total_successful,
        'failed_samples': total_failed,
        'input_samples_processed': len(filtered_samples),
        'target_magnification': target_mag,
        'filtered_from_total': len(sample_names),
        'magnifications_available': {str(k): v for k, v in mag_info.items()},  # JSON serializable
        'samples_per_shard': adjusted_samples_per_shard,
        'num_shards': num_shards,
        'pixel_size_nm': mag_info[target_mag]['pixel_size_nm'],
        'field_of_view_um': mag_info[target_mag]['field_of_view_um'],
        'cropping': {
            'enabled': crop_size is not None,
            'crop_size': crop_size,
            'overlap': overlap if crop_size is not None else 0,
            'crops_per_input': expected_samples_per_input,
            'original_image_size': 1024
        },
        'shard_files': [f"{output_path.stem}_{i:04d}_of_{num_shards:04d}.tfrecord" for i in range(num_shards)]
    }
    
    metadata_file = output_path.parent / f"{output_path.stem}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Dataset creation complete!")
    print(f"Input samples processed: {len(filtered_samples)}")
    print(f"Total output samples: {total_successful}")
    print(f"Failed samples: {total_failed}")
    print(f"Magnification: {target_mag:.1f}X ({mag_info[target_mag]['pixel_size_nm']:.1f} nm/pixel)")
    if crop_size is not None:
        print(f"Crop settings: {crop_size}x{crop_size} with {overlap}px overlap")
        print(f"Average crops per input: {total_successful / len(filtered_samples):.1f}")
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
