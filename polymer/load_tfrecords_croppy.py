import tensorflow as tf
import numpy as np
import jax.numpy as jnp
import jax
from functools import partial
import glob

def data_to_jax(tensor_dict, dtype=jnp.float32):
    """Convert TensorFlow tensors to JAX arrays"""
    def convert_leaf(x):
        # Convert to numpy first to handle all tensor types including strings
        x_np = x.numpy() if hasattr(x, 'numpy') else x
        # For numeric types, apply dtype conversion
        if np.issubdtype(x_np.dtype, np.number):
            return jnp.array(x_np, dtype=dtype)
        # For strings and other types, keep as numpy
        return x_np
    
    return jax.tree.map(convert_leaf, tensor_dict)

def create_input_iter(tfds, dtype=jnp.float32):
    """Create JAX input iterator"""
    d_t_j = partial(data_to_jax, dtype=dtype)
    return map(d_t_j, tfds)

def parse_example(example_proto):
    """Parse TFRecord example"""
    feature_description = {
        'intensity': tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'sample_name': tf.io.FixedLenFeature([], tf.string),
    }
    
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    # Decode the image
    height = tf.cast(parsed['height'], tf.int32)
    width = tf.cast(parsed['width'], tf.int32)
    intensity = tf.io.parse_tensor(parsed['intensity'], out_type=tf.float32)
    intensity = tf.reshape(intensity, [height, width, 1])
    
    # Handle NaNs
    intensity = tf.where(tf.math.is_nan(intensity), 0.0, intensity)
    
    # Decode sample name
    sample_name = parsed['sample_name']
    
    return {'image': intensity, 'sample_name': sample_name}

def create_overlapping_crops(data_dict, crop_size=256, stride=192, 
                           global_mean=None, global_std=None):
    """
    Create overlapping crops from a single image using extract_patches.
    
    Args:
        data_dict: Dict with 'image' and 'sample_name'
        crop_size: Size of each crop (e.g., 256)
        stride: Step size between crops (e.g., 192 gives 64px overlap)
        global_mean: Global mean for standardization (optional)
        global_std: Global std for standardization (optional)
    
    Returns:
        Dict with crops and metadata
    """
    image = data_dict['image']
    sample_name = data_dict['sample_name']
    
    # Standardize if stats provided
    if global_mean is not None and global_std is not None:
        image = (image - global_mean) / (global_std + 1e-8)
    
    # Add batch dimension: [1, H, W, C]
    image = tf.expand_dims(image, 0)
    
    # Extract patches
    patches = tf.image.extract_patches(
        images=image,
        sizes=[1, crop_size, crop_size, 1],
        strides=[1, stride, stride, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    
    # patches shape: [1, num_rows, num_cols, crop_size*crop_size*channels]
    patches_shape = tf.shape(patches)
    num_rows = patches_shape[1]
    num_cols = patches_shape[2]
    num_channels = tf.shape(image)[3]
    num_crops = num_rows * num_cols
    
    # Reshape to [num_rows*num_cols, crop_size, crop_size, channels]
    patches = tf.reshape(patches, [-1, crop_size, crop_size, num_channels])
    
    # Create position indices for each crop
    row_indices = tf.repeat(tf.range(num_rows), num_cols)
    col_indices = tf.tile(tf.range(num_cols), [num_rows])
    
    # Tile sample name for each crop
    sample_names = tf.tile([sample_name], [num_crops])
    
    return {
        'image': patches,
        'sample_name': sample_names,
        'row_idx': row_indices,
        'col_idx': col_indices,
        'num_rows': tf.fill([num_crops], num_rows),
        'num_cols': tf.fill([num_crops], num_cols),
    }

def augment_crop(data_dict, flip_prob=0.5, rotate_prob=0.5, 
                brightness_delta=0.1, contrast_range=(0.9, 1.1),
                noise_std=0.02):
    """
    Apply random augmentations to a crop using stateless random ops
    
    Args:
        data_dict: Dict with 'image' and metadata
        flip_prob: Probability of flipping
        rotate_prob: Probability of rotation
        brightness_delta: Max brightness change
        contrast_range: (min, max) contrast multiplier
        noise_std: Standard deviation of Gaussian noise
    
    Returns:
        Augmented data dict
    """
    image = data_dict['image']
    
    # Use stateless random ops that work with tf.data
    # Random horizontal flip
    if tf.random.uniform(()) < flip_prob:
        image = tf.image.flip_left_right(image)
    
    # Random vertical flip
    if tf.random.uniform(()) < flip_prob:
        image = tf.image.flip_up_down(image)
    
    # Random rotation (0, 90, 180, 270 degrees)
    if tf.random.uniform(()) < rotate_prob:
        k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k=k)
    
    # Random brightness adjustment
    brightness_factor = tf.random.uniform((), minval=-brightness_delta, maxval=brightness_delta)
    image = image + brightness_factor
    
    # Random contrast adjustment
    contrast_factor = tf.random.uniform((), minval=contrast_range[0], maxval=contrast_range[1])
    mean = tf.reduce_mean(image)
    image = (image - mean) * contrast_factor + mean
    
    # Add Gaussian noise
    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=noise_std)
    image = image + noise
    
    data_dict['image'] = image
    return data_dict

def build_dataset(tfrecord_pattern, crop_size=256, stride=192, batch_size=32, 
                  shuffle=True, seed=None, augment=False,
                  global_mean=None, global_std=None,
                  flip_prob=0.5, rotate_prob=0.5,
                  brightness_delta=0.1, contrast_range=(0.9, 1.1),
                  noise_std=0.02):
    """
    Build dataset that generates overlapping crops from TFRecords
    
    Args:
        tfrecord_pattern: Glob pattern for TFRecord files
        crop_size: Size of each crop (e.g., 256)
        stride: Step size between crops (e.g., 192 gives 64px overlap)
        batch_size: Batch size
        shuffle: Whether to shuffle crops
        seed: Random seed for determinism (if None, non-deterministic)
        augment: Whether to apply augmentations
        global_mean: Global mean for standardization
        global_std: Global std for standardization
        flip_prob: Probability of flipping
        rotate_prob: Probability of rotation
        brightness_delta: Max brightness change
        contrast_range: (min, max) contrast multiplier
        noise_std: Standard deviation of Gaussian noise
    
    Returns:
        tf.data.Dataset
    """
    # Set determinism options
    if seed is not None:
        tf.random.set_seed(seed)
        options = tf.data.Options()
        options.deterministic = True
    
    # Find TFRecord files
    tfrecord_files = sorted(glob.glob(tfrecord_pattern))
    if not tfrecord_files:
        raise ValueError(f"No TFRecord files found: {tfrecord_pattern}")
    
    print(f"Found {len(tfrecord_files)} TFRecord files")
    if global_mean is not None:
        print(f"Standardization: mean={global_mean:.4f}, std={global_std:.4f}")
    if augment:
        print(f"Augmentations enabled: flip_prob={flip_prob}, rotate_prob={rotate_prob}, noise_std={noise_std}")
    
    # Build dataset
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    
    if seed is not None:
        dataset = dataset.with_options(options)
    
    # Parse examples
    dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Generate crops from each image (with standardization)
    dataset = dataset.map(
        lambda data: create_overlapping_crops(data, crop_size, stride, global_mean, global_std),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Unbatch to treat each crop as an individual sample
    dataset = dataset.unbatch()
    
    # Apply augmentations if enabled
    if augment:
        dataset = dataset.map(
            lambda data: augment_crop(data, flip_prob, rotate_prob,
                                    brightness_delta, contrast_range, noise_std),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Shuffle, batch, and prefetch
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000, seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def get_jax_iterator(tfrecord_pattern, crop_size=256, stride=192, batch_size=32, 
                     shuffle=True, seed=None, augment=False,
                     global_mean=None, global_std=None,
                     flip_prob=0.5, rotate_prob=0.5,
                     brightness_delta=0.1, contrast_range=(0.9, 1.1),
                     noise_std=0.02):
    """Get JAX iterator for training"""
    tfds = build_dataset(tfrecord_pattern, crop_size, stride, batch_size, shuffle, seed,
                        augment, global_mean, global_std, flip_prob, rotate_prob,
                        brightness_delta, contrast_range, noise_std)
    return create_input_iter(tfds)

def compute_global_stats(tfrecord_pattern):
    """
    Compute global mean and std across entire dataset (excluding NaNs)
    
    Args:
        tfrecord_pattern: Glob pattern for TFRecord files
    
    Returns:
        (global_mean, global_std) tuple
    """
    print("Computing global statistics...")
    
    tfrecord_files = sorted(glob.glob(tfrecord_pattern))
    if not tfrecord_files:
        raise ValueError(f"No TFRecord files found: {tfrecord_pattern}")
    
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parse_example)
    
    # Welford's algorithm for numerically stable computation
    count = 0
    mean = 0.0
    m2 = 0.0
    
    for data in dataset:
        image = data['image'].numpy()
        
        # Flatten and remove NaNs
        values = image.flatten()
        values = values[~np.isnan(values)]
        
        for value in values:
            count += 1
            delta = value - mean
            mean += delta / count
            delta2 = value - mean
            m2 += delta * delta2
    
    if count < 2:
        raise ValueError("Not enough valid values to compute statistics")
    
    variance = m2 / (count - 1)
    std = np.sqrt(variance)
    
    print(f"Global statistics computed from {count:,} pixels")
    print(f"  Mean: {mean:.6f}")
    print(f"  Std:  {std:.6f}")
    
    return float(mean), float(std)

def reconstruct_from_crops(crops, num_rows, num_cols, crop_size, stride):
    """
    Reconstruct full image from crops using averaging in overlap regions
    
    Args:
        crops: Dict of {(row_idx, col_idx): crop_array}
        num_rows: Number of crop rows
        num_cols: Number of crop columns
        crop_size: Size of each crop
        stride: Stride between crops
    
    Returns:
        Reconstructed image array
    """
    # Calculate output dimensions
    height = (num_rows - 1) * stride + crop_size
    width = (num_cols - 1) * stride + crop_size
    channels = crops[(0, 0)].shape[2]
    
    # Initialize output and weight arrays for averaging overlaps
    output = np.zeros((height, width, channels), dtype=np.float32)
    weights = np.zeros((height, width, channels), dtype=np.float32)
    
    # Place each crop
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            if (row_idx, col_idx) not in crops:
                continue
            
            crop = crops[(row_idx, col_idx)]
            
            # Calculate position
            y_start = row_idx * stride
            x_start = col_idx * stride
            y_end = y_start + crop_size
            x_end = x_start + crop_size
            
            # Add crop to output with weights
            output[y_start:y_end, x_start:x_end, :] += crop
            weights[y_start:y_end, x_start:x_end, :] += 1.0
    
    # Average overlapping regions
    output = output / np.maximum(weights, 1.0)
    
    return output

def verify_crops_reconstruction(tfrecord_pattern, crop_size=256, stride=192):
    """
    Verify crops by reconstructing original images and comparing
    NO standardization or augmentation - just pure crop verification
    
    Args:
        tfrecord_pattern: Glob pattern for TFRecord files
        crop_size: Size of each crop
        stride: Step size between crops
    
    Returns:
        Dict with verification results
    """
    print("=== Crop Reconstruction Verification ===\n")
    
    # Load original images
    print("Loading original images...")
    tfrecord_files = sorted(glob.glob(tfrecord_pattern))
    original_dataset = tf.data.TFRecordDataset(tfrecord_files)
    original_dataset = original_dataset.map(parse_example)
    
    originals = {}
    for data in original_dataset:
        sample_name = data['sample_name'].numpy().decode('utf-8')
        originals[sample_name] = data['image'].numpy()
    
    print(f"Loaded {len(originals)} original images\n")
    
    # Load cropped dataset (NO standardization, NO augmentation)
    print("Loading cropped images...")
    crop_dataset = build_dataset(tfrecord_pattern, crop_size, stride, batch_size=1, 
                                 shuffle=False, augment=False)
    
    crops_by_sample = {}
    for batch in crop_dataset:
        sample_name = batch['sample_name'].numpy()[0].decode('utf-8')
        row_idx = int(batch['row_idx'].numpy()[0])
        col_idx = int(batch['col_idx'].numpy()[0])
        num_rows = int(batch['num_rows'].numpy()[0])
        num_cols = int(batch['num_cols'].numpy()[0])
        image = batch['image'].numpy()[0]
        
        if sample_name not in crops_by_sample:
            crops_by_sample[sample_name] = {
                'num_rows': num_rows,
                'num_cols': num_cols,
                'crops': {}
            }
        
        crops_by_sample[sample_name]['crops'][(row_idx, col_idx)] = image
    
    print(f"Loaded crops from {len(crops_by_sample)} images\n")
    
    # Reconstruct and compare
    results = []
    all_passed = True
    
    for sample_name, original in originals.items():
        print(f"Verifying: {sample_name}")
        
        if sample_name not in crops_by_sample:
            print(f"  ❌ ERROR: No crops found for this sample\n")
            all_passed = False
            continue
        
        crop_data = crops_by_sample[sample_name]
        num_rows = crop_data['num_rows']
        num_cols = crop_data['num_cols']
        crops = crop_data['crops']
        
        # Reconstruct image from crops
        reconstructed = reconstruct_from_crops(crops, num_rows, num_cols, crop_size, stride)
        
        # Compare
        orig_h, orig_w = original.shape[:2]
        recon_h, recon_w = reconstructed.shape[:2]
        
        print(f"  Original shape: {original.shape}")
        print(f"  Reconstructed shape: {reconstructed.shape}")
        print(f"  Crops: {num_rows}x{num_cols} = {len(crops)}")
        
        # Check dimensions match
        if orig_h != recon_h or orig_w != recon_w:
            print(f"  ❌ ERROR: Shape mismatch!")
            all_passed = False
            results.append({
                'sample_name': sample_name,
                'passed': False,
                'error': 'shape_mismatch'
            })
            continue
        
        # Check pixel values match
        max_diff = np.max(np.abs(original - reconstructed))
        mean_diff = np.mean(np.abs(original - reconstructed))
        
        print(f"  Max pixel difference: {max_diff:.6f}")
        print(f"  Mean pixel difference: {mean_diff:.6f}")
        
        if np.allclose(original, reconstructed, rtol=1e-5, atol=1e-6):
            print(f"  ✅ PASSED: Reconstruction matches original perfectly!\n")
            results.append({
                'sample_name': sample_name,
                'passed': True,
                'max_diff': float(max_diff),
                'mean_diff': float(mean_diff)
            })
        else:
            print(f"  ❌ FAILED: Reconstruction differs from original\n")
            all_passed = False
            results.append({
                'sample_name': sample_name,
                'passed': False,
                'max_diff': float(max_diff),
                'mean_diff': float(mean_diff),
                'error': 'pixel_mismatch'
            })
    
    # Summary
    print("=" * 60)
    print("=== VERIFICATION SUMMARY ===")
    passed_count = sum(1 for r in results if r['passed'])
    print(f"Images verified: {len(results)}")
    print(f"Passed: {passed_count}/{len(results)}")
    
    if all_passed:
        print("\n✅ ALL CHECKS PASSED - Cropping is perfect!")
    else:
        print("\n❌ SOME CHECKS FAILED - Review errors above")
    
    return {
        'passed': all_passed,
        'results': results,
        'total': len(results),
        'passed_count': passed_count,
        'originals': originals,
        'crops_by_sample': crops_by_sample
    }

def visualize_reconstruction_comparison(sample_name, original, reconstructed, save_path=None):
    """
    Visualize original vs reconstructed image and difference
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original
    axes[0].imshow(original[:, :, 0], cmap='gray')
    axes[0].set_title(f'Original\n{sample_name}')
    axes[0].axis('off')
    
    # Reconstructed
    axes[1].imshow(reconstructed[:, :, 0], cmap='gray')
    axes[1].set_title('Reconstructed from Crops')
    axes[1].axis('off')
    
    # Difference
    diff = np.abs(original - reconstructed)
    im = axes[2].imshow(diff[:, :, 0], cmap='hot')
    axes[2].set_title(f'Absolute Difference\nMax: {np.max(diff):.6f}')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig