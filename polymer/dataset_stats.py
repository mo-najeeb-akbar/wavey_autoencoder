import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import gc

def load_dataset_index(dataset_folder):
    """Load the dataset index to see what's available"""
    index_file = Path(dataset_folder) / 'dataset_index.json'
    if not index_file.exists():
        raise ValueError(f"No dataset_index.json found in {dataset_folder}")
    
    with open(index_file, 'r') as f:
        return json.load(f)

def get_available_datasets(index):
    """Get list of available datasets across all files"""
    all_datasets = set()
    for file_info in index['files'].values():
        all_datasets.update(file_info['datasets'])
    return sorted(list(all_datasets))

def load_sample_data(dataset_folder, sample_name, dataset_name):
    """Load data for one sample"""
    sample_dir = Path(dataset_folder) / sample_name
    data_file = sample_dir / f'{dataset_name}.npy'
    
    if not data_file.exists():
        return None
    
    return np.load(data_file)

def load_instrument_params(dataset_folder, sample_name):
    """Load instrument parameters for one sample"""
    sample_dir = Path(dataset_folder) / sample_name
    params_file = sample_dir / 'instrument_params.json'
    
    if not params_file.exists():
        return None
    
    with open(params_file, 'r') as f:
        return json.load(f)

def compute_sample_stats(args):
    """Compute stats for a single sample - for parallel processing"""
    dataset_folder, sample_name, dataset_name = args
    
    try:
        data = load_sample_data(dataset_folder, sample_name, dataset_name)
        if data is None:
            return None
        
        valid_data = data[~np.isnan(data)]
        if len(valid_data) == 0:
            return None
        
        # Compute stats without storing all data
        stats = {
            'sample': sample_name,
            'total_pixels': data.size,
            'valid_pixels': len(valid_data),
            'valid_percent': 100 * len(valid_data) / data.size,
            'min': float(np.min(valid_data)),
            'max': float(np.max(valid_data)),
            'mean': float(np.mean(valid_data)),
            'std': float(np.std(valid_data)),
            'median': float(np.median(valid_data))
        }
        
        # Add percentiles
        percentiles = [1, 5, 25, 75, 95, 99]
        for p in percentiles:
            stats[f'p{p}'] = float(np.percentile(valid_data, p))
        
        # Clean up memory
        del data, valid_data
        gc.collect()
        
        return stats
        
    except Exception as e:
        print(f"Error processing {sample_name}: {e}")
        return None

def pixel_stats(dataset_folder, index, dataset_name):
    """Calculate pixel-level statistics using parallel processing"""
    print(f"\n=== PIXEL STATS: {dataset_name.upper()} ===")
    
    # Prepare arguments for parallel processing
    sample_names = list(index['files'].keys())
    args_list = [(dataset_folder, name, dataset_name) for name in sample_names]
    
    # Use parallel processing
    n_cores = min(cpu_count(), 8)  # Don't use too many cores
    print(f"Processing {len(sample_names)} samples using {n_cores} cores...")
    
    with Pool(n_cores) as pool:
        results = pool.map(compute_sample_stats, args_list)
    
    # Filter out None results
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("No valid data found!")
        return
    
    # Aggregate statistics efficiently
    total_valid_pixels = sum(r['valid_pixels'] for r in valid_results)
    total_pixels = sum(r['total_pixels'] for r in valid_results)
    
    print(f"Samples with data: {len(valid_results)}/{len(sample_names)}")
    print(f"Total valid pixels: {total_valid_pixels:,}")
    print(f"Overall valid %: {100 * total_valid_pixels / total_pixels:.1f}%")
    
    # Global statistics from sample statistics (approximate but fast)
    all_mins = [r['min'] for r in valid_results]
    all_maxs = [r['max'] for r in valid_results]
    all_means = [r['mean'] for r in valid_results]
    all_stds = [r['std'] for r in valid_results]
    
    print(f"Global range: {min(all_mins):.3f} to {max(all_maxs):.3f}")
    print(f"Sample means: {min(all_means):.3f} to {max(all_means):.3f}")
    print(f"Sample stds: {min(all_stds):.3f} to {max(all_stds):.3f}")
    
    # Show range of percentiles across samples
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    for p in percentiles:
        if p == 50:
            values = [r['median'] for r in valid_results]
            print(f"Sample medians: {min(values):.3f} to {max(values):.3f}")
        else:
            values = [r[f'p{p}'] for r in valid_results]
            print(f"Sample {p}th percentiles: {min(values):.3f} to {max(values):.3f}")

def sample_stats(dataset_folder, index, dataset_name):
    """Calculate per-sample statistics"""
    print(f"\n=== SAMPLE STATS: {dataset_name.upper()} ===")
    
    # Reuse the parallel computation
    sample_names = list(index['files'].keys())
    args_list = [(dataset_folder, name, dataset_name) for name in sample_names]
    
    n_cores = min(cpu_count(), 8)
    with Pool(n_cores) as pool:
        results = pool.map(compute_sample_stats, args_list)
    
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("No valid data found!")
        return
    
    # Summary across samples
    valid_percents = [s['valid_percent'] for s in valid_results]
    means = [s['mean'] for s in valid_results]
    stds = [s['std'] for s in valid_results]
    
    print(f"Samples: {len(valid_results)}")
    print(f"Valid pixel %: {min(valid_percents):.1f}% to {max(valid_percents):.1f}%")
    print(f"Sample means: {min(means):.3f} to {max(means):.3f}")
    print(f"Sample stds: {min(stds):.3f} to {max(stds):.3f}")
    
    # Show worst/best samples
    worst = min(valid_results, key=lambda x: x['valid_percent'])
    best = max(valid_results, key=lambda x: x['valid_percent'])
    
    print(f"\nWorst coverage: {worst['sample']} ({worst['valid_percent']:.1f}%)")
    print(f"Best coverage: {best['sample']} ({best['valid_percent']:.1f}%)")
    
    # Show samples with very low coverage
    low_coverage = [r for r in valid_results if r['valid_percent'] < 50]
    if low_coverage:
        print(f"\nLow coverage samples (<50%): {len(low_coverage)}")
        for r in sorted(low_coverage, key=lambda x: x['valid_percent'])[:5]:
            print(f"  {r['sample']}: {r['valid_percent']:.1f}%")

def fast_coverage_check(args):
    """Quick check if sample has data for dataset"""
    dataset_folder, sample_name, dataset_name = args
    
    sample_dir = Path(dataset_folder) / sample_name
    data_file = sample_dir / f'{dataset_name}.npy'
    
    return data_file.exists()

def coverage_stats(dataset_folder, index):
    """Show data coverage across samples and datasets"""
    print(f"\n=== COVERAGE STATS ===")
    
    available_datasets = get_available_datasets(index)
    sample_names = list(index['files'].keys())
    
    print(f"Total samples: {len(sample_names)}")
    print(f"Available datasets: {available_datasets}")
    
    # Fast parallel coverage check
    n_cores = min(cpu_count(), 8)
    
    coverage = {}
    for dataset in available_datasets:
        args_list = [(dataset_folder, name, dataset) for name in sample_names]
        
        with Pool(n_cores) as pool:
            exists_list = pool.map(fast_coverage_check, args_list)
        
        coverage[dataset] = sum(exists_list)
    
    print("\nDataset coverage:")
    for dataset, count in coverage.items():
        percent = 100 * count / len(sample_names)
        print(f"  {dataset}: {count}/{len(sample_names)} ({percent:.1f}%)")

def instrument_stats(dataset_folder, index):
    """Show instrument parameter variations across dataset"""
    print(f"\n=== INSTRUMENT STATS ===")
    
    # Sample a subset for efficiency if dataset is very large
    sample_names = list(index['files'].keys())
    if len(sample_names) > 100:
        print(f"Sampling 100 files from {len(sample_names)} for instrument analysis...")
        import random
        sample_names = random.sample(sample_names, 100)
    
    # Collect all instrument parameters
    params_by_category = defaultdict(list)
    
    for sample_name in sample_names:
        params = load_instrument_params(dataset_folder, sample_name)
        if params:
            for category, values in params.items():
                if isinstance(values, dict):
                    for key, value in values.items():
                        if isinstance(value, (int, float)):
                            params_by_category[f"{category}.{key}"].append(value)
                        elif isinstance(value, str) and value:
                            params_by_category[f"{category}.{key}"].append(value)
    
    # Show variations
    for param_name, values in params_by_category.items():
        if len(set(values)) == 1:
            print(f"{param_name}: {values[0]} (constant)")
        else:
            if all(isinstance(v, (int, float)) for v in values):
                print(f"{param_name}: {min(values)} to {max(values)} (varies)")
            else:
                unique_vals = list(set(values))
                if len(unique_vals) <= 3:
                    print(f"{param_name}: {unique_vals} (varies)")
                else:
                    print(f"{param_name}: {len(unique_vals)} different values")

def instrument_correlation_analysis(dataset_folder, index, dataset_name):
    """Analyze how instrument settings correlate with pixel-level results"""
    print(f"\n=== INSTRUMENT CORRELATION: {dataset_name.upper()} ===")
    
    # Collect both instrument params and pixel stats
    sample_names = list(index['files'].keys())
    args_list = [(dataset_folder, name, dataset_name) for name in sample_names]
    
    n_cores = min(cpu_count(), 8)
    print(f"Processing {len(sample_names)} samples...")
    
    with Pool(n_cores) as pool:
        pixel_results = pool.map(compute_sample_stats, args_list)
    
    # Filter valid results and collect instrument params
    combined_data = []
    for i, pixel_stats in enumerate(pixel_results):
        if pixel_stats is not None:
            sample_name = sample_names[i]
            instrument_params = load_instrument_params(dataset_folder, sample_name)
            if instrument_params:
                combined_data.append({
                    'sample': sample_name,
                    'pixel_stats': pixel_stats,
                    'instrument': instrument_params
                })
    
    if len(combined_data) < 10:
        print("Not enough data for correlation analysis")
        return
    
    print(f"Analyzing {len(combined_data)} samples with complete data")
    
    # Extract key variables for correlation
    analyze_correlations(combined_data, dataset_name)

def analyze_correlations(combined_data, dataset_name):
    """Analyze correlations between instrument settings and pixel statistics"""
    
    # Key instrument parameters to analyze
    instrument_keys = [
        ('measurement.light_level_pct', 'Light Level %'),
        ('optics.numerical_aperture', 'Numerical Aperture'),
        ('optics.magnification', 'Magnification'),
        ('camera.frame_count', 'Frame Count'),
        ('scan.increment_nm', 'Scan Increment (nm)'),
        ('stage_position.x_mm', 'Stage X (mm)'),
        ('stage_position.y_mm', 'Stage Y (mm)'),
        ('stage_position.z_mm', 'Stage Z (mm)'),
        ('stage_position.pitch_urad', 'Stage Pitch (μrad)'),
        ('stage_position.roll_urad', 'Stage Roll (μrad)')
    ]
    
    # Key pixel statistics to analyze
    pixel_keys = [
        ('valid_percent', 'Valid Pixel %'),
        ('mean', 'Mean Value'),
        ('std', 'Std Dev'),
        ('median', 'Median'),
        ('min', 'Min Value'),
        ('max', 'Max Value')
    ]
    
    correlations_found = []
    
    for inst_key, inst_name in instrument_keys:
        # Extract instrument values
        inst_values = []
        pixel_stats_list = []
        
        for data in combined_data:
            # Navigate nested dict structure
            inst_val = data['instrument']
            for key_part in inst_key.split('.'):
                if isinstance(inst_val, dict) and key_part in inst_val:
                    inst_val = inst_val[key_part]
                else:
                    inst_val = None
                    break
            
            if inst_val is not None and isinstance(inst_val, (int, float)):
                inst_values.append(float(inst_val))
                pixel_stats_list.append(data['pixel_stats'])
        
        if len(inst_values) < 10:
            continue
        
        # Check if instrument parameter varies enough
        inst_std = np.std(inst_values)
        if inst_std == 0:
            print(f"{inst_name}: constant across all samples")
            continue
        
        # Calculate correlations with each pixel statistic
        for pixel_key, pixel_name in pixel_keys:
            pixel_values = [stats[pixel_key] for stats in pixel_stats_list]
            
            # Calculate correlation coefficient
            correlation = np.corrcoef(inst_values, pixel_values)[0, 1]
            
            if abs(correlation) > 0.3:  # Strong correlation threshold
                correlations_found.append({
                    'instrument': inst_name,
                    'pixel_stat': pixel_name,
                    'correlation': correlation,
                    'inst_range': (min(inst_values), max(inst_values)),
                    'pixel_range': (min(pixel_values), max(pixel_values))
                })
    
    # Display significant correlations
    if correlations_found:
        print(f"\n🔍 SIGNIFICANT CORRELATIONS (|r| > 0.3):")
        correlations_found.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        for corr in correlations_found:
            direction = "📈" if corr['correlation'] > 0 else "📉"
            print(f"{direction} {corr['instrument']} ↔ {corr['pixel_stat']}")
            print(f"   Correlation: {corr['correlation']:.3f}")
            print(f"   {corr['instrument']}: {corr['inst_range'][0]:.2f} to {corr['inst_range'][1]:.2f}")
            print(f"   {corr['pixel_stat']}: {corr['pixel_range'][0]:.3f} to {corr['pixel_range'][1]:.3f}")
            print()
        
        # Suggest preprocessing strategies
        print("💡 PREPROCESSING RECOMMENDATIONS:")
        for corr in correlations_found[:3]:  # Top 3 correlations
            if 'Light Level' in corr['instrument']:
                print(f"• Normalize by light level: intensity = intensity / (light_level_pct / 100)")
            elif 'Stage' in corr['instrument']:
                print(f"• Consider spatial standardization across stage positions")
            elif 'Frame Count' in corr['instrument']:
                print(f"• Normalize by exposure: intensity = intensity / frame_count")
            elif 'Magnification' in corr['instrument']:
                print(f"• Standardize field of view based on magnification")
    else:
        print("No strong correlations found (|r| > 0.3)")
        print("Your data appears consistent across instrument settings!")

def group_analysis_by_setting(dataset_folder, index, dataset_name, setting_key):
    """Analyze pixel stats grouped by a specific instrument setting"""
    print(f"\n=== GROUP ANALYSIS: {dataset_name.upper()} by {setting_key} ===")
    
    # Collect data
    sample_names = list(index['files'].keys())
    args_list = [(dataset_folder, name, dataset_name) for name in sample_names]
    
    n_cores = min(cpu_count(), 8)
    with Pool(n_cores) as pool:
        pixel_results = pool.map(compute_sample_stats, args_list)
    
    # Group by setting value
    groups = defaultdict(list)
    
    for i, pixel_stats in enumerate(pixel_results):
        if pixel_stats is not None:
            sample_name = sample_names[i]
            instrument_params = load_instrument_params(dataset_folder, sample_name)
            if instrument_params:
                # Extract setting value
                setting_val = instrument_params
                for key_part in setting_key.split('.'):
                    if isinstance(setting_val, dict) and key_part in setting_val:
                        setting_val = setting_val[key_part]
                    else:
                        setting_val = None
                        break
                
                if setting_val is not None:
                    groups[setting_val].append(pixel_stats)
    
    # Display group statistics
    if len(groups) > 1:
        print(f"Found {len(groups)} different {setting_key} values:")
        
        for setting_val, stats_list in sorted(groups.items()):
            means = [s['mean'] for s in stats_list]
            valid_percents = [s['valid_percent'] for s in stats_list]
            
            print(f"\n{setting_key} = {setting_val} ({len(stats_list)} samples):")
            print(f"  Mean values: {min(means):.3f} to {max(means):.3f}")
            print(f"  Valid %: {min(valid_percents):.1f}% to {max(valid_percents):.1f}%")
    else:
        print(f"{setting_key} is constant across all samples")

def main():
    if len(sys.argv) != 2:
        print("Usage: python dataset_stats.py <dataset_folder>")
        sys.exit(1)
    
    dataset_folder = Path(sys.argv[1])
    
    try:
        index = load_dataset_index(dataset_folder)
        available_datasets = get_available_datasets(index)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    print(f"Dataset: {dataset_folder}")
    print(f"Available datasets: {available_datasets}")
    
    while True:
        print("\n" + "="*50)
        print("OPTIONS:")
        print("1. Coverage stats (overview)")
        print("2. Instrument settings")
        print("3. Pixel stats for dataset")
        print("4. Sample stats for dataset") 
        print("5. All stats for dataset")
        print("6. Instrument correlation analysis")
        print("7. Group analysis by setting")
        print("0. Exit")
        
        choice = input("\nChoice: ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            coverage_stats(dataset_folder, index)
        elif choice == "2":
            instrument_stats(dataset_folder, index)
        elif choice in ["3", "4", "5", "6"]:
            print(f"\nAvailable datasets: {available_datasets}")
            dataset = input("Dataset name: ").strip()
            
            if dataset not in available_datasets:
                print(f"Error: '{dataset}' not found")
                continue
            
            if choice == "3":
                pixel_stats(dataset_folder, index, dataset)
            elif choice == "4":
                sample_stats(dataset_folder, index, dataset)
            elif choice == "5":
                pixel_stats(dataset_folder, index, dataset)
                sample_stats(dataset_folder, index, dataset)
            elif choice == "6":
                instrument_correlation_analysis(dataset_folder, index, dataset)
        elif choice == "7":
            print(f"\nAvailable datasets: {available_datasets}")
            dataset = input("Dataset name: ").strip()
            
            if dataset not in available_datasets:
                print(f"Error: '{dataset}' not found")
                continue
            
            print("\nCommon settings to analyze:")
            print("  measurement.light_level_pct")
            print("  optics.magnification") 
            print("  camera.frame_count")
            print("  stage_position.x_mm")
            
            setting = input("Setting key: ").strip()
            group_analysis_by_setting(dataset_folder, index, dataset, setting)
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()
