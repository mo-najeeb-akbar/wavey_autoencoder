import sys
import h5py
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

def safe_decode(value):
    """Safely decode bytes to string, handle both bytes and strings"""
    if isinstance(value, bytes):
        return value.decode()
    return str(value) if value is not None else ''

def extract_useful_params(filepath):
    """Extract key measurement parameters from DATX file"""
    params = {}
    
    with h5py.File(filepath, 'r') as f:
        attrs = f['Measurement/Attributes'].attrs
        
        params['camera'] = {
            'width': int(attrs.get('Data Context.Data Attributes.Camera Width:Value', [0])[0]),
            'height': int(attrs.get('Data Context.Data Attributes.Camera Height:Value', [0])[0]),
            'mode': safe_decode(attrs.get('Data Context.Data Attributes.Camera Mode', [''])[0]),
            'frame_count': int(attrs.get('Data Context.Data Attributes.Frame Count', [0])[0])
        }
        
        params['optics'] = {
            'objective': safe_decode(attrs.get('Data Context.Data Attributes.ObjectiveID', [''])[0]),
            'numerical_aperture': float(attrs.get('Data Context.Data Attributes.Numerical Aperture:Value', [0])[0]),
            'magnification': float(attrs.get('Data Context.Data Attributes.System Magnification:Value', [0])[0]),
            'wavelength_m': float(attrs.get('Data Context.Data Attributes.Wavelength:Value', [0])[0]),
            'wavelength_nm': float(attrs.get('Data Context.Data Attributes.Wavelength:Value', [0])[0]) * 1e9
        }
        
        params['scan'] = {
            'increment_m': float(attrs.get('Data Context.Data Attributes.Scan Increment:Value', [0])[0]),
            'increment_nm': float(attrs.get('Data Context.Data Attributes.Scan Increment:Value', [0])[0]) * 1e9,
            'direction': safe_decode(attrs.get('Data Context.Data Attributes.Scan Direction', [''])[0]),
            'length': safe_decode(attrs.get('Data Context.Data Attributes.Scan Length', [''])[0]),
            'rate': safe_decode(attrs.get('Data Context.Data Attributes.Scan Rate', [''])[0]),
            'device': safe_decode(attrs.get('Data Context.Data Attributes.Scan Device', [''])[0])
        }
        
        params['resolution'] = {
            'lateral_m': float(attrs.get('Data Context.Lateral Resolution:Value', [0])[0]),
            'lateral_nm': float(attrs.get('Data Context.Lateral Resolution:Value', [0])[0]) * 1e9,
            'z_mode': safe_decode(attrs.get('Data Context.Data Attributes.Z Resolution', [''])[0])
        }
        
        params['stage_position'] = {
            'x_mm': float(attrs.get('Data Context.Data Attributes.Stage X:Value', [0])[0]),
            'y_mm': float(attrs.get('Data Context.Data Attributes.Stage Y:Value', [0])[0]),
            'z_mm': float(attrs.get('Data Context.Data Attributes.Stage Z:Value', [0])[0]),
            'pitch_urad': float(attrs.get('Data Context.Data Attributes.Stage Pitch:Value', [0])[0]),
            'roll_urad': float(attrs.get('Data Context.Data Attributes.Stage Roll:Value', [0])[0])
        }
        
        params['measurement'] = {
            'type': safe_decode(attrs.get('Data Context.Data Attributes.Measurement Type', [''])[0]),
            'mode': safe_decode(attrs.get('Data Context.Data Attributes.Measurement Mode', [''])[0]),
            'light_level_pct': float(attrs.get('Data Context.Data Attributes.Light Level:Value', [0])[0]),
            'modulation_threshold_pct': float(attrs.get('Data Context.Data Attributes.Modulation Threshold:Value', [0])[0])
        }
        
        params['system'] = {
            'instrument': safe_decode(attrs.get('Data Context.Data Attributes.Instrument', [''])[0]),
            'system_type': safe_decode(attrs.get('Data Context.Data Attributes.System Type', [''])[0]),
            'serial_number': safe_decode(attrs.get('Data Context.Data Attributes.System Serial Number', [''])[0]),
            'software_version': safe_decode(attrs.get('Data Context.Data Attributes.Software Info Version', [''])[0])
        }
        
        window = attrs.get('Data Context.Window', [(0, 0, 0, 0)])[0]
        params['data_window'] = {
            'x_start': int(window[0]),
            'y_start': int(window[1]), 
            'width': int(window[2]),
            'height': int(window[3])
        }
    
    return params

def make_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def extract_dataset_metadata(dataset):
    """Extract metadata from a dataset"""
    attrs = dataset.attrs
    
    metadata = {
        'shape': dataset.shape,
        'dtype': str(dataset.dtype),
        'unit': safe_decode(attrs.get('Unit', [''])[0]),
        'no_data_value': float(attrs.get('No Data', [np.nan])[0]),
        'coordinates': list(attrs.get('Coordinates', [(0, 0, 0, 0)])[0]),
        'group_number': int(attrs.get('Group Number', [1])[0])
    }
    
    # Extract conversion factors for physical units
    x_conv = attrs.get('X Converter', [None])[0]
    y_conv = attrs.get('Y Converter', [None])[0]
    z_conv = attrs.get('Z Converter', [None])[0]
    
    if x_conv is not None and len(x_conv) >= 3:
        metadata['x_scale'] = float(x_conv[2][1])  # meters per pixel
        metadata['x_scale_nm'] = float(x_conv[2][1]) * 1e9  # nm per pixel
    
    if y_conv is not None and len(y_conv) >= 3:
        metadata['y_scale'] = float(y_conv[2][1])  # meters per pixel  
        metadata['y_scale_nm'] = float(y_conv[2][1]) * 1e9  # nm per pixel
    
    # Height-specific metadata
    if 'Surface' in dataset.name or 'Bottom' in dataset.name or 'Thickness' in dataset.name:
        metadata['interferometric_scale_factor'] = float(attrs.get('Interferometric Scale Factor', [1.0])[0])
        metadata['obliquity_factor'] = float(attrs.get('Obliquity Factor', [1.0])[0])
        metadata['wavelength'] = float(attrs.get('Wavelength', [0.0])[0])
        
        if z_conv is not None and len(z_conv) >= 3:
            metadata['z_scale'] = float(z_conv[2][1])  # typically wavelength fraction
            metadata['z_offset'] = float(z_conv[2][2]) if len(z_conv[2]) > 2 else 0.0
    
    return metadata

def process_data(data, metadata):
    """Process raw data - handle no-data values and apply basic filtering"""
    no_data_val = metadata['no_data_value']
    
    # For integer data, convert to float64 first to handle NaN
    if 'int' in str(data.dtype):
        processed_data = data.astype(np.float64)
    else:
        processed_data = data.copy()
    
    # Replace no-data values with NaN
    if not np.isnan(no_data_val):
        processed_data[processed_data == no_data_val] = np.nan
    
    # For originally integer data, handle overflow values
    if 'int' in str(data.dtype):
        max_val = np.iinfo(data.dtype).max
        processed_data[processed_data == max_val] = np.nan
    
    return processed_data

def find_datx_files(input_folder):
    """Find all .datx files recursively in input folder"""
    input_path = Path(input_folder)
    if not input_path.exists():
        raise ValueError(f"Input folder does not exist: {input_folder}")
    
    datx_files = list(input_path.rglob("*.datx"))
    
    if not datx_files:
        raise ValueError(f"No .datx files found in {input_folder}")
    
    return datx_files

def process_single_datx(datx_path, config, output_base_dir):
    """Process a single DATX file according to config"""
    
    # Create output directory for this file
    file_stem = datx_path.stem
    file_output_dir = output_base_dir / file_stem
    file_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Available dataset mapping
    dataset_mapping = {
        'surface': 'Measurement/Surface',
        'bottom_surface': 'Measurement/Bottom Surface',
        'thickness': 'Measurement/Thickness',
        'intensity': 'Measurement/Intensity',
        'quality': 'Measurement/Quality',
        'saturation_counts': 'Measurement/Saturation Counts'
    }
    
    results = {}
    
    with h5py.File(datx_path, 'r') as f:
        # Process requested datasets
        for output_name in config['datasets']:
            if output_name not in dataset_mapping:
                print(f"  Warning: Unknown dataset '{output_name}' requested, skipping")
                continue
                
            dataset_path = dataset_mapping[output_name]
            
            if dataset_path not in f:
                print(f"  Warning: Dataset '{dataset_path}' not found in file, skipping")
                continue
            
            dataset = f[dataset_path]
            
            # Extract metadata
            metadata = extract_dataset_metadata(dataset)
            
            # Load and process data
            raw_data = dataset[...]
            processed_data = process_data(raw_data, metadata)
            
            # Save numpy array
            array_file = file_output_dir / f'{output_name}.npy'
            np.save(array_file, processed_data)
            
            # Store metadata for later saving
            results[output_name] = {
                'metadata': metadata,
                'file_path': str(array_file.relative_to(output_base_dir)),
                'shape': processed_data.shape,
                'valid_pixels': int(np.sum(~np.isnan(processed_data))),
                'total_pixels': int(processed_data.size)
            }
    
    # Save metadata files
    if config.get('save_dataset_metadata', True):
        dataset_metadata_file = file_output_dir / 'dataset_metadata.json'
        dataset_metadata = {k: v['metadata'] for k, v in results.items()}
        dataset_metadata_serializable = make_json_serializable(dataset_metadata)
        with open(dataset_metadata_file, 'w') as f:
            json.dump(dataset_metadata_serializable, f, indent=2)
    
    if config.get('save_instrument_params', True):
        instrument_params_file = file_output_dir / 'instrument_params.json'
        instrument_params = extract_useful_params(datx_path)
        with open(instrument_params_file, 'w') as f:
            json.dump(instrument_params, f, indent=2)
    
    return results

def create_dataset_index(output_dir, all_results):
    """Create a master index of all processed files"""
    index = {
        'total_files': len(all_results),
        'files': {}
    }
    
    for file_stem, file_results in all_results.items():
        if file_results:  # Only add if we got results
            first_result = next(iter(file_results.values()))
            index['files'][file_stem] = {
                'datasets': list(file_results.keys()),
                'shape': first_result['shape'],
                'total_pixels': first_result['total_pixels'],
                'folder': file_stem
            }
    
    # Save master index
    index_file = output_dir / 'dataset_index.json'
    with open(index_file, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"\nDataset index saved to: {index_file}")

def create_config_template(output_path="config.json"):
    """Create a simple config template"""
    config = {
        "input_folder": "/path/to/your/datx/files",
        "output_folder": "/path/to/processed/dataset",
        "datasets": ["intensity", "quality", "surface"],
        "save_dataset_metadata": True,
        "save_instrument_params": True
    }
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Config template saved to: {output_path}")
    print("\nAvailable datasets:")
    print("  - surface: Height/topography data")
    print("  - bottom_surface: Bottom interface heights") 
    print("  - thickness: Film thickness measurements")
    print("  - intensity: Raw light intensity")
    print("  - quality: Measurement confidence")
    print("  - saturation_counts: Overexposure detection")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python generate_dataset.py config.json")
        print("  python generate_dataset.py --create-config [filename]")
        sys.exit(1)
    
    if sys.argv[1] == "--create-config":
        output_file = sys.argv[2] if len(sys.argv) > 2 else "config.json"
        create_config_template(output_file)
        return
    
    # Load config
    config_path = sys.argv[1]
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Validate required fields
    required_fields = ['input_folder', 'output_folder', 'datasets']
    for field in required_fields:
        if field not in config:
            print(f"Required field '{field}' missing from config")
            sys.exit(1)
    
    # Setup paths
    input_folder = Path(config['input_folder'])
    output_folder = Path(config['output_folder'])
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Input: {input_folder}")
    print(f"Output: {output_folder}")
    print(f"Datasets: {config['datasets']}")
    
    # Find files
    try:
        datx_files = find_datx_files(input_folder)
        print(f"Found {len(datx_files)} .datx files")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Process files
    all_results = {}
    
    for datx_path in tqdm(datx_files, desc="Processing"):
        try:
            results = process_single_datx(datx_path, config, output_folder)
            if results:
                all_results[datx_path.stem] = results
        except Exception as e:
            print(f"Error processing {datx_path.name}: {e}")
            continue
    
    # Create index
    create_dataset_index(output_folder, all_results)
    
    print(f"Processed {len(all_results)} files successfully")

if __name__ == "__main__":
    main()
