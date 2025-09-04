import sys
import gc
import os
import torch
import jax
import jax.numpy as jnp
import numpy as np
import jax.random as random
import jaxwt as jwt
from flax import linen as nn
from jax_vae_t import VAE as JVAE
from pytorch_vae_t import VAE as TVAE
from functools import reduce
from operator import getitem
import torch.onnx

get_nested = lambda d, keys: reduce(getitem, keys, d)

def convert_gn(params_j, params_t, keys_j, key_t):
    gn = get_nested(params_j, keys_j)
    gn_scale = np.array(gn['scale'])
    gn_bias = np.array(gn['bias'])

    params_t[f'{key_t}.weight'].copy_(torch.from_numpy(gn_scale))
    params_t[f'{key_t}.bias'].copy_(torch.from_numpy(gn_bias))

    # print('happy, converted', key_t)

def convert_convt(params_j, params_t, keys_j, key_t):
    conv = get_nested(params_j, keys_j)
    flax_kernel = conv['kernel']
    # Step 1: Convert from Flax format [kH, kW, inC, outC] to [outC, inC, kH, kW]
    kernel = jnp.transpose(flax_kernel, (3, 2, 0, 1))

    # Step 2: Flip spatially (both height and width dimensions)
    kernel = jnp.flip(kernel, axis=(2, 3))

    # Step 3: Swap input/output channels back to PyTorch format [inC, outC, kH, kW]
    kernel = jnp.transpose(kernel, (1, 0, 2, 3))

    conv_kernel = np.array(kernel) 
    params_t[f'{key_t}.weight'].copy_(torch.from_numpy(conv_kernel))

    if 'bias' in conv.keys():
        bias = np.array(conv['bias'])
        params_t[f'{key_t}.bias'].copy_(torch.from_numpy(bias))
    
    # print('happy, converted', key_t)


def convert_conv(params_j, params_t, keys_j, key_t):
    conv = get_nested(params_j, keys_j)
    conv_kernel = np.transpose(np.array(conv['kernel']), (3, 2, 0, 1))
    params_t[f'{key_t}.weight'].copy_(torch.from_numpy(conv_kernel))

    if 'bias' in conv.keys():
        bias = np.array(conv['bias'])
        params_t[f'{key_t}.bias'].copy_(torch.from_numpy(bias))
    
    # print('happy, converted', key_t)

def convert_dense(params_j, params_t, keys_j, key_t):
    dense = get_nested(params_j, keys_j)
    dense_kernel = np.transpose(np.array(dense['kernel']), (1, 0))
    params_t[f'{key_t}.weight'].copy_(torch.from_numpy(dense_kernel))
    
    if 'bias' in dense.keys():
        bias = np.array(dense['bias'])
        params_t[f'{key_t}.bias'].copy_(torch.from_numpy(bias))

    # print('happy, converted', key_t)


def router(name):
    if 'gn' in name or 'ln' in name:
        return convert_gn
    elif 'tonv' in name:
        return convert_convt
    elif 'conv' in name:
        return convert_conv
    elif 'dense' in name:
        return convert_dense

def pretty_print_dict(data, indent=0, path=[], path_torch=[], params_j=None, params_t=None):
    spacing = "  " * indent

    for key, value in data.items():
        if 'Residual' in key:
            num = key.split('_')[-1]
            key_torch = 'residual_blocks.' + num
        else:
            key_torch = key.lower()
        current_path = path + [key]  # Create new path for this key
        current_path_torch = path_torch + [key_torch]
        
        if isinstance(key, str) and key[0].isupper():
            # This is a subdictionary
            if isinstance(value, dict):
                pretty_print_dict(value, indent + 1, current_path, current_path_torch, params_j, params_t)
        else:
            # Regular key-value pair
            # print(f"{spacing}{key}: (path: {' -> '.join(current_path)})")
            # tmp = get_nested(batch_stats_j, current_path)
            # print(value.keys())
            final_torch_key = '.'.join(current_path_torch)
            # print(key)
            # print(final_torch_key)
            router(key)(params_j, params_t, current_path, final_torch_key) 



import orbax
from flax.training import orbax_utils
from add_gns import analyze_model_structure, replace_instancenorm_blocks_with_groupnorm

if __name__ == "__main__":

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    raw_restored = orbax_checkpointer.restore(sys.argv[1])
    params = {'params': raw_restored['model']}
    
    tvae = TVAE(base_features=48, latent_dim=256)
    tvae.eval()
    params_torch = tvae.state_dict()
    pretty_print_dict(params['params'], params_j=params['params'], params_t=params_torch)
    torch.save(params_torch, 'tvae_params.pth')
    dummy_input = torch.randn(1, 1, 256, 256)
    
    import onnxsim
    import onnx

    with torch.no_grad():
        torch.onnx.export(
            tvae,                          # PyTorch model
            dummy_input,                    # Model input
            "vae.onnx",                   # Output file name
            opset_version=20,               # ONNX version 16+ for Burn
            do_constant_folding=True,       # Optimize constant folding
            input_names=['input'],          # Input names
            output_names=['recon', 'waves', 'mu', 'logvar'],        # Output names
            dynamic_axes=None,
            keep_initializers_as_inputs=False,
            )

        print('Original model converted and saved')
        model = onnx.load("vae.onnx")
        onnx.checker.check_model(model)
        model_simplified, check = onnxsim.simplify(
            model,
        )
        onnx.save(model_simplified, "vae.onnx")
        print(f"Simplification successful: {check}")
        # Analyze original model
        print("1. Analyzing original model structure...")
        analyze_model_structure('vae.onnx')

        print("\n" + "="*50)
        dsfsdf
        # Convert InstanceNorm blocks to GroupNorm
        print("2. Converting InstanceNorm blocks to GroupNorm...")
        replace_instancenorm_blocks_with_groupnorm(
            'vae.onnx',
            'vae.onnx',
            num_groups=8  # Change this value as needed
        )

        print("\n" + "="*50)

        # Analyze converted model
        print("3. Analyzing converted model...")
        analyze_model_structure('vae.onnx')

        print(f"\n?~\? Conversion complete!")
        print(f"\nTry importing the converted model into Burn now!")

