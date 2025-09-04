import onnxruntime as ort
import numpy as np
import os 
import sys
import h5py

def load_hdf5_arrays(filename):
    """Load crop and net_out arrays from HDF5 file"""
    with h5py.File(filename, 'r') as f:
        crop = np.array(f['crop'])
        net_out = np.array(f['net_out'])
    return crop, net_out

if __name__ == "__main__":
    session = ort.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider'])
    session.set_providers(['CPUExecutionProvider'], [{'enable_cpu_mem_arena': False}])
    img, jax_out = load_hdf5_arrays(sys.argv[2])
    inp = np.squeeze(img)[np.newaxis, np.newaxis, :, :]
    print(inp.shape)
    torch_out = session.run(None, {"input": inp})[0]
    torch_out = np.squeeze(torch_out)

    np.testing.assert_almost_equal(np.squeeze(jax_out), torch_out, decimal=4)

