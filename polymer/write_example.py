import sys, os
sys.path.append('/code')
from load_tfrecords import Dataloader
from models.jax_vae_wavelet import VAE
import numpy as np
import h5py
import orbax
import jax
import jax.numpy as jnp
import jax.random as random
import jaxwt as jwt
from flax.training import orbax_utils

if __name__ == "__main__":
    dataset_path = sys.argv[1]
    dataloader = Dataloader(tfrecord_pattern=os.path.join(dataset_path, "*.tfrecord"), batch_size=1)
    jax_ds = dataloader.get_jax_iterator(shuffle=True)
    
    vae = VAE(base_features=32, latent_dim=128)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    raw_restored = orbax_checkpointer.restore(sys.argv[2])

    @jax.jit
    def run_inference(sample):
        (x_recon, x_waves, mu, log_var) = vae.apply(
            {'params': raw_restored['model']},
            sample,
            training=False,
            key=random.key(0)
        )
        return x_recon

    ex = next(jax_ds)
    imgs = ex['features']
    transformed = jwt.wavedec2(imgs, "haar", level=1, mode="reflect", axes=(1,2))
    waves = jnp.concatenate([transformed[0], transformed[1][1], transformed[1][0], transformed[1][2]], axis=-1)
    res = run_inference(waves)
    print(np.array(res[0]).shape)
    with h5py.File('single_crop.datx', 'w') as f:
        f.create_dataset('crop', data=np.array(imgs[0]))
        f.create_dataset('net_out', data=np.array(res[0]))
