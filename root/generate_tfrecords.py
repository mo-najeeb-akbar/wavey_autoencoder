import tensorflow as tf
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
from pathlib import Path
import sys
import math

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_tfrecord_shard(args):
    shard_files, shard_path = args
    
    with tf.io.TFRecordWriter(str(shard_path)) as writer:
        for img_path in shard_files:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                _, encoded_img = cv2.imencode('.png', img)
                img_bytes = encoded_img.tobytes()
                
                example = tf.train.Example(features=tf.train.Features(feature={
                    'features': _bytes_feature(img_bytes)
                }))
                
                writer.write(example.SerializeToString())
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    print(f"Created shard: {shard_path} with {len(shard_files)} images")
    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <cropped_images_dir>")
        sys.exit(1)
    
    img_dir = Path(sys.argv[1])
    output_dir = img_dir.parent / "tfrecords"
    output_dir.mkdir(exist_ok=True)
    
    img_files = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
    
    if not img_files:
        print("No image files found")
        return
    
    num_shards = min(cpu_count(), math.ceil(len(img_files) / 100))
    files_per_shard = math.ceil(len(img_files) / num_shards)
    
    shard_args = []
    for i in range(num_shards):
        start_idx = i * files_per_shard
        end_idx = min(start_idx + files_per_shard, len(img_files))
        shard_files = img_files[start_idx:end_idx]
        shard_path = output_dir / f"shard_{i:04d}.tfrecord"
        shard_args.append((shard_files, shard_path))
    
    print(f"Creating {num_shards} shards with ~{files_per_shard} images each")
    
    with Pool(cpu_count()) as pool:
        results = pool.map(create_tfrecord_shard, shard_args)
    
    print(f"Successfully created {sum(results)} shards from {len(img_files)} images")

if __name__ == "__main__":
    main()
