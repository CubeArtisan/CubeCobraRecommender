from pathlib import Path

import numpy as np
import tensorflow as tf

from ..ml.generator import DraftBotGenerator
from .parse_picks import FEATURES, MAX_IN_PACK

COMPRESSION = 'GZIP'
NUM_TRAIN_PICKS = 4096000
NUM_TRAIN_SHARDS = 256
NUM_TEST_SHARDS = 32

if __name__ == '__main__':
    pick_cache_dir = Path('data/parsed_picks/')
    parsed_picks = [np.memmap(pick_cache_dir / f'{name}.bin', mode='r', dtype=dtype,
                              shape=(int((pick_cache_dir / f'{name}.bin').stat().st_size // np.prod(shape) // np.dtype(dtype).itemsize), *shape))
                    for dtype, shape, name in FEATURES]
    generator = DraftBotGenerator(1024, parsed_picks)

    train_cache_dir = pick_cache_dir / 'train'
    train_cache_dir.mkdir(exist_ok=True)
    train_dataset = tf.data.Dataset.from_generator(
        lambda: generator(end=NUM_TRAIN_PICKS),
        output_signature=(tuple(tf.TensorSpec(shape=shape, dtype=dtype) for dtype, shape, _ in FEATURES),
                          tf.TensorSpec(shape=(MAX_IN_PACK,)))
    ).enumerate()
    tf.data.experimental.save(train_dataset, str(train_cache_dir),
                              compression=COMPRESSION,
                              shard_func=lambda x, _: x % NUM_TRAIN_SHARDS)

    test_cache_dir = pick_cache_dir / 'test'
    test_cache_dir.mkdir(exist_ok=True)
    test_dataset = tf.data.Dataset.from_generator(
        lambda: generator(start=NUM_TRAIN_PICKS),
        output_signature=(tuple(tf.TensorSpec(shape=shape, dtype=dtype) for dtype, shape, _ in FEATURES),
                          tf.TensorSpec(shape=(MAX_IN_PACK,)))
    ).enumerate()
    tf.data.experimental.save(test_dataset, str(test_cache_dir),
                              compression=COMPRESSION,
                              shard_func=lambda x, _: x % NUM_TEST_SHARDS)
