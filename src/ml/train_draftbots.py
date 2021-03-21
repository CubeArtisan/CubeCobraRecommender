import datetime
import glob
import heapq
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

if __name__ == "__main__":
    import os.path
    sys.path.append(os.path.dirname(sys.path[0]) + '../')

from .generator import DraftBotGenerator
from .draftbots import DraftBot, EMBEDS_DIM
from ..non_ml.parse_picks import FEATURES, MAX_IN_PACK, NUM_LAND_COMBS


if __name__ == "__main__":
    args = sys.argv[1:]
    epochs = int(args[0])
    batch_size = int(args[1])
    name = args[2]
    temperature = float(args[3])

    with open('data/intToCard.json', 'r') as cards_file:
        cards_json = json.load(cards_file)
        card_ratings = [0] + [10 ** ((c.get('elo', 1200) / 800) - 2) for c in cards_json]
        blank_embedding = [1 for _ in range EMBEDS_DIM]
        card_embeddings = [c.get('embedding', blank_embedding)
                           if c.get('embedding', blank_embedding) is not None and len(c.get('embedding', blank_embedding)) == EMBEDS_DIM
                           else blank_embedding for c in cards_json]
    pick_cache_dir = Path('data/parsed_picks/')
    parsed_picks = [np.memmap(pick_cache_dir / f'{name}.bin', mode='r', dtype=dtype,
                              shape=(int((pick_cache_dir / f'{name}.bin').stat().st_size // np.prod(shape) // np.dtype(dtype).itemsize), *shape))
                    for dtype, shape, name in FEATURES]

    output_dir = f'././ml_files/{name}/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print('Setting up Generator . . .\n')
    generator = DraftBotGenerator(batch_size, parsed_picks)
    print('Setting Up Model . . . \n')
    draftbots = DraftBot(card_ratings, card_embeddings, temperature, NUM_LAND_COMBS)
    latest = tf.train.latest_checkpoint(output_dir)
    if latest is not None:
        print('Loading Checkpoint. Saved values are:')
        draftbots.load_weights(latest)
        for var in draftbots.variables:
            print(f'{var.name}: {var.numpy()}')
    draftbots.compile(
        optimizer=tfa.optimizers.AdamW(weight_decay=0, learning_rate=5e-01),
        loss=['categorical_crossentropy'],
        metrics=['categorical_accuracy'],
    )
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=output_dir + 'model',
        monitor='loss',
        verbose=True,
        save_best_only=True,
        mode='min',
        save_freq='epoch')
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    dataset = tf.data.Dataset.from_generator(generator,
                                             output_signature=(tuple(tf.TensorSpec(shape=(batch_size, *shape), dtype=dtype)
                                                                     for dtype, shape, _ in FEATURES),
                                                               tf.TensorSpec(shape=(batch_size, MAX_IN_PACK)))).prefetch(tf.data.AUTOTUNE)
    draftbots.fit(
        dataset,
        epochs=epochs,
        callbacks=[cp_callback]
    )
    Path(f'{output_dir}/final').mkdir(parents=True, exist_ok=True)
    draftbots.save(f'{output_dir}/final', save_format='tf')
