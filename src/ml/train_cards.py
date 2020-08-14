# enable sibling imports
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    from os.path import dirname as dir
    sys.path.append(dir(sys.path[0]))

from generator import CardDataGenerator
from model import CardEncoderWrapper
from non_ml import utils

if __name__ == "__main__":
    args = sys.argv[1:]

    epochs = int(args[0])
    batch_size = int(args[1])
    name = args[2]
    walk_len = int(args[3])
    num_walks = int(args[4])

    map_file = '././data/maps/nameToId.json'
    folder = "././data/cube/"

    print('Loading Cube Data . . .\n')

    int_to_card = json.load(open('././output/int_to_card.json', 'rb'))
    int_to_card = {int(k): v for k, v in int_to_card.items()}  # if int(k) < 1000}
    card_to_int = {v: k for k, v in int_to_card.items()}
    num_cards = len(int_to_card)

    print('Loading Adjacency Matrix . . .\n')

    adj_mtx = np.load('././output/full_adj_mtx.npy')

    print('Setting up Generator . . .\n')
    generator = CardDataGenerator(
        adj_mtx,
        walk_len,
        num_walks,
        batch_size=batch_size,
        data_path='././output/card_generator_data.json',
    )

    print('Setting Up Model . . . \n')
    output_dir = f'././ml_files/{name}'
    with open('cards.json', 'r', encoding="utf-8") as cardsjson:
        cards = json.load(cardsjson)
        cards = [cards.get(int_to_card[i], "") for i in range(num_cards)]
        for card in cards:
            if "otherParses" in card:
                del card["otherParses"]
    autoencoder = CardEncoderWrapper(cards)
    autoencoder.run_eagerly = True
    autoencoder.compile(
        optimizer='adam',
        loss=['binary_crossentropy'],
        loss_weights=[1.0],
        metrics=['accuracy'],
    )
    autoencoder.fit(generator, epochs=1)  # , use_multiprocessing=True)

    print(autoencoder.summary())

    # pdb.set_trace()
    checkpoint_path = output_dir + '/checkpoints/cp-{epoch:04d}.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint is not None:
        autoencoder.load_weights(latest_checkpoint)
        autoencoder.save(output_dir, save_format='tf')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        period=5)
    autoencoder.fit(
        generator,
        epochs=epochs,
        callbacks=[cp_callback]
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    autoencoder.save(output_dir, save_format='tf')
