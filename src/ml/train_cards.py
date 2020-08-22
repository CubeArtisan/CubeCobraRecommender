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

    map_file = '././data/maps/nameToId.json'
    folder = "././data/cube/"

    print('Loading Cube Data . . .\n')

    int_to_card = json.load(open('././output/int_to_card.json', 'rb'))
    int_to_card = {int(k): v for k, v in int_to_card.items()}  # if int(k) < 1000}
    card_to_int = {v: k for k, v in int_to_card.items()}
    num_cards = len(int_to_card)

    print('Loading Adjacency Matrix . . .\n')

    adj_mtx = np.load('././output/full_adj_mtx.npy')
    card_counts = np.load('././output/card_counts.npy')

    print('Setting up Generator . . .\n')
    output_dir = f'././ml_files/{name}'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open('cards.json', 'r', encoding="utf-8") as cardsjson:
        cards = json.load(cardsjson)
        cards = [cards.get(int_to_card[i], "") for i in range(num_cards)]
        for card in cards:
            if "otherParses" in card:
                del card["otherParses"]
    generator = CardDataGenerator(
        adj_mtx,
        walk_len,
        card_counts,
        cards,
        batch_size=batch_size,
        data_path='././output/card_generator_data.json',
    )
    print('Setting Up Model . . . \n')
    autoencoder = CardEncoderWrapper(generator.vocab_dict, generator.max_paths, generator.max_path_length)
    autoencoder.run_eagerly = True
    autoencoder.compile(
        optimizer='adam',
        loss=['binary_crossentropy'],
        loss_weights=[1.0],
        metrics=['accuracy'],
    )

    # pdb.set_trace()
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=output_dir,
        monitor='loss',
        verbose=True,
        save_best_only=True,
        mode='min',
        save_freq='epoch')
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                           patience=10,
                                                           min_delta=0.002)
    autoencoder.fit(
        generator,
        epochs=epochs,
        callbacks=[cp_callback]  #, early_stop_callback]
    )
