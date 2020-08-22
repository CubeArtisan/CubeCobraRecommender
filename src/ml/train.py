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

from generator import DataGenerator
from model import CC_Recommender, CardEncoderWrapper
from non_ml import utils
from ml import ml_utils


def reset_random_seeds(seed):
    # currently not used
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    args = sys.argv[1:]

    epochs = int(args[0])
    batch_size = int(args[1])
    name = args[2]
    reg = float(args[3])
    noise = float(args[4])

    if len(args) == 6:
        seed = int(args[5])
        reset_random_seeds(seed)

    map_file = '././data/maps/nameToId.json'
    folder = "././data/cube/"

    print('Loading Cube Data . . .\n')

    int_to_card = json.load(open('././output/int_to_card.json', 'rb'))
    int_to_card = {int(k): v for k, v in int_to_card.items()}  # if int(k) < 1000}
    card_to_int = {v: k for k, v in int_to_card.items()}

    num_cubes = utils.get_num_cubes(folder)
    num_cards = len(int_to_card)

    cubes, max_cube_size = utils.build_cubes(folder, num_cubes,
                                             num_cards, card_to_int)

    print('Loading Adjacency Matrix . . .\n')

    adj_mtx = np.load('././output/full_adj_mtx.npy')

    # print('Converting Graph Weights to Probabilities . . . \n')
    print('Creating Graph for Regularization . . . \n')

    # make easier to learn by dropping super low conditional probabilities
    # too_small = np.where(adj_mtx < thresh)
    # y_mtx = adj_mtx.copy()
    # y_mtx[too_small] = 0
    # np.fill_diagonal(y_mtx,1)
    # y_mtx = (adj_mtx/adj_mtx.sum(1)[:,None])
    # y_mtx = np.nan_to_num(y_mtx,0)
    # y_mtx[np.where(y_mtx.sum(1) == 0),np.where(y_mtx.sum(1) == 0)] = 1
    #  adj_mtx = adj_mtx[:1000, :1000]
    y_mtx = (adj_mtx/adj_mtx.sum(1)[:, None])

    print('Setting Up Data for Training . . .\n')


    # x_items = np.zeros(adj_mtx.shape)
    # np.fill_diagonal(x_items,1)
    print('Setting up Generator . . .\n')
    generator = DataGenerator(
        y_mtx,
        cubes,
        num_cards,
        batch_size=batch_size,
        noise=noise,
    )

    print('Setting Up Model . . . \n')
    output_dir = f'././ml_files/{name}'
    temp_save_dir = f'{output_dir}/initial_model'
    if Path(temp_save_dir).is_dir():
        autoencoder = tf.keras.models.load_model(temp_save_dir)
    else:

        with open('cards.json', 'r', encoding="utf-8") as cardsjson:
            cards = json.load(cardsjson)
        cards = [cards.get(int_to_card[i], "") for i in range(num_cards)]
        for card in cards:
            if "otherParses" in card:
                del card["otherParses"]
        all_paths, vocab_count = ml_utils.generate_paths(cards, return_vocab_count=True)
        our_paths = []
        for a in all_paths:
            a += [[0 for _ in range(ml_utils.MAX_PATH_LENGTH)]
                  for _ in range(len(a), ml_utils.NUM_INPUT_PATHS)]
            np.random.shuffle(a)
            a = a[:ml_utils.NUM_INPUT_PATHS]
            our_paths.append(a)
        print('loading model')
        card_model = CardEncoderWrapper(vocab_count, ml_utils.NUM_INPUT_PATHS,
                                        ml_utils.MAX_PATH_LENGTH)
        latest = tf.train.latest_checkpoint('ml_files/')
        card_model.load_weights(latest)
        print('Looking up embeddings')
        STRIDE = 512
        embeddings = []
        for i in range(1, len(all_paths), STRIDE):
            cur_paths = tf.constant(our_paths[i:i + STRIDE])
            card_embeddings = card_model.card_encoder(cur_paths)
            embeddings.append(card_embeddings)
        card_embeddings = tf.concat(embeddings, 0).numpy()
        autoencoder = CC_Recommender(cards, max_cube_size, card_embeddings)
        autoencoder.run_eagerly = True
        autoencoder.compile(
            optimizer='adam',
            loss=['binary_crossentropy', 'kullback_leibler_divergence'],
            loss_weights=[1.0, reg],
            metrics=['accuracy'],
        )
        autoencoder.fit(generator, epochs=1)
        Path(temp_save_dir).mkdir(parents=True, exist_ok=True)
        autoencoder.save(temp_save_dir)
        print("Saved initial model")

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
