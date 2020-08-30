# enable sibling imports
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    from os.path import dirname as dir
    sys.path.append(dir(sys.path[0]))

from generator import DraftBotGenerator
from draftbots import DraftBot
from metrics import FilteredBinaryAccuracy, TripletFilteredAccuracy, ContrastiveFilteredAccuracy
from losses import TripletLoss, ContrastiveLoss

if __name__ == "__main__":
    args = sys.argv[1:]

    epochs = int(args[0])
    batch_size = int(args[1])
    name = args[2]

    map_file = '././data/maps/nameToId.json'
    folder = "././data/cube/"

    print('Loading Cube Data . . .\n')

    int_to_card = json.load(open('././output/int_to_card.json', 'rb'))
    int_to_card = {int(k): v for k, v in int_to_card.items()}
    card_to_int = {v: k for k, v in int_to_card.items()}
    num_cards = len(int_to_card)

    picks = [
        {
            'cards': [0, 1, 2, 3, 4, 5, 6, 7],
            'seen': [17, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            'picked': [8],
            'pickNum': 2,
            'packNum': 1,
            'packs': 6,
            'packSize': 9,
            'pickedIndex': 3
        }
    ]

    output_dir = f'././ml_files/{name}/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print('Setting up Generator . . .\n')
    generator = DraftBotGenerator(
        batch_size,
        num_cards,
        picks,
        card_to_int,
        data_path='./output',
    )
    print('Setting Up Model . . . \n')
    card_model = tf.keras.models.load_model('ml_files/recommender')
    STRIDE = 1024
    embs = [tf.zeros((1, 64))]
    for start in range(0, num_cards, STRIDE):
        cards = np.zeros((STRIDE, num_cards))
        print(min(start + STRIDE, num_cards))
        for i in range(min(STRIDE, num_cards - start)):
            cards[i, i + start] = 1
        our_embs = card_model.encoder(cards)
        embs.append(our_embs)
    embs = tf.concat(embs, axis=0)
    prob_to_play =[[1 for _ in range(18)]] + [[i / 17 for i in range(0, 18)] for _ in range(20)]
    land_requirements = [[j % (i+1) for i in range(5)] for j in range(num_cards + 1)]
    card_colors = [[1, 1, 1, 1, 1] for _ in range(num_cards + 1)]
    is_land = [False, False, True] + [False for _ in range(3, num_cards + 1)]
    is_fetch = [False for _ in range(num_cards)]
    has_basic_land_types = is_land
    tf.compat.v1.disable_eager_execution()
    autoencoder = DraftBot(num_cards, embs, prob_to_play, land_requirements, card_colors, is_land,
                           is_fetch, has_basic_land_types)
    autoencoder.compile(optimizer='adam', loss=['binary_cross_entropy'])
    latest = tf.train.latest_checkpoint(output_dir)
    if latest is not None:
        autoencoder.load_weights(latest)
    # pdb.set_trace()
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=output_dir + '/model',
        monitor='loss',
        verbose=True,
        save_best_only=True,
        mode='min',
        save_freq='epoch')
    autoencoder.fit(
        generator,
        epochs=epochs,
        callbacks=[cp_callback]
    )
    Path(f'{output_dir}/final').mkdir(parents=True, exist_ok=True)
    autoencoder.save(f'{output_dir}/final', save_format='tf')
