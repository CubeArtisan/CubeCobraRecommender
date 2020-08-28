# enable sibling imports
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    from os.path import dirname as dir
    sys.path.append(dir(sys.path[0]))

from generator import CardDataGenerator
from model import CardEncoderWrapper, CardEncoder
from metrics import FilteredBinaryAccuracy, TripletFilteredAccuracy, ContrastiveFilteredAccuracy
from losses import TripletLoss, ContrastiveLoss

if __name__ == "__main__":
    args = sys.argv[1:]

    epochs = int(args[0])
    batch_size = int(args[1])
    name = args[2]
    walk_len = int(args[3])
    example_count = int(args[4])
    temperature = float(args[5])
    margin = float(args[6])

    map_file = '././data/maps/nameToId.json'
    folder = "././data/cube/"

    print('Loading Cube Data . . .\n')

    int_to_card = json.load(open('././output/int_to_card.json', 'rb'))
    int_to_card = {int(k): v for k, v in int_to_card.items()}  # if int(k) < 1000}
    card_to_int = {v: k for k, v in int_to_card.items()}
    num_cards = len(int_to_card)

    print('Setting up Generator . . .\n')
    output_dir = f'././ml_files/{name}/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open('cards.json', 'r', encoding="utf-8") as cardsjson:
        cards = json.load(cardsjson)
        cards = [cards.get(int_to_card[i], "") for i in range(num_cards)]
        for card in cards:
            if 'typeLine' in card:
                if isinstance(card['typeLine'], str):
                    card['type'] = card['typeLine'].replace('— ', '').split(' ')
                else:
                    for key, value in card['typeLine'][0].items():
                        if isinstance(value, dict):
                            value = value['and']
                        else:
                            value = [value]
                        card[key] = value
                del card['typeLine']
            if "otherParses" in card:
                del card["otherParses"]

    generator = CardDataGenerator(
        walk_len,
        cards,
        batch_size=batch_size,
        example_count=example_count,
        data_path='./output',
    )
    print('Setting Up Model . . . \n')
    autoencoder = CardEncoder("card_encoder", generator.vocab_dict, generator.max_paths,
                              generator.max_path_length, example_count + 1,
                              generator.continuous_features_count,
                              generator.categorical_features_count)
    assert example_count == 2
    # autoencoder.compile(
    #     optimizer='adam',
    #     loss=[TripletLoss(margin)],
    #     loss_weights=[1.0],
    #     metrics=[TripletFilteredAccuracy(None, margin, "accuracy"),
    #              TripletFilteredAccuracy(True, margin, "true_accuracy"),
    #              TripletFilteredAccuracy(False, margin, "false_accuracy")]
    # )
    autoencoder.compile(
        optimizer='adam',
        loss=[ContrastiveLoss(example_count, temperature)],
        metrics=[ContrastiveFilteredAccuracy(None, example_count, margin, "accuracy"),
                 ContrastiveFilteredAccuracy(True, example_count, margin, "true_accuracy"),
                 ContrastiveFilteredAccuracy(False, example_count, margin, "false_accuracy")]
    )
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
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                           patience=10,
                                                           min_delta=0.002)
    autoencoder.fit(
        generator,
        epochs=epochs,
        callbacks=[cp_callback]  #, early_stop_callback]
    )
    Path(f'{output_dir}/final').mkdir(parents=True, exist_ok=True)
    autoencoder.save(f'{output_dir}/final', save_format='tf')