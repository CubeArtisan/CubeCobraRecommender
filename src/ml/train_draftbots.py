import datetime
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorboard.plugins.hparams import api as hp

if __name__ == "__main__":
    import os.path
    sys.path.append(os.path.dirname(sys.path[0]) + '../')

from .generator import DraftBotGenerator
from .draftbots import DraftBot, EMBEDS_DIM
from ..non_ml.parse_picks import COMPRESSION, FEATURES, MAX_IN_PACK, NUM_LAND_COMBS, PICK_SIGNATURE, load_picks

HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([128, 256, 512, 1024, 2048, 4096, 8192]))
HP_TEMPERATURE = hp.HParam('temperature', hp.RealInterval(1e-01, 1e+01))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.RealInterval(1e-06, 1e+01))

METRIC_ACCURACY = 'categorical_accuracy'

if __name__ == "__main__":
    args = sys.argv[1:]
    epochs = int(args[0])
    name = args[1]
    batch_size = int(args[2])
    temperature = float(args[3])
    learning_rate = float(args[4])
    debugging = len(args) > 5 and args[5] == 'debug'
    hparams = { HP_BATCH_SIZE: batch_size, HP_TEMPERATURE: temperature, HP_LEARNING_RATE: learning_rate }

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    debug_dir = "logs/debug/"
    if debugging:
        print('Enabling Debugging')
        tf.debugging.experimental.enable_dump_debug_info(
            debug_dir,
            tensor_debug_mode='FULL_HEALTH',
            circular_buffer_size=-1,
        )
    tf.config.optimizer.set_jit(True)

    print('Loading card data for seeding weights.')
    with open('data/intToCard.json', 'r') as cards_file:
        cards_json = json.load(cards_file)
        card_ratings = [0] + [10 ** ((c.get('elo', 1200) / 800) - 2) for c in cards_json]
        blank_embedding = [1 for _ in range(EMBEDS_DIM)]
        card_embeddings = [blank_embedding] + [c.get('embedding', blank_embedding)
                           if c.get('embedding', blank_embedding) is not None and len(c.get('embedding', blank_embedding)) == EMBEDS_DIM
                           else blank_embedding for c in cards_json]

    with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_BATCH_SIZE, HP_LEARNING_RATE, HP_TEMPERATURE],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
        )

    print('Creating the pick Datasets.')
    pick_cache_dir = Path('data/parsed_picks/')
    train_dataset = load_picks(pick_cache_dir / 'train', batch_size)
    test_dataset = load_picks(pick_cache_dir / 'test', batch_size)

    print('Loading DraftBot model.')
    output_dir = f'././ml_files/{name}/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print('Setting Up Model . . . \n')
    draftbots = DraftBot(card_ratings, card_embeddings, temperature)
    latest = tf.train.latest_checkpoint(output_dir)
    if latest is not None:
        print('Loading Checkpoint. Saved values are:')
        draftbots.load_weights(latest)
        for var in draftbots.variables:
            print(f'{var.name}: {var.numpy()}')
    draftbots.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-02),
        loss=['categorical_crossentropy'],
        metrics=['categorical_accuracy', tf.keras.metrics.TopKCategoricalAccuracy(3)]
    )

    print('Starting training')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=output_dir + 'model',
        monitor='loss',
        verbose=True,
        save_best_only=True,
        mode='min',
        save_freq='epoch')
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    hp_callback = hp.KerasCallback(log_dir, hparams)
    draftbots.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        callbacks=[cp_callback, tb_callback, hp_callback]
    )

    print('Saving final model.')
    Path(f'{output_dir}/final').mkdir(parents=True, exist_ok=True)
    draftbots.save(f'{output_dir}/final', save_format='tf')
