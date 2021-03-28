import datetime
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorboard.plugins.hparams import api as hp
from tensorboard.plugins import projector

if __name__ == "__main__":
    import os.path
    sys.path.append(os.path.dirname(sys.path[0]) + '../')

from .draftbots import DraftBot
from ..non_ml.parse_picks import COMPRESSION, load_picks, MAX_IN_PACK

EMBEDS_DIM = 64

HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([128, 256, 512, 1024, 2048, 4096, 8192]))
HP_TEMPERATURE = hp.HParam('temperature', hp.RealInterval(1e-01, 1e+02))
HP_INTERNAL_SYNERGY_DROPOUT = hp.HParam('internal_synergy_dropout_rate', hp.RealInterval(0.0, 1.0))
HP_PICK_SYNERGY_DROPOUT = hp.HParam('pick_synergy_dropout_rate', hp.RealInterval(0.0, 1.0))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.RealInterval(1e-06, 1e+01))

if __name__ == "__main__":
    args = sys.argv[1:]
    epochs = int(args[0])
    name = args[1]
    batch_size = int(args[2])
    temperature = float(args[3])
    internal_synergy_dropout_rate = float(args[4])
    pick_synergy_dropout_rate = float(args[5])
    learning_rate = float(args[6])
    auto16 = False
    floattype = tf.float32
    if args[7] == '16':
        floattype = tf.float16
    elif args[7] == '16rw':
        auto16 = True
    floattype = tf.float16 if args[7] == '16' else tf.float32
    debugging = len(args) > 8 and args[8] == 'debug'
    no_xla = debugging or len(args) > 8 and args[8] == 'noxla'
    hparams = { HP_BATCH_SIZE: batch_size, HP_TEMPERATURE: temperature, HP_LEARNING_RATE: learning_rate,
                HP_INTERNAL_SYNERGY_DROPOUT: internal_synergy_dropout_rate, HP_PICK_SYNERGY_DROPOUT: pick_synergy_dropout_rate }

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    Path(log_dir).mkdir(exist_ok=False, parents=True)
    debug_dir = "logs/debug/"
    if debugging:
        print('Enabling Debugging')
        tf.debugging.experimental.enable_dump_debug_info(
            debug_dir,
            tensor_debug_mode='FULL_HEALTH',
            circular_buffer_size=-1,
        )
    else:
        tf.config.experimental.enable_mlir_graph_optimization()
        # tf.config.experimental.enable_mlir_bridge()
    projector_config = projector.ProjectorConfig()
    embedding = projector_config.embeddings.add()
    embedding.tensor_name = "card_embeddings/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    embedding = projector_config.embeddings.add()
    embedding.tensor_name = "card_rating_embeddings/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(log_dir, projector_config)

    if floattype == tf.float16:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    tf.config.optimizer.set_jit(not (debugging or no_xla))
    tf.config.optimizer.set_experimental_options=({
        'layout_optimizer': True,
        'constant_folding': True,
        'shape_optimization': True,
        'remapping': True,
        'arithmetic_optimization': True,
        'dependency_optimization': True,
        'loop_optimization': True,
        'function_optimization': True,
        'debug_stripper': not debugging,
        'disable_model_pruning': False,
        'scoped_allocator_optimization': True,
        'pin_to_host_optimization': True,
        'implementation_selector': True,
        'auto_mixed_precision': True,
        'disable_meta_optimizer': False,
        'min_graph_nodes': 1,
    })
    tf.config.threading.set_intra_op_parallelism_threads(32)
    tf.config.threading.set_inter_op_parallelism_threads(32)

    print('Loading card data for seeding weights.')
    with open('data/intToCard.json', 'r') as cards_file:
        cards_json = json.load(cards_file)
        card_ratings = [0] + [10 ** ((c.get('elo', 1200) / 800) - 2) for c in cards_json]
        blank_embedding = [1 for _ in range(EMBEDS_DIM)]
        card_embeddings = [blank_embedding] + [c.get('embedding', blank_embedding)
                           if c.get('embedding', blank_embedding) is not None and len(c.get('embedding', blank_embedding)) == EMBEDS_DIM
                           else blank_embedding for c in cards_json]
        card_names = [''] + [c['name'] for c in cards_json]

    metadata = os.path.join(log_dir, 'metadata.tsv')
    with open(metadata, "w") as f:
        for cname in card_names:
            f.write(f'"{cname}"\n')

    print('Creating the pick Datasets.')
    pick_cache_dir = Path('data/parsed_picks/')
    train_dataset = load_picks(pick_cache_dir / 'train', batch_size)
    train_ys = np.zeros((train_dataset.cardinality(), batch_size, MAX_IN_PACK))
    train_ys[:,:,0] = 1
    train_ys = tf.data.Dataset.from_tensor_slices(tf.cast(train_ys, dtype=tf.float32))
    # test_dataset = load_picks(pick_cache_dir / 'test', batch_size)

    print('Loading DraftBot model.')
    output_dir = f'././ml_files/{name}/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print('Setting Up Model . . . \n')
    draftbots = DraftBot(card_ratings, card_embeddings, temperature, internal_synergy_dropout_rate,
                         pick_synergy_dropout_rate, floattype, debugging or no_xla)
    latest = tf.train.latest_checkpoint(output_dir)
    if latest is not None:
        print('Loading Checkpoint. Saved values are:')
        draftbots.load_weights(latest)
        for var in draftbots.variables:
            print(f'{var.name}: {var.numpy()}')
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if floattype == tf.float16:
        print("WARNING 16 bit mode is currently giving very poor performance and should not be used.")
        opt = tf.keras.mixed_precision.LossScaleOptimizer(
            opt, dynamic_growth_steps = 100, initial_scale=2**16,
        )
    elif auto16:
        tf.compat.v1.mixed_precision.enable_mixed_precision_graph_rewrite(opt)
    draftbots.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=(tf.keras.metrics.TopKCategoricalAccuracy(3, name='top_3_accuracy'), 'categorical_accuracy'),
        # run_eagerly=debugging,
    )

    print('Starting training')
    mcp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=output_dir + 'model',
        monitor='loss',
        verbose=False,
        save_best_only=True,
        mode='min',
        save_freq='epoch')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=log_dir + '/model-{epoch:04d}.ckpt',
        monitor='loss',
        verbose=False,
        save_best_only=True,
        mode='min',
        save_freq='epoch')
    nan_callback = tf.keras.callbacks.TerminateOnNaN()
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='categorical_accuracy', patience=8,
                                                   mode='max', restore_best_weights=True, verbose=True)
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='categorical_accuracy', factor=0.5, mode='max',
                                                       patience=2, min_delta=1/(2**8), cooldown=1, min_lr=1/(2**13), verbose=True)
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True,
                                                 update_freq=20, embeddings_freq=0, profile_batch=(602, 627))
    hp_callback = hp.KerasCallback(log_dir, hparams)
    draftbots.fit(
        tf.data.Dataset.zip((train_dataset, train_ys)),
        # validation_data=test_dataset,
        epochs=epochs,
        callbacks=[mcp_callback, cp_callback, nan_callback, es_callback, lr_callback,
                   tb_callback, hp_callback],
    )

    print('Saving final model.')
    Path(f'{output_dir}/final').mkdir(parents=True, exist_ok=True)
    draftbots.save(f'{output_dir}/final', save_format='tf')
