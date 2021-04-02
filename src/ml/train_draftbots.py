import argparse
import datetime
import json
import os
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorboard.plugins.hparams import api as hp
from tensorboard.plugins import projector

from .draftbots import DraftBot
from .losses import CustomCrossEntropy
from ..non_ml.parse_picks import COMPRESSION, load_picks, MAX_IN_PACK

BATCH_CHOICES = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 64532)
EMBED_DIMS_CHOICES = (2, 4, 8, 16, 32, 64, 128, 256, 512)

HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete(BATCH_CHOICES))
HP_TEMPERATURE = hp.HParam('temperature', hp.RealInterval(1e-05, 1e+05))
HP_MAX_TEMPERATURE = hp.HParam('max_temperature', hp.RealInterval(1e-05, 1e+05))
HP_EMBED_DIMS = hp.HParam('embedding_dimensions', hp.Discrete(EMBED_DIMS_CHOICES))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.RealInterval(1e-06, 1e+05))


class TensorBoardFix(tf.keras.callbacks.TensorBoard):
    """
    This fixes incorrect step values when using the TensorBoard callback with custom summary ops
    """

    def on_train_begin(self, *args, **kwargs):
        super(TensorBoardFix, self).on_train_begin(*args, **kwargs)
        tf.summary.experimental.set_step(self._train_step)


    def on_test_begin(self, *args, **kwargs):
        super(TensorBoardFix, self).on_test_begin(*args, **kwargs)
        tf.summary.experimental.set_step(self._val_step)


class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end
        
    def __str__(self):
        return f'values in the inclusive range [{self.start}, {self.end}]'
        
    def __repr__(self):
        return str(self)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', type=int, required=True, help="The maximum number of epochs to train for")
    parser.add_argument('--name', '-o', '-n', type=str, required=True, help="The name to save this model under.")
    parser.add_argument('--batch-size', '-b', type=int, choices=BATCH_CHOICES, required=True, help="The batch size to train over.")
    parser.add_argument('--temperature', '-t', type=float, default=100, choices=[Range(1e-05, 1e+05)], help="The initial temperature (scaling factor) for the scores.")
    parser.add_argument('--max-temperature', type=float, default=25000, choices=[Range(1e-05, 1e+05)], help="The maximum temperature (scaling factor for scores) that should be used.")
    parser.add_argument('--learning-rate', '-l', type=float, default=0.25, choices=[Range(1e-05, 1e+05)], help="The initial learning rate to train with.")
    parser.add_argument('--embed-dims', '-d', type=int, default=16, choices=EMBED_DIMS_CHOICES, help="The number of dimensions to use for card embeddings.")
    float_type_group = parser.add_mutually_exclusive_group()
    float_type_group.add_argument('-16', dest='float_type', const=tf.float16, action='store_const', help='Use 16 bit numbers throughout the model.')
    float_type_group.add_argument('--auto16', '-16rw', action='store_true', help='Automatically rewrite some operations to use 16 bit numbers.')
    float_type_group.add_argument('--keras16', '-16k', action='store_true', help='Have Keras automatically convert the synergy calculations to 16 bit.')
    float_type_group.add_argument('-32', dest='float_type', const=tf.float32, action='store_const', help='Use 32 bit numbers (the default) throughout the model.')
    float_type_group.add_argument('-64', dest='float_type', const=tf.float64, action='store_const', help='Use 64 bit numbers throughout the model.')
    parser.add_argument('--debug', action='store_true', help='Enable debug dumping of tensor stats.')
    xla_group = parser.add_mutually_exclusive_group()
    xla_group.add_argument('--xla', action='store_true', dest='use_xla', help='Enable using xla to optimize the model (the default).')
    xla_group.add_argument('--no-xla', action='store_false', dest='use_xla', help='Disable using xla to optimize the model.')
    parser.add_argument('--compressed', action='store_true', help='Use the compressed version of the data to reduce disk usage.')
    parser.add_argument('--num-workers', '-j', type=int, default=32, choices=[Range(1, 512)], help='The number of threads to use for loading data from disk.')
    parser.add_argument('--mlir', action='store_true', help='Enable MLIR passes on the data (EXPERIMENTAL).')
    parser.add_argument('--ragged', action='store_true', help='Enable loading from ragged datasets instead of dense.')
    profile_group = parser.add_mutually_exclusive_group()
    profile_group.add_argument('--profile', action='store_true', help='Enable profiling a range of batches from the first epoch.')
    profile_group.add_argument('--no-profile', action='store_false', dest='profile', help='Disable profiling a range of batches from the first epoch (the default).')
    parser.set_defaults(float_type=tf.float32, use_xla=True, profile=False)    
    args = parser.parse_args()
    hparams = { HP_BATCH_SIZE: args.batch_size, HP_TEMPERATURE: args.temperature, HP_LEARNING_RATE: args.learning_rate,
                HP_MAX_TEMPERATURE: args.max_temperature, HP_EMBED_DIMS: args.embed_dims }

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.debug:
        log_dir = "logs/debug/"
        print('Enabling Debugging')
        tf.debugging.experimental.enable_dump_debug_info(
            log_dir,
            tensor_debug_mode='FULL_HEALTH',
            circular_buffer_size=-1,
            tensor_dtypes=[args.float_type],
            # op_regex="(?!^(Placeholder|Constant)$)"
        )
    if args.mlir:
        tf.config.experimental.enable_mlir_graph_optimization()
        tf.config.experimental.enable_mlir_bridge()
    
    Path(log_dir).mkdir(exist_ok=True, parents=True)

    if args.keras16 or args.float_type == tf.float16:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print('use_xla', args.use_xla)
    tf.config.optimizer.set_jit(args.use_xla)
    tf.config.optimizer.set_experimental_options=({
        'layout_optimizer': True,
        'constant_folding': True,
        'shape_optimization': True,
        'remapping': True,
        'arithmetic_optimization': True,
        'dependency_optimization': True,
        'loop_optimization': True,
        'function_optimization': True,
        'debug_stripper': not args.debug,
        'disable_model_pruning': False,
        'scoped_allocator_optimization': True,
        'pin_to_host_optimization': True,
        'implementation_selector': True,
        'auto_mixed_precision': True,
        'disable_meta_optimizer': False,
        'min_graph_nodes': 1,
    })
    tf.config.threading.set_intra_op_parallelism_threads(128)
    tf.config.threading.set_inter_op_parallelism_threads(128)

    print('Loading card data for seeding weights.')
    with open('data/intToCard.json', 'r') as cards_file:
        cards_json = json.load(cards_file)
        card_ratings = [0] + [10 ** ((c.get('elo', 1200) / 800) - 2) for c in cards_json]
        blank_embedding = [1 for _ in range(64)]
        card_names = [''] + [c['name'] for c in cards_json]

    metadata = os.path.join(log_dir, 'metadata.tsv')
    with open(metadata, "w") as f:
        for cname in card_names:
            f.write(f'"{cname}"\n')
    projector_config = projector.ProjectorConfig()
    embedding = projector_config.embeddings.add()
    embedding.tensor_name = "card_embeddings/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = metadata
    embedding = projector_config.embeddings.add()
    embedding.tensor_name = "card_ratings/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = metadata
    projector.visualize_embeddings(log_dir, projector_config)
    
    print('Creating the pick Datasets.')
    # full_dataset = load_picks('full', args.batch_size, num_workers=args.num_workers, compressed=args.compressed, ragged=args.ragged)
    train_dataset = load_picks('train', args.batch_size, num_workers=args.num_workers, compressed=args.compressed, ragged=args.ragged, pad=True)
    test_dataset = load_picks('test', args.batch_size, num_workers=args.num_workers, compressed=args.compressed, ragged=args.ragged, pad=True)
    
    print('Loading DraftBot model.')
    output_dir = f'././ml_files/{args.name}/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    draftbots = DraftBot(len(card_ratings), args.temperature,
                         args.float_type, embed_dims=args.embed_dims, use_xla=args.use_xla)
    # latest = tf.train.latest_checkpoint(output_dir)
    # if latest is not None:
        # print('Loading Checkpoint. Saved values are:')
        # draftbots.load_weights(latest)
        # draftbots.temperature.assign(args.temperature)
        # for var in draftbots.variables:
            # print(f'{var.name}: {var.numpy()}')
    # opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    opt = tfa.optimizers.LazyAdam(learning_rate=args.learning_rate)
    if args.float_type == tf.float16 or args.keras16:
        # opt = tf.keras.mixed_precision.LossScaleOptimizer(opt, dynamic=False, initial_scale=8192)
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt, dynamic_growth_steps=5, initial_scale=2**15)
    if args.auto16:
        print("WARNING 16 bit rewrite mode can cause numerical instabilities.")
        tf.compat.v1.mixed_precision.enable_mixed_precision_graph_rewrite(opt)
    draftbots.compile(
        optimizer=opt,
        loss=tf.keras.losses.CategoricalCrossentropy(name='crossentropy'),
        metrics=(tf.keras.metrics.TopKCategoricalAccuracy(3, name='top_3_accuracy'), 'categorical_accuracy'),
        # run_eagerly=debugging,
    )

    print('Starting training')
    callbacks = []
    mcp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=output_dir + 'model',
        monitor='loss',
        verbose=False,
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        save_freq='epoch')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=log_dir + '/model-{epoch:04d}.ckpt',
        monitor='loss',
        verbose=False,
        save_best_only=False,
        save_weights_only=True,
        mode='min',
        save_freq='epoch')
    # if not args.debug:
        # callbacks.append(mcp_callback)
        # callbacks.append(cp_callback)
    nan_callback = tf.keras.callbacks.TerminateOnNaN()
    callbacks.append(nan_callback)
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=16,
                                                   mode='max', restore_best_weights=True, verbose=True)
    # callbacks.append(es_callback)
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, mode='max',
                                                       patience=12, min_delta=1/(2**12), cooldown=6, min_lr=1/(2**17),
                                                       verbose=True)
    callbacks.append(lr_callback)
    print(train_dataset.cardinality())
    tb_callback = TensorBoardFix(log_dir=log_dir, histogram_freq=1, write_graph=False,
                                 update_freq=train_dataset.cardinality().numpy() // 10, embeddings_freq=1,
                                 profile_batch=0 if args.debug or not args.profile
                                                 else (train_dataset.cardinality().numpy() * 1.3 + 1,
                                                       1.4 * train_dataset.cardinality().numpy() + 1))
    callbacks.append(tb_callback)
    hp_callback = hp.KerasCallback(log_dir, hparams)
    # callbacks.append(hp_callback)
    scale_factor = tf.cast(tf.math.pow(args.max_temperature / args.temperature, 1 / (args.epochs - 2)), dtype=tf.float64)
    temp_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: draftbots.temperature.assign(draftbots.temperature * scale_factor))
    callbacks.append(temp_callback)
    draftbots.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    if not args.debug:
        print('Saving final model.')
        Path(f'{output_dir}/final').mkdir(parents=True, exist_ok=True)
        draftbots.save(f'{output_dir}/final', save_format='tf')