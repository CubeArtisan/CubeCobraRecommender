import argparse
import datetime
import json
import locale
import logging
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorboard.plugins.hparams import api as hp
from tensorboard.plugins import projector

from src.generated.draftbot_generator import DraftPickGenerator
from src.ml.callbacks import DynamicLearningRateCallback
from src.ml.draftbots import DraftBot

locale.setlocale(locale.LC_ALL, '')

BATCH_CHOICES = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 64532)
EMBED_DIMS_CHOICES = (2, 4, 8, 16, 32, 64, 128, 256, 512)
NUM_HEAD_CHOICES = tuple(2 ** i for i in range(6))

HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete(BATCH_CHOICES))
HP_EMBED_DIMS = hp.HParam('embedding_dimensions', hp.Discrete(EMBED_DIMS_CHOICES))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.RealInterval(1e-06, 1e+05))
HP_RATING_L2_WEGIHT = hp.HParam('l2_loss_weight', hp.RealInterval(0.0, 1.0))
HP_NUM_HEADS = hp.HParam('num_heads', hp.Discrete(NUM_HEAD_CHOICES))


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


class GeneratorWrapper(tf.keras.utils.Sequence):
    def __init__(self, generator, num_cards, training=False):
        self.generator = generator
        self.num_cards = num_cards
        self.training = training

    def __getitem__(self, item):
        item = self.generator.__getitem__(item)
        if False:
            num_seen = item[2]
            num_picked = item[4]
            for i, pair in enumerate(zip(num_seen, num_picked)):
                seen_count, picked_count = pair
                seen_count = int(seen_count)
                picked_count = int(picked_count)
                idxs = list(range(seen_count))
                random.shuffle(idxs)
                for idx in idxs[:seen_count // 2]:
                    item[1][i][idx] = random.randrange(self.num_cards)
                idxs = list(range(picked_count))
                random.shuffle(idxs)
                for idx in idxs[:picked_count // 2]:
                    item[3][i][idx] = random.randrange(self.num_cards)
        return (item,)

    def __len__(self):
        return len(self.generator)
        # return 8192

    def on_epoch_end(self):
        return self.generator.on_epoch_end()


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

    
class ExponentialCyclingLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, maximal_learning_rate=32e-03, minimal_learning_rate=1e-03, decrease_steps=61440, increase_steps=8192):
        self.maximal_learning_rate = tf.constant(maximal_learning_rate, dtype=tf.float32)
        self.minimal_learning_rate = tf.constant(minimal_learning_rate, dtype=tf.float32)
        self.cycle_steps = decrease_steps + increase_steps
        self.decreasing_rate = tf.constant((minimal_learning_rate / maximal_learning_rate) ** (1 / decrease_steps), dtype=tf.float32)
        self.increasing_rate = tf.constant((maximal_learning_rate / minimal_learning_rate) ** (1 / increase_steps), dtype=tf.float32)
        self.increase_steps = increase_steps
        self.decrease_steps = decrease_steps
        
        
    def __call__(self, step):
        with tf.name_scope("CyclicalLearningRate"):
            cycle_pos = step % self.cycle_steps
            lr = tf.cond(cycle_pos >= self.increase_steps,
                         lambda: self.maximal_learning_rate * (self.decreasing_rate ** tf.cast((cycle_pos - self.increase_steps) % self.decrease_steps, dtype=tf.float32)),
                         lambda: self.minimal_learning_rate * (self.increasing_rate ** tf.cast(cycle_pos, dtype=tf.float32)))
            tf.summary.scalar('learning_rate', lr)
            return lr


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', type=int, required=True, help="The maximum number of epochs to train for")
    parser.add_argument('--name', '-o', '-n', type=str, required=True, help="The name to save this model under.")
    parser.add_argument('--batch-size', '-b', type=int, required=True, help="The batch size to train over.")
    parser.add_argument('--learning-rate', '-l', type=float, default=None, choices=[Range(1e-06, 1e+05)], help="The initial learning rate to train with.")
    parser.add_argument('--embed-dims', '-d', type=int, default=16, choices=EMBED_DIMS_CHOICES, help="The number of dimensions to use for card embeddings.")
    parser.add_argument('--seen-dims', type=int, default=16, choices=EMBED_DIMS_CHOICES, help='The number of dimensions to use for seen card embeddings.')
    parser.add_argument('--picked-dims', type=int, default=16, choices=EMBED_DIMS_CHOICES, help='The number of dimensions to use for picked card embeddings.')
    parser.add_argument('--rating-l2-weight', type=float, default=0.0, help='The relative weight of the l2 regularization on the rating logits that pulls them towards 0.')
    parser.add_argument('--rating-l1-weight', type=float, default=0.0, help='The relative weight of the l1 regularization on the ratings that pulls them towards 0.')
    parser.add_argument('--dropout-picked', type=float, default=0.0, help='The percent of cards to drop from picked when calculating the pool embedding.')
    parser.add_argument('--dropout-seen', type=float, default=0.0, help='The percent of cards to drop from picked when calculating the seen embedding.')
    parser.add_argument('--dropout-dense', type=float, default=0.0, help='The percent of values to drop from the dense layers when calculating pool/seen embeddings.')
    parser.add_argument('--seed', type=int, default=37, help='The random seed to initialize things with to improve reproducibility.')
    parser.add_argument('--runtime', type=int, default=None, help='Number of minutes to train for.')
    parser.add_argument('--oracle-stddev-weight', type=float, default=0.0, help='The relative weight of the loss based on stddev of the oracles. This encourages oracles to have diverse values.')
    float_type_group = parser.add_mutually_exclusive_group()
    float_type_group.add_argument('-16', dest='float_type', const=tf.float16, action='store_const', help='Use 16 bit numbers throughout the model.')
    float_type_group.add_argument('--auto16', '-16rw', action='store_true', help='Automatically rewrite some operations to use 16 bit numbers.')
    float_type_group.add_argument('--keras16', '-16k', action='store_true', help='Have Keras automatically convert the synergy calculations to 16 bit.')
    float_type_group.add_argument('-32', dest='float_type', const=tf.float32, action='store_const', help='Use 32 bit numbers (the default) throughout the model.')
    float_type_group.add_argument('-64', dest='float_type', const=tf.float64, action='store_const', help='Use 64 bit numbers throughout the model.')
    xla_group = parser.add_mutually_exclusive_group()
    xla_group.add_argument('--xla', action='store_true', dest='use_xla', help='Enable using xla to optimize the model (the default).')
    xla_group.add_argument('--no-xla', action='store_false', dest='use_xla', help='Disable using xla to optimize the model.')
    parser.add_argument('--debug', action='store_true', help='Enable debug dumping of tensor stats.')
    parser.add_argument('--mlir', action='store_true', help='Enable MLIR passes on the data (EXPERIMENTAL).')
    parser.add_argument('--profile', action='store_true', help='Enable profiling a range of batches from the first epoch.')
    parser.add_argument('--num-workers', '-j', type=int, default=32, choices=[Range(1, 128)], help='The number of threads to use for loading data from disk.')
    parser.add_argument('--starting-step', type=int, default=0, help='The starting step number in batches.')
    parser.set_defaults(float_type=tf.float32, use_xla=True)
    args = parser.parse_args()

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
        'disable_meta_optimizer': False,
        'min_graph_nodes': 1,
    })
    tf.config.threading.set_intra_op_parallelism_threads(32)
    tf.config.threading.set_inter_op_parallelism_threads(32)

    print('Loading card data for seeding weights.')
    with open('data/maps/int_to_card.json', 'r') as cards_file:
        cards_json = json.load(cards_file)
        card_ratings = [-1] + [(c.get('elo', 1200) / 1200) - 1  for c in cards_json]
        blank_embedding = [1 for _ in range(64)]
        card_names = [''] + [c['name'] for c in cards_json]

    metadata = os.path.join(log_dir, 'metadata.tsv')
    with open(metadata, "w") as f:
        f.write('index\tName\tColors\tType\n')
        f.write('0\t"PlaceholderForTraining"\tN/A\tN/A\n')
        for i, card in enumerate(cards_json):
            f.write(f'{i+1}\t"{card["name"]}"\t{"".join(sorted(card.get("color_identity")))}\t{card.get("type")}\n')

    print('Creating the pick Datasets.')
    pick_generator_train = DraftPickGenerator(args.batch_size, args.num_workers, args.seed, "data/parsed_picks/training/")
    print(f"There are {len(pick_generator_train):,} training batches")
    pick_generator_test = DraftPickGenerator(args.batch_size, args.num_workers, args.seed, "data/parsed_picks/validation/")
    print(f"There are {len(pick_generator_test):,} validation batches")
    print('Loading DraftBot model.')
    output_dir = f'././ml_files/{args.name}/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    num_batches = len(pick_generator_train)
    tensorboard_period = num_batches // 8
    draftbots = DraftBot(num_cards=len(card_ratings), embed_dims=args.embed_dims, summary_period=tensorboard_period * 8,
                         rating_l1_loss_weight=args.rating_l1_weight, rating_l2_loss_weight=args.rating_l2_weight,
                         oracle_stddev_loss_weight=args.oracle_stddev_weight, seen_dims=args.seen_dims, picked_dims=args.picked_dims,
                         dropout_picked_rate=args.dropout_picked, dropout_seen_rate=args.dropout_seen, dropout_dense_rate=args.dropout_dense)
    latest = tf.train.latest_checkpoint(output_dir)
    learning_rate = args.learning_rate or 0.001
    learning_rate = ExponentialCyclingLearningRate(maximal_learning_rate=learning_rate, minimal_learning_rate=learning_rate / 64,
                                                   decrease_steps=num_batches * (args.epochs - 1), increase_steps=num_batches)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # opt = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
    # opt = tfa.optimizers.LazyAdam(learning_rate=learning_rate)
    # opt = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)
    # opt = tfa.optimizers.NovoGrad(learning_rate=learning_rate)
    # opt = tfa.optimizers.LAMB(learning_rate=learning_rate)
    # opt = tfa.optimizers.Lookahead(opt, sync_period=16, slow_step_size=0.5)
    if args.float_type == tf.float16:
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt, dynamic_growth_steps=num_batches // 128)
    if args.auto16:
        print("WARNING 16 bit rewrite mode can cause numerical instabilities.")
        tf.compat.v1.mixed_precision.enable_mixed_precision_graph_rewrite(opt)
    if latest is not None:
        print('Loading Checkpoint.')
        # draftbots.load_weights(latest)
        # draftbots.compile()
        # draftbots.save_weights(latest)
        # draftbots = DraftBot(card_ratings, random_embeddings, args.batch_size, l1_loss_weight=args.l1_weight,
        #                      l2_loss_weight=args.l2_weight, embed_dims=args.embed_dims, summary_period=tensorboard_period * 16,
        #                      num_heads=args.num_heads, name='DraftBots')
        draftbots.load_weights(latest)
        # if args.learning_rate:
            # tf.keras.backend.set_value(draftbots.optimizer.lr, args.learning_rate)
    # tf.keras.backend.set_value(draftbots.oracle_weights, [[[5, 5, 5, 5, 5, 5] for _ in range(15)] for _ in range(3)])
    draftbots.compile(optimizer=opt)
    hparams = {
        HP_BATCH_SIZE: args.batch_size,
        HP_LEARNING_RATE: args.learning_rate or 0.001,
        HP_EMBED_DIMS: args.embed_dims, HP_RATING_L2_WEGIHT: args.rating_l2_weight,
    }

    print('Starting training')
    callbacks = []
    if not args.debug:
        mcp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=output_dir + 'model',
            monitor='accuracy',
            verbose=False,
            save_best_only=True,
            save_weights_only=True,
            mode='max',
            save_freq='epoch')
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=log_dir + '/model-{epoch:04d}.ckpt',
            monitor='loss',
            verbose=False,
            save_best_only=False,
            save_weights_only=True,
            mode='min',
            save_freq='epoch')
        callbacks.append(mcp_callback)
        callbacks.append(cp_callback)
    if args.runtime:
        ts_callback = tfa.callbacks.TimeStopping(args.runtime * 60)
        callbacks.append(ts_callback)
    nan_callback = tf.keras.callbacks.TerminateOnNaN()
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=16,
                                                   mode='max', restore_best_weights=True, verbose=True)
    tb_callback = TensorBoardFix(log_dir=log_dir, histogram_freq=1, write_graph=True,
                                 update_freq=tensorboard_period, embeddings_freq=None,
                                 profile_batch=0 if args.debug or not args.profile else (5 * num_batches // 4, 5 * num_batches // 4 + 64))
    hp_callback = hp.KerasCallback(log_dir, hparams)
    callbacks.append(nan_callback)
    # callbacks.append(es_callback)
    callbacks.append(tb_callback)
    callbacks.append(hp_callback)
    tf.summary.experimental.set_step(args.starting_step)
    with pick_generator_train:
        with pick_generator_test:
            draftbots.fit(
                GeneratorWrapper(pick_generator_train, len(card_ratings), True),
                validation_data=GeneratorWrapper(pick_generator_test, len(card_ratings), False),
                epochs=args.epochs,
                callbacks=callbacks,
                verbose=1
            )
    if not args.debug:
        print('Saving final model.')
        Path(f'{output_dir}/final').mkdir(parents=True, exist_ok=True)
        draftbots.save(f'{output_dir}/final', save_format='tf')