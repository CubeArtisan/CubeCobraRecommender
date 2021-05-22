import contextlib
import json
import os
import random
from datetime import datetime
from pathlib import Path

import scipy.sparse as sp
from tqdm.auto import tqdm
from tqdm.keras import TqdmCallback


def is_valid_cube(cube):
    return cube['numDecks'] > 0 and len(set(cube['cards'])) >= 120


if __name__ == '__main__':
    import argparse
    import locale

    import numpy as np
    import src.non_ml.utils as utils
    import tensorflow as tf

    from src.ml.metapath import MetapathRecommender
    from src.generated.generator import GeneratorWithoutAdj

    locale.setlocale(locale.LC_ALL, '')

    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        print('Could not enable dynamic memory growth on the GPU.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', type=int, help='The number of epochs to train for.')
    parser.add_argument('--batch-size', type=int, choices=[2**i for i in range(0, 15)], help='The number of cubes/decks to process at a time.')
    parser.add_argument('--embed-dims', type=int, default=128, choices=[2**i for i in range(0, 12)], help='The number of dimensions for the card ebeddings.')
    parser.add_argument('--metapath-dims', type=int, default=64, choices=[2**i for i in range(0, 10)], help='The number of dimensions for the metapath specific views of the pool embeddings.')
    parser.add_argument('--num-heads', type=int, default=16, choices=[2**i for i in range(0, 9)], help='The number of attention heads to use for combining the metapaths.')
    parser.add_argument('--dropout', type=float, default=0.2, help='The percentage of dropout in the attention layer over paths, only applies for training.')
    parser.add_argument('--margin', type=float, default=1.0, help='The margin to use for scaling things away from -1.')
    parser.add_argument('--name', '-n', '-o', type=str, help='The folder under ml_files to save the model in.')
    parser.add_argument('--noise', type=float, default=0.5, help='The mean number of random swaps to make per cube.')
    parser.add_argument('--noise-stddev', type=float, default=0.1, help="The standard deviation of the amount of noise to apply.")
    parser.add_argument('--decks-weight', type=float, default=1.0, help="The relative weight of rebuilding decks to reconstructing cubes.")
    parser.add_argument('--l1-weight', type=float, default=0.1, help="The relative weight of the l1 regularization to the cube reconstruction.")
    parser.add_argument('--l2-weight', type=float, default=0.5, help="The relative weight of the l2 regularization to the cube reconstruction.")
    parser.add_argument('--plateau-patience', type=int, default=32, help='The number of epochs without improvement before halving the learning rate.')
    parser.add_argument('--learning-rate', type=float, default=1e-04, help="The initial learning rate.")
    parser.add_argument('--seed', type=int, default=37, help="A random seed to provide reproducible runs.")
    parser.add_argument('--profile', action='store_true', help='Run profiling on part of the second batch to analyze performance.')
    parser.add_argument('--debug', action='store_true', help='Enable dumping debug information to logs/debug.')
    parser.add_argument('--num-workers', '-j', type=int, default=1, help='Number of simulataneous workers to run to generate the data.')
    parser.add_argument('--xla', action='store_true', help='Use the XLA optimizer on the model.')
    parser.add_argument('--mlir', action='store_true', help='Highly experimental option to use the MLIR optimizer in tensorflow.')
    precision_group = parser.add_mutually_exclusive_group()
    precision_group.add_argument('--mixed', action='store_const', dest='dtype_policy', const='mixed_float16',
                                 help='Enable the automatic mixed-precision support in Keras.')
    precision_group.add_argument('--mixed-bfloat', action='store_const', dest='dtype_policy', const='mixed_bfloat16',
                                 help='Enable the automatic mixed-precision support in Keras with the bfloat16 type.')
    precision_group.add_argument('-16', action='store_const', dest='dtype_policy', const='float16',
                                 help='Use 16 bit variables and operations. This will likely result in numerical instability.')
    precision_group.add_argument('-32', action='store_const', dest='dtype_policy', const='float32',
                                 help='Use 32 bit variables and operations.')
    precision_group.add_argument('-64', action='store_const', dest='dtype_policy', const='float64',
                                 help='Use 64 bit variables and operations.')
    parser.set_defaults(dtype_policy='float32')
    args = parser.parse_args()

    output = Path('ml_files') / args.name
    data = Path('data')
    maps = data / 'maps'
    int_to_card_filepath = maps / 'int_to_card.json'
    cube_folder = data / "cubes"
    decks_folder = data / "decks"
    log_dir = Path("logs/fit/") / datetime.now().strftime("%Y%m%d-%H%M%S")

    def load_adj_mtx():
        print('Loading Adjacency Matrix . . .\n')
        adj_mtx = np.load('data/adj_mtx.npy')
        return adj_mtx

    print('Loading card data and number of cubes.')
    with open(int_to_card_filepath, 'rb') as int_to_card_file:
        int_to_card = json.load(int_to_card_file)
    card_to_int = {v: i for i, v in enumerate(int_to_card)}
    num_cards = len(int_to_card)
    num_cubes = utils.get_num_objs(cube_folder, validation_func=is_valid_cube)
    print(f'There are {num_cubes} valid cubes.')

    def load_cubes():
        print('Loading Cube Data')
        cubes = utils.build_cubes(cube_folder, num_cubes, num_cards,
                                  validation_func=is_valid_cube)
        return cubes

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

    def reset_random_seeds(seed):
        # currently not used
        os.environ['PYTHONHASHSEED'] = str(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        print("Reset random seeds")

    reset_random_seeds(args.seed)

    if args.debug:
        log_dir = "logs/debug/"
        print('Enabling Debugging')
        tf.debugging.experimental.enable_dump_debug_info(
            log_dir,
            tensor_debug_mode='FULL_HEALTH',
            circular_buffer_size=-1,
            # op_regex=b"(?!^(Placeholder|Constant)$)"
        )

    tf.config.optimizer.set_jit(args.xla)
    if args.xla:
        def jit_scope():
            return tf.xla.experimental.jit_scope(compile_ops=True, separate_compiled_gradients=True)
    else:
        def jit_scope():
            return contextlib.nullcontext()
    if args.mlir:
        print('MLIR is very experimental currently and so may cause errors.')
        tf.config.experimental.enable_mlir_graph_optimization()
        tf.config.experimental.enable_mlir_bridge()
    tf.config.set_soft_device_placement(True)
    tf.keras.mixed_precision.set_global_policy(args.dtype_policy)
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
    tf.config.threading.set_intra_op_parallelism_threads(64)
    tf.config.threading.set_inter_op_parallelism_threads(64)

    class GeneratorWrapper(tf.keras.utils.Sequence):
        def __init__(self, generator, decks, num_cards, batch_size):
            self.generator = generator
            self.num_decks = len(decks)
            self.decks = decks
            self.num_cards = num_cards
            self.batch_size = batch_size

        def get_one_hot(self, deck_indices, max_one=False):
            deck = np.zeros((self.num_cards,), dtype=np.float32)
            if max_one:
                for index in deck_indices:
                    deck[index] += 1
            else:
                for index in deck_indices:
                    deck[index] = 1
            return deck

        def __getitem__(self, item):
            x, y = self.generator.__getitem__(item)
            chosen_indices = np.random.choice(self.num_decks, size=self.batch_size, replace=False)
            deck_x_tensor = np.stack([self.get_one_hot(self.decks[deck_index]['main'])
                                      + self.get_one_hot(self.decks[deck_index]['side']) for deck_index in chosen_indices])
            deck_y_tensor = np.stack([self.get_one_hot(self.decks[deck_index]['main'], max_one=True) for deck_index in chosen_indices])
            return (
                (x, deck_x_tensor),
                (np.clip(y, 0, 1), np.clip(deck_y_tensor, 0, 1)),
            )

        def __len__(self):
            return len(self.generator)

    def load_metapaths():
        print(f'Loading metapath adjacency matrices')
        return tuple(sp.load_npz(filename).tocoo() / 1024 for filename in tqdm(list((data / 'adjs').iterdir())))

    mirrored_strategy = tf.distribute.MirroredStrategy()
    print('Setting Up Model . . . \n')
    checkpoint_dir = output / 'checkpoint'
    THRESHOLDS=[0.1, 0.25, 0.5, 0.75, 0.9]
    with mirrored_strategy.scope():
        recommender = MetapathRecommender(
            card_metapaths=load_metapaths(), embed_dims=args.embed_dims, l1_weight=args.l1_weight,
            metapath_dims=args.metapath_dims, num_heads=args.num_heads, dropout=args.dropout,
            margin=args.margin, decks_weight=args.decks_weight, l2_weight=args.l2_weight,
            jit_scope=jit_scope, name='MetapathRecommender',
            cube_metrics=[
                *[tf.keras.metrics.Recall(t, name=f'cube_recall_{t}') for t in THRESHOLDS],
                *[tf.keras.metrics.PrecisionAtRecall(t, name=f'cube_precision_at_recall_{t}') for t in THRESHOLDS],
                *[tf.keras.metrics.RecallAtPrecision(t, name=f'cube_recall_at_precision_{t}') for t in THRESHOLDS],
                *[tf.keras.metrics.Precision(t, name=f'cube_precision_{t}') for t in THRESHOLDS],
            ],
            deck_metrics=[
                *[tf.keras.metrics.Recall(t, name=f'deck_recall_{t}') for t in THRESHOLDS],
                *[tf.keras.metrics.PrecisionAtRecall(t, name=f'deck_precision_at_recall_{t}') for t in THRESHOLDS],
                *[tf.keras.metrics.RecallAtPrecision(t, name=f'deck_recall_at_precision_{t}') for t in THRESHOLDS],
                *[tf.keras.metrics.Precision(t, name=f'deck_precision_{t}') for t in THRESHOLDS],
            ]
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    output.mkdir(exist_ok=True, parents=True)
    latest = tf.train.latest_checkpoint(str(output))
    if latest is not None:
        print('Loading Checkpoint. Saved values are:')
        recommender.load_weights(latest)
    else:
        with open(output / 'int_to_card.json', 'w') as int_to_card_file:
            json.dump(int_to_card, int_to_card_file)
        with open(output / 'args.json', 'w') as args_file:
            json.dump(args.__dict__, args_file, indent=2)
    recommender.compile(optimizer=optimizer)

    generator = GeneratorWithoutAdj(
        load_adj_mtx(),
        load_cubes(),
        args.num_workers,
        args.batch_size,
        args.seed,
        args.noise,
        args.noise_stddev,
    )
    print(f'Creating a pool with {args.num_workers} different workers.')
    with generator, mirrored_strategy.scope():
        print('Loading decks with sideboards.')
        # cubes, cube_ids = utils.build_sparse_cubes(cube_folder, is_valid_cube)
        # cube_id_to_index = {v: i for i, v in enumerate(cube_ids)}
        cube_id_to_index = {}
        wrapped = GeneratorWrapper(
            generator, num_cards=num_cards, batch_size=args.batch_size,
            decks=utils.build_deck_with_sides(decks_folder, cube_id_to_index,
                                              lambda deck: len(deck['side']) > 5 and 23 <= len(deck['main']) <= 60)
        )

        print('Starting training')
        callbacks = []
        mcp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir,
            monitor='loss',
            verbose=False,
            save_best_only=False,
            save_weights_only=True,
            mode='min',
            save_freq='epoch')
        nan_callback = tf.keras.callbacks.TerminateOnNaN()
        es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=64,
                                                       mode='min', restore_best_weights=True, verbose=True)
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, mode='min',
                                                           patience=args.plateau_patience,
                                                           cooldown=args.plateau_patience // 4,
                                                           min_delta=1 / (2 ** 14),
                                                           min_lr=1 / (2 ** 20),
                                                           verbose=True)
        tb_callback = TensorBoardFix(log_dir=log_dir, histogram_freq=1, write_graph=True,
                                     update_freq=num_cubes // 10 // args.batch_size,
                                     profile_batch=0 if not args.profile
                                     else (int(1.4 * num_cubes / args.batch_size),
                                           int(1.6 * num_cubes / args.batch_size)))
        tqdm_callback = TqdmCallback(epochs=args.epochs, data_size=len(wrapped) * args.batch_size,
                                     batch_size=args.batch_size, dynamic_ncols=True)
        callbacks.append(mcp_callback)
        callbacks.append(nan_callback)
        callbacks.append(tb_callback)
        callbacks.append(lr_callback)
        # callbacks.append(tqdm_callback)
        # callbacks.append(es_callback)
        recommender.fit(
            wrapped,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=2,
        )
        print('Saving final model')
        recommender.save(output, save_format='tf')

