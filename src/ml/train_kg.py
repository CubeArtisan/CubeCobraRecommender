import contextlib
import itertools
import json
import os
import random
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm
from tqdm.keras import TqdmCallback

def is_valid_cube(cube):
    # return True
    return cube['numDecks'] > 0 and len(set(cube['cards'])) >= 120


if __name__ == '__main__':
    import argparse
    import locale

    import numpy as np
    import src.non_ml.utils as utils
    import tensorflow as tf

    from src.ml.kgin import KGRecommender, EDGE_TYPES, NODE_TYPES, NUM_EDGE_TYPES
    from src.generated.generator import GeneratorWithoutAdj

    locale.setlocale(locale.LC_ALL, '')

    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        print('Could not enable dynamic memory growth on the GPU.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', type=int, help='The number of epochs to train for.')
    parser.add_argument('--batch-size', type=int, choices=[2**i for i in range(0, 15)], help='The number of cubes to process at a time.')
    parser.add_argument('--chunk-size', type=int, choices=[2**i for i in range(0, 18)], help='The number of edges to process at a time.')
    parser.add_argument('--layers', type=int, default=3, choices=list(range(9)), help='The number of message passing layers.')
    parser.add_argument('--entity-dims', type=int, default=128, choices=[2**i for i in range(0, 12)], help='The number of dimensions for the node embeddings')
    parser.add_argument('--edge-type-dims', type=int, default=64, choices=[2**i for i in range(0, 10)], help='The number of dimensions for the edge_type specific views of the node embeddings.')
    parser.add_argument('--initializer', type=str, default='glorot_uniform', help='The initializer for the model weights, value can be any supported by keras.')
    parser.add_argument('--edge-type-activation', type=str, default='linear', help='The activation function for mapping to the edge_type views of nodes. Value can be any supported by keras.')
    parser.add_argument('--message-dropout', type=float, default=0.2, help='The number of edges that get skipped for message passing in each layer, only applies for training.')
    parser.add_argument('--intents', type=int, default=4, choices=[2**i for i in range(0, 7)], help='The number of distinct intents(edge_type prioritizations) to model for each node type.')
    parser.add_argument('--intent-activation', type=str, default='tanh', help='The activation function for the intent values. Value can by any supported by keras.')
    parser.add_argument('--message-activation', type=str, default='linear', help='The activation function for the passed messages. Value can by any supported by keras.')
    parser.add_argument('--name', '-n', '-o', type=str, help='The folder under ml_files to save the model in.')
    parser.add_argument('--noise', type=float, default=0.5, help='The mean number of random swaps to make per cube.')
    parser.add_argument('--noise-stddev', type=float, default=0.1, help="The standard deviation of the amount of noise to apply.")
    parser.add_argument('--learning-rate', type=float, default=1e-04, help="The initial learning rate.")
    parser.add_argument('--seed', type=int, default=37, help="A random seed to provide reproducible runs.")
    parser.add_argument('--xla', action='store_true', help='Use the XLA optimizer on the model.')
    parser.add_argument('--profile', action='store_true', help='Run profiling on part of the second batch to analyze performance.')
    parser.add_argument('--debug', action='store_true', help='Enable dumping debug information to logs/debug.')
    parser.add_argument('--num-workers', '-j', type=int, default=1, help='Number of simulataneous workers to run to generate the data.')
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

    print('Loading card data and cube counts.')
    with open(int_to_card_filepath, 'rb') as int_to_card_file:
        int_to_card = json.load(int_to_card_file)
    card_to_int = {v: i for i, v in enumerate(int_to_card)}
    num_cards = len(int_to_card)
    num_cubes = utils.get_num_objs(cube_folder, validation_func=is_valid_cube)
    print(f'There are {num_cubes} valid cubes.')

    def load_cubes():
        print('Loading Cube Data')
        cubes = utils.build_cubes(cube_folder, num_cubes, num_cards, card_to_int,
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
            op_regex="(?!^(Placeholder|Constant)$)"
        )

    tf.config.optimizer.set_jit(args.xla)
    if args.xla:
        def jit_scope():
            return tf.xla.experimental.jit_scope(compile_ops=True, separate_compiled_gradients=True)
    else:
        def jit_scope():
            return contextlib.nullcontext()
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
    tf.config.threading.set_intra_op_parallelism_threads(32)
    tf.config.threading.set_inter_op_parallelism_threads(32)
    if args.mlir:
        print('MLIR is very experimental currently and so may cause errors.')
        tf.config.experimental.enable_mlir_graph_optimization()
        tf.config.experimental.enable_mlir_bridge()

    class GeneratorWrapper(tf.keras.utils.Sequence):
        def __init__(self, generator, decks, cubes, num_cards, chunk_size, batch_size):
            self.num_decks = len(decks)
            self.num_cubes = len(cubes)
            self.num_cards = num_cards
            self.chunk_size = chunk_size
            self.batch_size = batch_size
            self.cube_start_index = 0
            self.deck_start_index = self.cube_start_index + self.num_cubes
            self.card_start_index = self.deck_start_index + self.num_decks
            self.placeholder_index = self.card_start_index + self.num_cards
            self.batch_start_index = self.placeholder_index + 1
            self.num_entities = self.batch_start_index + self.batch_size
            self.generator = generator
            node_types = [-1 for _ in range(self.placeholder_index)]
            for i in range(self.cube_start_index, self.cube_start_index + self.num_cubes):
                node_types[i] = NODE_TYPES.CUBE
            for i in range(self.deck_start_index, self.deck_start_index + self.num_decks):
                node_types[i] = NODE_TYPES.DECK
            for i in range(self.card_start_index, self.card_start_index + self.num_cards):
                node_types[i] = NODE_TYPES.CARD
            self.edges = [(i, i, getattr(EDGE_TYPES, f'REFLEXIVE_{node_type.name}'))
                          for i, node_type in enumerate(node_types)]
            for i, cube in enumerate(tqdm(cubes, unit='cube', dynamic_ncols=True, unit_scale=True)):
                self._add_edges(self.cube_start_index + i, set(cube), EDGE_TYPES.CUBE_CONTAINS,
                                EDGE_TYPES.IN_CUBE)
            for i, deck in enumerate(tqdm(decks, unit='deck', dynamic_ncols=True, unit_scale=True)):
                deck_id = self.deck_start_index + i
                self._add_edges(deck_id, set(deck['main']), EDGE_TYPES.MAIN_CONTAINS, EDGE_TYPES.IN_MAIN)
                self._add_edges(deck_id, set(deck['side']), EDGE_TYPES.SIDE_CONTAINS, EDGE_TYPES.IN_SIDE)
                if deck['cube'] is not None:
                    cube_id = self.cube_start_index + deck['cube']
                    self.edges.append((deck_id, cube_id, EDGE_TYPES.DECK_FROM))
                    self.edges.append((cube_id, deck_id, EDGE_TYPES.DECK_FOR_CUBE))
            # self.edges = sorted(self.edges, key=lambda x: (x[1], x[0]))
            self.edges = np.int32(self.edges)
            print(f'There are {self.num_entities - self.batch_size:n} nodes and {len(self.edges):n} edges which is an average of {len(self.edges) / (self.num_entities - self.batch_size):n} edges per node.')
            self.placeholder_edge = np.int32((self.placeholder_index, self.placeholder_index, EDGE_TYPES.REFLEXIVE_PLACEHOLDER))

        def _add_edges(self, entity_id, card_ids, out_edge_type, in_edge_type):
            self.edges += [(entity_id, self.card_start_index + card_id, out_edge_type) for card_id in card_ids]
            self.edges += [(self.card_start_index + card_id, entity_id,  in_edge_type) for card_id in card_ids]

        def __getitem__(self, item):
            x, y = self.generator.__getitem__(item)
            cube_indices, card_indices = np.int32(x.nonzero())
            edges = list(set(zip(card_indices + self.card_start_index, cube_indices,
                                 (EDGE_TYPES.IN_CUBE for _ in cube_indices))))
            edges = np.int32(edges)
            placeholder_count = -(len(self.edges) + len(edges)) % self.chunk_size
            # edges = sorted(edges, key=lambda x: (x[1], x[0]))
            if placeholder_count > 0:
                placeholder_edges = np.tile(self.placeholder_edge, (placeholder_count, 1))
                edges = np.concatenate([placeholder_edges, edges], 0)
            sources = tf.convert_to_tensor(edges[:, 0].reshape(-1), dtype=tf.int32, name='new_sources')
            targets = tf.convert_to_tensor(edges[:, 1].reshape(-1), dtype=tf.int32, name='new_targets')
            edge_types = np.zeros((len(edges), NUM_EDGE_TYPES))
            edge_types[:, edges[:, 2]] = 1.0
            edge_types = tf.convert_to_tensor(edge_types, dtype=tf.float32, name='new_edge_types')
            return (sources, targets, edge_types), y

        def __len__(self):
            return len(self.generator)

    print(f'Creating a pool with {args.num_workers} different workers.')
    generator = GeneratorWithoutAdj(
        load_adj_mtx(),
        load_cubes(),
        args.num_workers,
        args.batch_size,
        args.seed,
        args.noise,
        args.noise_stddev,
    )

    with generator:
        print('Setting up graph and lookup tables')
        cubes, cube_ids = utils.build_sparse_cubes(cube_folder, card_to_int, is_valid_cube)
        cube_id_to_index = {v: i for i, v in enumerate(cube_ids)}
        decks = utils.build_deck_with_sides(decks_folder, card_to_int, cube_id_to_index,
                                            lambda deck: len(deck['side']) > 5 and 23 <= len(deck['main']) <= 60)
        wrapped = GeneratorWrapper(generator, decks=decks, cubes=cubes,
                                   num_cards=num_cards, chunk_size=args.chunk_size, batch_size=args.batch_size)

        print('Setting Up Model . . . \n')
        checkpoint_dir = output / 'checkpoint'
        recommender = KGRecommender(num_cubes=len(cubes), num_decks=len(decks), chunk_size=args.chunk_size,
                                    num_cards=num_cards, batch_size=args.batch_size, num_layers=args.layers,
                                    entity_dims=args.entity_dims, edges=wrapped.edges,
                                    edge_type_dims=args.edge_type_dims, initializer=args.initializer,
                                    edge_type_activation=args.edge_type_activation,
                                    message_dropout=args.message_dropout, num_intents=args.intents,
                                    intent_activation=args.intent_activation, message_activation=args.message_activation,
                                    jit_scope=jit_scope, name='knowledge_graph_cube_recommender')
        del decks
        del cubes

        latest = tf.train.latest_checkpoint(str(output))
        if latest is not None:
            print('Loading Checkpoint. Saved values are:')
            recommender.load_weights(latest)
        THRESHOLDS=[0.1, 0.25, 0.5, 0.75, 0.9]
        recommender.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(name='cube_loss'),
            metrics=[
                (
                    *[tf.keras.metrics.Recall(t, name=f'cube_recall_{t}') for t in THRESHOLDS],
                    *[tf.keras.metrics.PrecisionAtRecall(t, name=f'cube_precision_at_recall_{t}') for t in THRESHOLDS],
                    *[tf.keras.metrics.RecallAtPrecision(t, name=f'cube_recall_at_precision_{t}') for t in THRESHOLDS],
                    *[tf.keras.metrics.Precision(t, name=f'cube_precision_{t}') for t in THRESHOLDS],
                ),
            ],
            from_serialized=True,
        )

        print('Starting training')
        output.mkdir(exist_ok=True, parents=True)
        callbacks = []
        mcp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir,
            monitor='loss',
            verbose=False,
            save_best_only=False,
            mode='min',
            save_freq='epoch')
        nan_callback = tf.keras.callbacks.TerminateOnNaN()
        es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=64,
                                                       mode='min', restore_best_weights=True, verbose=True)
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, mode='min',
                                                           patience=32, min_delta=1 / (2 ** 14), cooldown=16,
                                                           min_lr=1 / (2 ** 20),
                                                           verbose=True)
        tb_callback = TensorBoardFix(log_dir=log_dir, histogram_freq=1, write_graph=True,
                                     update_freq=num_cubes // 10 // num_cubes,
                                     profile_batch=0 if not args.profile
                                     else (int(1.4 * num_cubes / num_cubes),
                                           int(1.6 * num_cubes / num_cubes)))
        tqdm_callback = TqdmCallback(epochs=args.epochs, data_size=len(wrapped) * args.batch_size,
                                     batch_size=args.batch_size, dynamic_ncols=True)
        callbacks.append(mcp_callback)
        callbacks.append(nan_callback)
        callbacks.append(tb_callback)
        callbacks.append(lr_callback)
        callbacks.append(tqdm_callback)
        # callbacks.append(es_callback)
        recommender.fit(
            wrapped,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=0,
        )
        print('Saving final model')
        recommender.save(output, save_format='tf')
