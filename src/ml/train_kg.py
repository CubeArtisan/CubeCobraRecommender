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

    from src.ml.kgin import KGRecommender, EDGE_TYPES, NODE_TYPES
    from src.generated.generator import GeneratorWithoutAdj

    locale.setlocale(locale.LC_ALL, '')

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', type=int, help='The number of epochs to train for.')
    parser.add_argument('--batch-size', '-b', type=int, choices=[2**i for i in range(0, 16)], help='The number of cubes/cards to train on at a time.')
    parser.add_argument('--layers', type=int, default=3, choices=list(range(9)), help='The number of message passing layers.')
    parser.add_argument('--entity-dims', type=int, default=128, choices=[2**i for i in range(0, 12)], help='The number of dimensions for the node embeddings')
    parser.add_argument('--normalize', type=int, default=None, choices=[1, 2], help='The kind of normalization to apply to the embeddings, L1 or L2 default is no normalization.')
    parser.add_argument('--edge-type-dims', type=int, default=64, choices=[2**i for i in range(0, 10)], help='The number of dimensions for the edge_type specific views of the node embeddings.')
    parser.add_argument('--initializer', type=str, default='glorot_uniform', help='The initializer for the model weights, value can be any supported by keras.')
    parser.add_argument('--edge-type-activation', type=str, default='linear', help='The activation function for mapping to the edge_type views of nodes. Value can be any supported by keras.')
    parser.add_argument('--message-dropout', type=float, default=0.2, help='The number of edges that get skipped for message passing in each layer, only applies for training.')
    parser.add_argument('--intents', type=int, default=4, choices=[2**i for i in range(0, 7)], help='The number of distinct intents(edge_type prioritizations) to model for each node type.')
    parser.add_argument('--intent-activation', type=str, default='tanh', help='The activation function for the intent values. Value can by any supported by keras.')
    parser.add_argument('--message-activation', type=str, default='linear', help='The activation function for the passed messages. Value can by any supported by keras.')
    parser.add_argument('--chunks', type=int, default=64, choices=[2**i for i in range(0, 18)], help='The number of chunks to break up the edges into for processing to reduce memory usage.')
    parser.add_argument('--min-chunk-size', type=int, default=8192, choices=[2**i for i in range(0, 18)], help='The smallest size for the edge chunks.')
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

    print("Setting jit flag")
    tf.config.optimizer.set_jit(args.xla)
    if args.xla:
        def jit_scope():
            return tf.xla.experimental.jit_scope(compile_ops=True, separate_compiled_gradients=True)
    else:
        def jit_scope():
            return contextlib.nullcontext()
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
        # 'auto_mixed_precision': True,
        'disable_meta_optimizer': False,
        'min_graph_nodes': 8,
    })
    tf.config.threading.set_intra_op_parallelism_threads(32)
    tf.config.threading.set_inter_op_parallelism_threads(32)
    tf.config.set_soft_device_placement(True)
    tf.keras.mixed_precision.set_global_policy(args.dtype_policy)
    if args.mlir:
        print('MLIR is very experimental currently and so may cause errors.')
        tf.config.experimental.enable_mlir_graph_optimization()
        tf.config.experimental.enable_mlir_bridge()

    print('Setting Up Model . . . \n')
    checkpoint_dir = output / 'checkpoint'
    cubes = utils.build_sparse_cubes(cube_folder, card_to_int, is_valid_cube)
    decks = utils.build_deck_with_sides(decks_folder, card_to_int,
                                        lambda deck: len(deck['side']) > 5 and 22 <= len(deck['main']) <= 100)
    recommender = KGRecommender(num_cubes=len(cubes), num_decks=len(decks),
                                num_cards=num_cards, batch_size=args.batch_size, num_layers=args.layers,
                                entity_dims=args.entity_dims, normalize=args.normalize, num_chunks=args.chunks,
                                edge_type_dims=args.edge_type_dims, initializer=args.initializer,
                                return_weights=False, edge_type_activation=args.edge_type_activation,
                                message_dropout=args.message_dropout, num_intents=args.intents,
                                intent_activation=args.intent_activation, message_activation=args.message_activation,
                                jit_scope=jit_scope, name='knowledge_graph_cube_recommender')

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
                *[tf.keras.metrics.PrecisionAtRecall(t, name=f'cube_prec_at_recall_{t}') for t in THRESHOLDS],
                *[tf.keras.metrics.RecallAtPrecision(t, name=f'cube_recall_at_precision_{t}') for t in THRESHOLDS],
                *[tf.keras.metrics.Precision(t, name=f'cube_precision_{t}') for t in THRESHOLDS],
            ),
        ],
        from_serialized=True,
    )

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

    class GeneratorWrapper(tf.keras.utils.Sequence):
        def __init__(self, generator, decks, cubes, num_cards, batch_size, num_layers, num_chunks, min_chunk_size):
            self.num_layers = num_layers
            self.num_cards = num_cards
            self.batch_size = batch_size
            self.num_decks = len(decks)
            self.num_cubes = len(cubes)
            self.min_chunk_size = min_chunk_size
            self.num_chunks = num_chunks
            self.cube_start_index = self.batch_size
            self.deck_start_index = self.cube_start_index + self.num_cubes
            self.card_start_index = self.deck_start_index + self.num_decks
            self.num_entities = self.card_start_index + self.num_cards
            self.generator = generator
            self.recommender = recommender
            self.edges = set()
            self.node_types = [-1 for _ in range(self.num_entities)]
            for i in range(self.batch_size):
                self.node_types[i] = NODE_TYPES.CUBE
            for i in range(self.cube_start_index, self.cube_start_index + self.num_cubes):
                self.node_types[i] = NODE_TYPES.CUBE
            for i in range(self.deck_start_index, self.deck_start_index + self.num_decks):
                self.node_types[i] = NODE_TYPES.DECK
            for i in range(self.card_start_index, self.card_start_index + self.num_cards):
                self.node_types[i] = NODE_TYPES.CARD
            for i, cube in enumerate(tqdm(cubes, unit='cube', dynamic_ncols=True, unit_scale=True)):
                self._add_edges(self.cube_start_index + i, cube, EDGE_TYPES.CUBE_CONTAINS,
                                EDGE_TYPES.IN_CUBE, NODE_TYPES.CUBE)
            for i, deck in enumerate(tqdm(decks, unit='deck', dynamic_ncols=True, unit_scale=True)):
                deck_id = self.deck_start_index + i
                self._add_edges(deck_id, deck['main'], EDGE_TYPES.POOL_CONTAINS, EDGE_TYPES.IN_POOL,
                                NODE_TYPES.DECK)
                self._add_edges(deck_id, deck['main'], EDGE_TYPES.MAIN_CONTAINS, EDGE_TYPES.IN_MAIN,
                                NODE_TYPES.DECK)
                self._add_edges(deck_id, deck['side'], EDGE_TYPES.POOL_CONTAINS, EDGE_TYPES.IN_POOL,
                                NODE_TYPES.DECK)
                self._add_edges(deck_id, deck['side'], EDGE_TYPES.SIDE_CONTAINS, EDGE_TYPES.IN_SIDE,
                                NODE_TYPES.DECK)
            self.edges.update((i, i, self.node_types[i], EDGE_TYPES.REFLEXIVE) for i in range(self.batch_size, self.num_entities))
            self.edges = np.int32(list(self.edges))
            print(f'There are {self.num_entities:n} nodes and {len(self.edges):n} edges which is an average of {len(self.edges) / self.num_entities:n} edges per node.')
            # self._populate_edges_by_card(num_cards, num_layers)

        def _populate_edges_by_card(self, num_cards, num_layers):
            edges_by_target = [[] for _ in range(self.num_entities)]
            for i, edge in enumerate(tqdm(self.edges, unit='edge', unit_scale=True, dynamic_ncols=True)):
                edges_by_target[edge[1]].append(i)
            edges_by_target = [np.int32(indices) for indices in edges_by_target]
            sources_by_target = [np.int32(list(set(self.edges[idx][0] for idx in indices)))
                                 for indices in tqdm(edges_by_target, unit='node', unit_scale=True, dynamic_ncols=True)]
            self.edges_by_card = []
            closure_size = 0
            for card_id in tqdm(list(range(self.card_start_index, self.card_start_index + num_cards)),
                                unit='card', unit_scale=True, dynamic_ncols=True):
                edges = []
                nodes = {card_id}
                new_nodes = {card_id}
                # One hop is from the batch cubes to cards so we have 1 less layer to consider here.
                for j in range(num_layers - 1):
                    edges += list(itertools.chain.from_iterable(edges_by_target[target] for target in new_nodes))
                    # We can skip this on the last layer to save a little time
                    if j < num_layers - 2:
                        new_nodes = set(itertools.chain.from_iterable(sources_by_target[target] for target in new_nodes)) - nodes
                        nodes.update(new_nodes)
                closure_size += len(edges)
                self.edges_by_card.append(np.int32(edges))
            print(f'There are {closure_size:n} edges in the {num_layers - 1}-hop neighborhood of the cards which is {closure_size / num_cards:.2f} edges per node.')

        def _add_edges(self, entity_id, card_ids, out_edge_type, in_edge_type, entity_type):
            self.edges.update((entity_id, self.card_start_index + card_id, NODE_TYPES.CARD, out_edge_type) for card_id in card_ids)
            self.edges.update((self.card_start_index + card_id, entity_id,     entity_type,  in_edge_type) for card_id in card_ids)

        def calculate_closure(self, target_indices, edges):
            print('Started calculating closure')
            cumulative_bitvector = np.zeros((self.num_entities,), dtype=np.bool_)
            cur_bitvector = np.zeros((self.num_entities,), dtype=np.bool_)
            cumulative_bitvector[target_indices] = 1
            cur_bitvector[target_indices] = 1
            for _ in range(self.num_layers - 1):
                sources = edges[cur_bitvector[edges[:, 1]]][:, 0]
                cur_bitvector = np.zeros((self.num_entities,), dtype=np.bool_)
                cur_bitvector[sources] = 1
                cur_bitvector = cur_bitvector & np.logical_not(cumulative_bitvector)
                cumulative_bitvector = cumulative_bitvector | cur_bitvector
            return edges[cumulative_bitvector[edges[:, 1]]]

        def __getitem__(self, item):
            x, y = self.generator.__getitem__(item)
            cube_indices, card_indices = np.int32(x.nonzero())
            edges = list(set(zip(
                card_indices + self.card_start_index,
                cube_indices,
                [NODE_TYPES.CUBE for _ in cube_indices],
                [EDGE_TYPES.IN_CUBE for _ in cube_indices],
            ))) + [(i, i, NODE_TYPES.CUBE, EDGE_TYPES.REFLEXIVE) for i in range(self.batch_size)]
            edges = np.concatenate([self.edges, np.int32(sorted(edges, key=lambda x: (x[1], x[0])))], 0)
            edges = self.calculate_closure(np.arange(self.batch_size), edges)
            chunks = []
            chunk_size = len(edges) // self.num_chunks
            print(f'Making chunks of size {chunk_size:n} out of {len(edges):n} edges')
            next_start = 0
            for i in range(self.num_chunks):
                if next_start == len(edges):
                    chunks.append(tf.constant([], shape=(0,), dtype=tf.int32))
                    continue
                guess = min(max(next_start + self.min_chunk_size, (i + 1) * chunk_size), len(edges) - 1)
                while guess < len(edges) - 1 and edges[guess][1] == edges[guess + 1][1]:
                    guess += 1
                chunks.append(tf.range(next_start, guess + 1, dtype=tf.int32))
                next_start = guess + 1
            return (edges, chunks), y

        def __len__(self):
            return len(self.generator)

    with generator:
        print('Setting up graph and lookup tables')
        wrapped = GeneratorWrapper(generator, decks=decks, cubes=cubes, num_layers=args.layers,
                                   batch_size=args.batch_size, num_cards=num_cards, num_chunks=args.chunks,
                                   min_chunk_size=args.min_chunk_size)
        del decks
        del cubes
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
                                     update_freq=num_cubes // 10 // args.batch_size,
                                     profile_batch=0 if not args.profile
                                     else (int(1.4 * num_cubes / args.batch_size),
                                           int(1.6 * num_cubes / args.batch_size)))
        tqdm_callback = TqdmCallback(epochs=args.epochs, data_size=num_cubes, batch_size=args.batch_size,
                                     tqdm_class=tqdm, dynamic_ncols=True)
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
