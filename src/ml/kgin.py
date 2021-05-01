import contextlib
from enum import Enum

import tensorflow as tf

EDGE_TYPES = Enum('EDGE_TYPES', (
    'REFLEXIVE',
    'IN_POOL',
    'POOL_CONTAINS',
    'IN_MAIN',
    'MAIN_CONTAINS',
    'IN_SIDE',
    'SIDE_CONTAINS',
    'IN_CUBE',
    'CUBE_CONTAINS'
), start=0, type=int, module=__name__)
NUM_EDGE_TYPES = len(EDGE_TYPES)
NODE_TYPES = Enum('NODE_TYPES', (
    'CUBE',
    'DECK',
    'CARD',
), start=0, type=int, module=__name__)
NUM_NODE_TYPES = len(NODE_TYPES)

class MultiDense(tf.keras.layers.Layer):
    def __init__(self, dims, out_dims, initializer='glorot_uniform', bias_initializer='zeros',
                 use_bias=True, activation='linear', **kwargs):
        super(MultiDense, self).__init__(**kwargs, dynamic=False)
        self.dims = dims
        self.out_dims = out_dims
        self.initializer = initializer
        self.bias_initializer = bias_initializer
        self.activation = activation
        self.use_bias = use_bias

    def build(self, input_shape):
        self.weights_lookup = self.add_weight(shape=(self.dims, input_shape[1][-1], self.out_dims),
                                              trainable=True, initializer=self.initializer, name='weights_lookup')
        if self.use_bias:
            self.bias_lookup = self.add_weight(shape=(self.dims, self.out_dims),
                                               trainable=True, initializer=self.bias_initializer, name='bias_lookup')
        self.activation_layer = tf.keras.layers.Activation(self.activation)

    @tf.function()
    def call(self, inputs, training=False):
        lookups, values = inputs
        weights = tf.gather(self.weights_lookup, lookups)
        mapped = tf.einsum('...i,...io->...o', values, weights)
        if self.use_bias:
            biases = tf.gather(self.bias_lookup, lookups)
            mapped = mapped + biases
        return self.activation_layer(mapped)

    def get_config(self):
        config = super(MultiDense, self).get_config()
        config.update({
            'dims': self.dims,
            'out_dims': self.out_dims,
            'initializer': self.initializer,
            'bias_initializer': self.bias_initializer,
            'activation': self.activation,
            'use_bias': self.use_bias,
        })
        return config


class KGIN(tf.keras.layers.Layer):
    def __init__(self, num_entities, num_layers=3, entity_dims=128, normalize=None,
                 edge_type_dims=64, initializer='glorot_uniform', return_weights=True,
                 edge_type_activation='selu', message_dropout=0.2, num_intents=8,
                 intent_activation='softmax', message_activation='linear',
                 jit_scope=contextlib.nullcontext, **kwargs):
        super(KGIN, self).__init__(**kwargs, dynamic=False)
        self.num_entities = num_entities
        self.num_layers = num_layers
        self.entity_dims = entity_dims
        self.normalize = normalize
        self.edge_type_dims = edge_type_dims
        self.initializer = initializer
        self.return_weights = return_weights
        self.edge_type_activation = edge_type_activation
        self.message_dropout = message_dropout
        self.num_intents = num_intents
        self.intent_activation = intent_activation
        self.message_activation= message_activation
        self.jit_scope = jit_scope

    def get_config(self):
        config = super(KGIN, self).get_config()
        config.update({
            'num_entities': self.num_entities,
            'num_layers': self.num_layers,
            'entity_dims': self.entity_dims,
            'normalize': self.normalize,
            'edge_type_dims': self.edge_type_dims,
            'initializer': self.initializer,
            'return_weights': self.return_weights,
            'edge_type_activation': self.edge_type_activation,
            'message_dropout': self.message_dropout,
            'num_intents': self.num_intents,
            'intent_activation': self.intent_activation,
            'message_activation': self.message_activation,
        })
        return config

    def build(self, input_shape):
        self.edge_type_embeddings = tf.keras.layers.Embedding(NUM_EDGE_TYPES, self.edge_type_dims,
                                                             embeddings_initializer=self.initializer,
                                                             dtype=self.dtype_policy, name='edge_type_embeddings')
        self.transform_target_by_edge_type = MultiDense(NUM_EDGE_TYPES, self.edge_type_dims, initializer=self.initializer,
                                                       use_bias=True, activation=self.edge_type_activation,
                                                       dtype=self.dtype_policy, name='transform_target_by_edge_type')
        self.transform_source_by_edge_type = MultiDense(NUM_EDGE_TYPES, self.edge_type_dims, initializer=self.initializer,
                                                       use_bias=True, activation=self.edge_type_activation,
                                                       dtype=self.dtype_policy, name='transform_source_by_edge_type')
        self.message_for_edge_type = MultiDense(NUM_EDGE_TYPES, self.entity_dims, initializer=self.initializer,
                                               use_bias=False, activation=self.message_activation,
                                               dtype=self.dtype_policy, name='message_for_edge_type')
        self.intent_maps = MultiDense(NUM_NODE_TYPES, self.num_intents, activation='softmax',
                                      use_bias=True, initializer=self.initializer, dtype=self.dtype_policy,
                                      name='intent_maps')
        self.intents_by_node = MultiDense(NUM_NODE_TYPES, NUM_EDGE_TYPES, activation=self.intent_activation,
                                          use_bias=True, initializer=self.initializer, dtype=self.dtype_policy,
                                          name='intents_by_node')
        self.message_dropout_layer = tf.keras.layers.Dropout(self.message_dropout, name='message_dropout')

    @tf.function()
    def _pass_messages_through_chunk(self, chunked_edges, entity_embeds, training=False):
        with self.jit_scope():
            sources, targets, target_types, edge_types = tf.unstack(chunked_edges, num=4, axis=1)
            source_embeds = tf.gather(entity_embeds, sources)
            target_embeds = tf.gather(entity_embeds, targets)
            edge_type_embeds = self.edge_type_embeddings(edge_types)
            target_intents = self.intent_maps([target_types, target_embeds], training=training)
            target_edge_type_weights = self.intents_by_node([target_types, target_intents], training=training)
            intent_weights = tf.gather(target_edge_type_weights, edge_types, batch_dims=1)
            source_rel_embeds = self.transform_source_by_edge_type([edge_types, source_embeds], training=training)
            scores = tf.einsum('er,er,e->e', source_rel_embeds, edge_type_embeds, intent_weights)
            exp_scores = tf.math.exp(scores)
            exp_scores_by_target = tf.math.reciprocal_no_nan(tf.scatter_nd(tf.expand_dims(targets, 1),
                                                                           exp_scores, (self.num_entities,)))
            distribution = exp_scores * tf.gather(exp_scores_by_target, targets)
            chunked_edge_weights = tf.expand_dims(self.message_dropout_layer(distribution, training=training), 1)
            messages = chunked_edge_weights * self.message_for_edge_type([edge_types, source_embeds], training=training)
            new_embeds = tf.scatter_nd(tf.expand_dims(targets, 1), messages, entity_embeds.shape)
            if self.return_weights:
                return new_embeds, chunked_edge_weights
            else:
                return new_embeds,

    @tf.function()
    def pass_messages(self, entity_embeds, edges, chunks, training=False):
        with self.jit_scope():
            new_embeds = tf.zeros_like(entity_embeds)
            if self.return_weights:
                edge_weights = tf.zeros((edges.shape[0],))
            for chunk in chunks:
                chunked_edges = tf.gather(edges, chunk)
                chunk_results = self._pass_messages_through_chunk(chunked_edges, entity_embeds, training=training)
                if self.return_weights:
                    edge_weights = tf.tensor_scatter_nd_add(edge_weights, tf.expand_dims(chunk, 1), chunk_results[1])
                new_embeds += chunk_results[0]
            if self.normalize is not None:
                new_embeds = tf.linalg.normalize(new_embeds, ord=self.normalize, axis=1)[0]
            if self.return_weights:
                return new_embeds, edge_weights
            else:
                return new_embeds,

    @tf.function()
    def call(self, inputs, training=False):
        with self.jit_scope():
            entity_embeds, edges, chunks = inputs
            if self.return_weights:
                layer_edge_weights = []
            for _ in range(self.num_layers):
                updated_state = self.pass_messages(entity_embeds, edges, chunks, training=False)
                if self.return_weights:
                    layer_edge_weights.append(updated_state[1])
                entity_embeds = updated_state[0]
            if self.return_weights:
                return entity_embeds, layer_edge_weights
            else:
                return entity_embeds,

    @tf.function()
    def predict_links(self, source_embeds, target_embeds, edge_type, training=False):
        with self.jit_scope():
            source_rel_embeds = tf.math.l2_normalize(
                self.transform_source_by_edge_type([edge_type, source_embeds], training=training),
                axis=1
            )
            target_rel_embeds = tf.math.l2_normalize(
                self.transform_target_by_edge_type([edge_type, target_embeds], training=training),
                axis=1
            )
            return tf.einsum('sr,tr->st', source_rel_embeds, target_rel_embeds)


class KGRecommender(tf.keras.Model):
    def __init__(self, num_cubes, num_decks, num_cards, batch_size, num_layers=3, entity_dims=128,
                 normalize=None, edge_type_dims=64, initializer='glorot_uniform', return_weights=True,
                 edge_type_activation='selu', message_dropout=0.2, num_intents=8, num_chunks=32,
                 intent_activation='tanh', message_activation='linear', jit_scope=contextlib.nullcontext,
                 **kwargs):
        super(KGRecommender, self).__init__(**kwargs)
        self.num_decks = num_decks
        self.num_cubes = num_cubes
        self.num_cards = num_cards
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.entity_dims = entity_dims
        self.normalize = normalize
        self.edge_type_dims = edge_type_dims
        self.initializer = initializer
        self.return_weights = return_weights
        self.edge_type_activation = edge_type_activation
        self.message_dropout = message_dropout
        self.num_intents = num_intents
        self.intent_activation = intent_activation
        self.message_activation = message_activation

        self.cube_start_index = self.batch_size
        self.deck_start_index = self.cube_start_index + self.num_cubes
        self.card_start_index = self.deck_start_index + self.num_decks
        self.num_entities = self.card_start_index + self.num_cards
        node_types = [-1 for _ in range(self.num_entities)]
        for i in range(self.batch_size):
            node_types[i] = NODE_TYPES.CUBE
        for i in range(self.cube_start_index, self.cube_start_index + self.num_cubes):
            node_types[i] = NODE_TYPES.CUBE
        for i in range(self.deck_start_index, self.deck_start_index + self.num_decks):
            node_types[i] = NODE_TYPES.DECK
        for i in range(self.card_start_index, self.card_start_index + self.num_cards):
            node_types[i] = NODE_TYPES.CARD
        self.node_types = node_types
        self.card_indices = tf.range(self.card_start_index, self.card_start_index + self.num_cards, dtype=tf.int32)
        self.defaulted_node_types = tf.constant(self.node_types[:self.card_start_index], dtype=tf.int32)
        self.batch_indices = tf.range(0, self.batch_size, dtype=tf.int32)
        self.jit_scope = jit_scope

    def get_config(self):
        config = super(KGRecommender, self).get_config()
        config.update({
            'num_cubes': self.num_cubes,
            'num_decks': self.num_decks,
            'num_cards': self.num_cards,
            'batch_size': self.batch_size,
            'num_layers': self.num_layers,
            'entity_dims': self.entity_dims,
            'normalize': self.normalize,
            'edge_type_dims': self.edge_type_dims,
            'initializer': self.initializer,
            'return_weights': self.return_weights,
            'edge_type_activation': self.edge_type_activation,
            'message_dropout': self.message_dropout,
            'num_intents': self.num_intents,
            'num_chunks': len(self.chunks) + 1,
            'intent_activation': self.intent_activation,
            'message_activation': self.message_activation,
        })
        return config

    def build(self, input_shape):
        self.card_embeddings = self.add_weight(shape=(self.num_cards, self.entity_dims),
                                               initializer=self.initializer, trainable=True,
                                               name='card_embeddings')
        self.type_embeddings = tf.keras.layers.Embedding(NUM_NODE_TYPES, self.entity_dims,
                                                         embeddings_initializer=self.initializer,
                                                         dtype=self.dtype_policy, name='type_embeddings')
        self.kgin = KGIN(num_entities=self.num_entities, num_layers=self.num_layers,
                         entity_dims=self.entity_dims, normalize=self.normalize,
                         edge_type_dims=self.edge_type_dims, initializer=self.initializer,
                         return_weights=self.return_weights, edge_type_activation=self.edge_type_activation,
                         message_dropout=self.message_dropout, num_intents=self.num_intents,
                         intent_activation=self.intent_activation, message_activation=self.message_activation,
                         dtype=self.dtype_policy, jit_scope=self.jit_scope, name='kg_embeddings')

    @tf.function()
    def call(self, inputs, training=False):
        with self.jit_scope():
            edges, chunks = inputs
            entity_embeds = tf.concat([
                self.type_embeddings(self.defaulted_node_types),
                self.card_embeddings
            ], 0)
            kgin_results = self.kgin([entity_embeds, edges, chunks], training=training)
            if self.return_weights:
                layer_edge_weights = kgin_results[1]
            entity_embeds = kgin_results[0]
            card_embeds = self.card_embeddings
            batch_cube_embeds = tf.gather(entity_embeds, self.batch_indices)
            predictions = (self.kgin.predict_links(batch_cube_embeds, card_embeds, EDGE_TYPES.CUBE_CONTAINS,
                                                   training=training) + 1) / 2
            if self.return_weights:
                return predictions, layer_edge_weights
            else:
                return predictions,
