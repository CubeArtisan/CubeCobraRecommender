import contextlib
from enum import Enum

import numpy as np
import tensorflow as tf

NODE_TYPES = Enum('NODE_TYPES', (
    'CUBE',
    'DECK',
    'CARD',
    'PLACEHOLDER',
), start=0, type=int, module=__name__)
NUM_NODE_TYPES = len(NODE_TYPES)
EDGE_TYPES = Enum('EDGE_TYPES', (
    'IN_MAIN',
    'MAIN_CONTAINS',
    'IN_SIDE',
    'SIDE_CONTAINS',
    'IN_CUBE',
    'CUBE_CONTAINS',
    'DECK_FROM',
    'DECK_FOR_CUBE',
    *[f'REFLEXIVE_{node_type.name}' for node_type in NODE_TYPES]
), start=0, type=int, module=__name__)
NUM_EDGE_TYPES = len(EDGE_TYPES)
EDGE_NODE_TYPES = {
    EDGE_TYPES.IN_MAIN: (NODE_TYPES.CARD, NODE_TYPES.DECK),
    EDGE_TYPES.MAIN_CONTAINS: (NODE_TYPES.DECK, NODE_TYPES.CARD),
    EDGE_TYPES.IN_SIDE: (NODE_TYPES.CARD, NODE_TYPES.DECK),
    EDGE_TYPES.SIDE_CONTAINS: (NODE_TYPES.DECK, NODE_TYPES.CARD),
    EDGE_TYPES.IN_CUBE: (NODE_TYPES.CARD, NODE_TYPES.CUBE),
    EDGE_TYPES.CUBE_CONTAINS: (NODE_TYPES.CUBE, NODE_TYPES.CARD),
    EDGE_TYPES.DECK_FROM: (NODE_TYPES.DECK, NODE_TYPES.CUBE),
    EDGE_TYPES.DECK_FOR_CUBE: (NODE_TYPES.CUBE, NODE_TYPES.DECK),
}
EDGE_NODE_TYPES.update({
    getattr(EDGE_TYPES, f'REFLEXIVE_{node_type.name}'): (node_type, node_type)
    for node_type in NODE_TYPES
})
assert len(EDGE_NODE_TYPES) == NUM_EDGE_TYPES, 'EDGE_NODE_TYPES should contain an entry for every EDGE_TYPE.'
NEW_EDGE_NODE_TYPES = np.zeros((NUM_EDGE_TYPES, 2, NUM_NODE_TYPES), dtype=np.float32)
NEW_EDGE_NODE_TYPES[:, 0, [x[0] for x in EDGE_NODE_TYPES.values()]] = 1.0
NEW_EDGE_NODE_TYPES[:, 1, [x[1] for x in EDGE_NODE_TYPES.values()]] = 1.0
EDGE_NODE_TYPES = NEW_EDGE_NODE_TYPES.copy()
del NEW_EDGE_NODE_TYPES


class MultiDense(tf.keras.layers.Layer):
    def __init__(self, out_dims, initializer='glorot_uniform', bias_initializer='zeros',
                 use_bias=True, activation='linear', **kwargs):
        super(MultiDense, self).__init__(**kwargs, dynamic=False)
        self.out_dims = out_dims
        self.initializer = initializer
        self.bias_initializer = bias_initializer
        self.activation = activation
        self.use_bias = use_bias

    def build(self, input_shape):
        # Let's assume that the lookup argument will be a one_hot encoding of the in_dims
        # So we have input_shape = ((*batch_sizes, lookup_dims), (*batch_sizes, input_dims))
        self.weights_lookup = self.add_weight(shape=(input_shape[0][-1], input_shape[1][-1], self.out_dims),
                                              trainable=True, initializer=self.initializer, name='weights_lookup')
        if self.use_bias:
            self.bias_lookup = self.add_weight(shape=(input_shape[0][-1], self.out_dims),
                                               trainable=True, initializer=self.bias_initializer, name='bias_lookup')
        self.activation_layer = tf.keras.layers.Activation(self.activation)

    @tf.function()
    def call(self, inputs, training=False):
        lookups, values = inputs
        weighted = tf.einsum('...i,...l,lio->...o', values, lookups, self.weights_lookup, name='weighted')
        if self.use_bias:
            biases = tf.einsum('...l,lo->...o', lookups, self.bias_lookup, name='biases')
            weighted = weighted + biases
        return self.activation_layer(weighted)

    def get_config(self):
        config = super(MultiDense, self).get_config()
        config.update({
            'out_dims': self.out_dims,
            'initializer': self.initializer,
            'bias_initializer': self.bias_initializer,
            'activation': self.activation,
            'use_bias': self.use_bias,
        })
        return config


class KGIN(tf.keras.layers.Layer):
    def __init__(self, num_entities, chunk_size, num_edge_types, num_node_types, num_layers=3,
                 entity_dims=128, edge_type_dims=64, initializer='glorot_uniform',
                 edge_type_activation='selu', message_dropout=0.2, num_intents=8,
                 intent_activation='softmax', message_activation='linear',
                 jit_scope=contextlib.nullcontext, intent_node_activation='tanh',
                 query_activation='linear', **kwargs):
        super(KGIN, self).__init__(**kwargs, dynamic=False)
        self.num_entities = num_entities
        self.chunk_size = chunk_size
        self.num_edge_types = num_edge_types
        self.num_node_types = num_node_types
        self.num_layers = num_layers
        self.entity_dims = entity_dims
        self.edge_type_dims = edge_type_dims
        self.initializer = initializer
        self.edge_type_activation = edge_type_activation
        self.message_dropout = message_dropout
        self.num_intents = num_intents
        self.intent_activation = intent_activation
        self.message_activation= message_activation
        self.intent_node_activation = intent_node_activation
        self.query_activation = query_activation
        self.jit_scope = jit_scope

        self.edge_type_to_target_type = tf.constant(
            EDGE_NODE_TYPES[:, 0, :],
            dtype=self.dtype,
            shape=(self.num_edge_types, self.num_node_types),
            name='edge_type_to_target_type'
        )

    def get_config(self):
        config = super(KGIN, self).get_config()
        config.update({
            'num_entities': self.num_entities,
            'chunk_size': self.chunk_size,
            'num_edge_types': self.num_edge_types,
            'num_node_types': self.num_node_types,
            'num_layers': self.num_layers,
            'entity_dims': self.entity_dims,
            'edge_type_dims': self.edge_type_dims,
            'initializer': self.initializer,
            'edge_type_activation': self.edge_type_activation,
            'message_dropout': self.message_dropout,
            'num_intents': self.num_intents,
            'intent_activation': self.intent_activation,
            'message_activation': self.message_activation,
            'intent_node_activation': self.intent_node_activation,
            'query_activation': self.query_activation,
        })
        return config

    def build(self, input_shape):
        self.transform_target_by_edge_type = MultiDense(self.edge_type_dims, initializer=self.initializer,
                                                       use_bias=True, activation=self.edge_type_activation,
                                                       name='transform_target_by_edge_type')
        self.target_query_by_edge_type = MultiDense(self.edge_type_dims, initializer=self.initializer,
                                                    use_bias=False, activation=self.query_activation,
                                                    name='target_query_by_edge_type')
        self.transform_source_by_edge_type = MultiDense(self.edge_type_dims, initializer=self.initializer,
                                                       use_bias=True, activation=self.edge_type_activation,
                                                       name='transform_source_by_edge_type')
        self.message_for_edge_type = MultiDense(self.entity_dims, initializer=self.initializer,
                                               use_bias=True, activation=self.message_activation,
                                               name='message_for_edge_type')
        # [num_node_types, entity_dims, num_intents]
        self.node_to_intents = MultiDense(self.num_intents, activation=self.intent_node_activation,
                                          use_bias=True, initializer=self.initializer,
                                          name='node_to_intents')
        # [num_node_types, num_intents, num_edge_types]
        self.intents_to_edge_type_weights = MultiDense(self.num_edge_types, activation=self.intent_activation,
                                                       use_bias=False, initializer=self.initializer,
                                                       name='intents_to_edge_type_weights')
        self.message_dropout_layer = tf.keras.layers.Dropout(self.message_dropout, name='message_dropout')

    @tf.function()
    def _pass_messages_through_chunk(self, i, new_embeds, entity_embeds, sources, targets, edge_types,
                                     training=False):
        with self.jit_scope():
            sources = sources[i]
            targets = targets[i]
            edge_types = edge_types[i]
            source_embeds = tf.gather(entity_embeds, sources, name='source_embeds')
            target_embeds = tf.gather(entity_embeds, targets, name='target_embeds')
            target_types = tf.einsum('ce,en->cn', edge_types, self.edge_type_to_target_type)
            target_intents = self.node_to_intents([target_types, target_embeds], training=training)
            target_edge_type_weights = self.intents_to_edge_type_weights([target_types, target_intents], training=training)
            intent_edge_weights = tf.einsum('cr,cr->c', edge_types, target_edge_type_weights, name='intent_edge_weights')
            source_rel_embeds = self.transform_source_by_edge_type([edge_types, source_embeds], training=training)
            target_queries = self.target_query_by_edge_type([edge_types, source_embeds], training=training)
            scores = tf.einsum('er,er->e', source_rel_embeds, target_queries, name='scores')
            exp_scores = tf.math.exp(scores, name='exp_scores')
            chunked_edge_weights = intent_edge_weights * self.message_dropout_layer(exp_scores, training=training)
            messages = tf.expand_dims(chunked_edge_weights, 1) * self.message_for_edge_type([edge_types, source_embeds], training=training)
            return (
                i + tf.constant(1, dtype=tf.int32),
                tf.tensor_scatter_nd_add(new_embeds, tf.expand_dims(targets, 1), messages,
                                         name='update_new_embeds_with_messages')
            )

    @tf.function()
    def _pass_messages(self, i, entity_embeds, sources, targets,
                       edge_types, training=False):
        def compute(i, new_embeds):
            return self._pass_messages_through_chunk(
                i, new_embeds, entity_embeds, sources, targets, edge_types, training=training
            )
        with self.jit_scope():
            num_chunks = tf.shape(sources)[0]
            _, entity_embeds = tf.while_loop(
                lambda i, _: i < num_chunks, compute,
                (tf.constant(0, dtype=tf.int32), tf.zeros_like(entity_embeds)),
                parallel_iterations=1, swap_memory=True, maximum_iterations=num_chunks, name='update_new_embeds'
            )
            return (
                i + tf.constant(1, dtype=tf.int32),
                tf.math.l2_normalize(entity_embeds, axis=1, name='normalized_new_embeds'),
            )

    @tf.function()
    def call(self, inputs, training=False):
        entity_embeds, sources, targets, edge_types = inputs
        def compute(i, entity_embeds):
            return self._pass_messages(i, entity_embeds, sources,
                                       targets, edge_types, training=training)
        with self.jit_scope():
            _, entity_embeds = tf.while_loop(
                lambda i, *_: i < self.num_layers, compute,
                (tf.constant(0, dtype=tf.int32), entity_embeds),
                parallel_iterations=1, swap_memory=True, maximum_iterations=self.num_layers,
                name='update_new_embeds'
            )
            return entity_embeds

    @tf.function()
    def predict_links(self, source_embeds, target_embeds, edge_type, training=False):
        with self.jit_scope():
            source_edge_types = tf.tile(edge_type, (tf.shape(source_embeds)[0],1), name='source_edge_types')
            source_rel_embeds = tf.math.l2_normalize(
                self.transform_source_by_edge_type([source_edge_types, source_embeds], training=training),
                axis=1
            )
            target_edge_types = tf.tile(edge_type, (tf.shape(target_embeds)[0],1), name='target_edge_types')
            target_rel_embeds = tf.math.l2_normalize(
                self.transform_target_by_edge_type([target_edge_types, target_embeds], training=training),
                axis=1
            )
            similarities = tf.einsum('sr,tr->st', source_rel_embeds, target_rel_embeds, name='cosine_similarities')
            return (similarities + tf.constant(1, dtype=self.dtype)) / tf.constant(2, dtype=self.dtype)


class KGRecommender(tf.keras.Model):
    def __init__(self, num_cubes, num_decks, num_cards, batch_size, chunk_size, edges, num_layers=3,
                 entity_dims=128, edge_type_dims=64, initializer='glorot_uniform',
                 edge_type_activation='selu', message_dropout=0.2, num_intents=8,
                 intent_activation='softmax', message_activation='linear', jit_scope=contextlib.nullcontext,
                 **kwargs):
        super(KGRecommender, self).__init__(**kwargs, dynamic=False)
        self.edges = edges
        self.num_decks = num_decks
        self.num_cubes = num_cubes
        self.num_cards = tf.constant(num_cards, dtype=tf.int32, name='num_cards')
        self.batch_size = tf.constant(batch_size, dtype=tf.int32, name='batch_size')
        self.chunk_size = tf.constant(chunk_size, dtype=tf.int32, name='chunk_size')
        self.num_layers = tf.constant(num_layers, dtype=tf.int32, name='num_layers')
        self.entity_dims = tf.constant(entity_dims, dtype=tf.int32, name='entity_dims')
        self.edge_type_dims = tf.constant(edge_type_dims, dtype=tf.int32, name='edge_type_dims')
        self.initializer = initializer
        self.edge_type_activation = edge_type_activation
        self.message_dropout = tf.constant(message_dropout, dtype=tf.float32, name='message_dropout')
        self.num_intents = tf.constant(num_intents, dtype=tf.int32, name='num_intents')
        self.intent_activation = intent_activation
        self.message_activation = message_activation
        self.jit_scope = jit_scope

        self.num_edge_types = tf.constant(NUM_EDGE_TYPES, dtype=tf.int32, name='num_edge_types')
        self.num_node_types = tf.constant(NUM_NODE_TYPES, dtype=tf.int32, name='num_node_types')
        self.cube_start_index = tf.constant(0, dtype=tf.int32, name='cube_start_index')
        self.deck_start_index = tf.constant(self.cube_start_index + self.num_cubes, dtype=tf.int32, name='deck_start_index')
        self.card_start_index = tf.constant(self.deck_start_index + self.num_decks, dtype=tf.int32, name='card_start_index')
        self.placeholder_index = tf.constant(self.card_start_index + self.num_cards, dtype=tf.int32, name='placeholder_index')
        self.batch_start_index = tf.constant(self.placeholder_index + 1, dtype=tf.int32, name='batch_start_index')
        self.num_entities = tf.constant(self.batch_start_index + self.batch_size, dtype=tf.int32, name='num_entities')
        node_types = [-1 for _ in range(self.card_start_index)]
        for i in range(self.cube_start_index, self.cube_start_index + self.num_cubes):
            node_types[i] = NODE_TYPES.CUBE
        for i in range(self.deck_start_index, self.deck_start_index + self.num_decks):
            node_types[i] = NODE_TYPES.DECK
        self.defaulted_node_types = tf.constant(node_types, dtype=tf.int32, name='defaulted_node_types')
        self.sources = tf.constant(edges[:, 0], dtype=tf.int32, name='base_sources')
        self.targets = tf.constant(edges[:, 1], dtype=tf.int32, name='base_targets')
        self.edge_types = tf.constant(tf.one_hot(edges[:, 2], self.num_edge_types, dtype=self.dtype), name='base_edge_types')
        self.prediction_edge_type = tf.constant(tf.one_hot(EDGE_TYPES.CUBE_CONTAINS,
                                                           self.num_edge_types, dtype=self.dtype),
                                                shape=(1, self.num_edge_types), name='prediction_edge_type')
        self.batch_types = tf.constant(NODE_TYPES.CUBE, shape=(self.batch_size,), dtype=tf.int32, name='batch_types')
        self.placeholder_embed = tf.constant(0, shape=(1, self.entity_dims), dtype=self.dtype, name='placeholder_embed')

    def get_config(self):
        config = super(KGRecommender, self).get_config()
        config.update({
            'edges': self.edges,
            'num_cubes': self.num_cubes,
            'num_decks': self.num_decks,
            'num_cards': self.num_cards,
            'chunk_size': self.chunk_size,
            'batch_size': self.batch_size,
            'num_layers': self.num_layers,
            'entity_dims': self.entity_dims,
            'edge_type_dims': self.edge_type_dims,
            'initializer': self.initializer,
            'edge_type_activation': self.edge_type_activation,
            'message_dropout': self.message_dropout,
            'num_intents': self.num_intents,
            'intent_activation': self.intent_activation,
            'message_activation': self.message_activation,
        })
        return config

    def build(self, input_shape):
        self.card_embeddings = self.add_weight(shape=(self.num_cards, self.entity_dims),
                                               initializer=self.initializer, trainable=True,
                                               name='card_embeddings')
        self.type_embeddings = self.add_weight(shape=(self.num_node_types, self.entity_dims),
                                               initializer=self.initializer, trainable=True,
                                               name='node_type_embeddings')
        self.kgin = KGIN(num_entities=self.num_entities, num_layers=self.num_layers, chunk_size=self.chunk_size,
                         entity_dims=self.entity_dims, num_edge_types=self.num_edge_types, num_node_types=self.num_node_types,
                         edge_type_dims=self.edge_type_dims, initializer=self.initializer,
                         edge_type_activation=self.edge_type_activation,
                         message_dropout=self.message_dropout, num_intents=self.num_intents,
                         intent_activation=self.intent_activation, message_activation=self.message_activation,
                         jit_scope=self.jit_scope, name='kg_embeddings')

    @tf.function()
    def _add_new_edges(self, new_sources, new_targets, new_edge_types, training=False):
        with self.jit_scope():
            new_sources = tf.reshape(new_sources, (-1,), name='flattened_new_sources')
            new_targets = tf.reshape(new_targets, (-1,), name='flattened_new_targets')
            new_edge_types = tf.reshape(tf.cast(new_edge_types, dtype=self.dtype), (-1, self.num_edge_types),
                                        name='flattened_new_edge_types')
            sources = tf.reshape(tf.concat([self.sources, new_sources], 0), (-1, self.chunk_size), name='sources')
            targets = tf.reshape(tf.concat([self.targets, new_targets], 0), (-1, self.chunk_size), name='targets')
            edge_types = tf.reshape(tf.concat([self.edge_types, new_edge_types], 0),
                                    (-1, self.chunk_size, self.num_edge_types), name='edge_types')
            return sources, targets, edge_types

    @tf.function()
    def call(self, new_edges, training=False):
        with self.jit_scope():
            new_sources, new_targets, new_edge_types = new_edges
            sources, targets, edge_types = self._add_new_edges(
                new_sources, new_targets, new_edge_types, training=True
            )
            entity_embeds = tf.concat([
                tf.gather(self.type_embeddings, self.defaulted_node_types, name='defaulted_node_embeddings'),
                self.card_embeddings,
                self.placeholder_embed,
                tf.gather(self.type_embeddings, self.batch_types, name='batch_node_embeddings'),
            ], 0, name='original_entity_embeds')
            entity_embeds = self.kgin(
                [entity_embeds, sources, targets, edge_types], training=training
            )
            predictions = self.kgin.predict_links(
                entity_embeds[-self.batch_size:], entity_embeds[self.card_start_index:self.card_start_index + self.num_cards],
                self.prediction_edge_type, training=training
            )
            return predictions
