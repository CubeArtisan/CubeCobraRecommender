from enum import Enum

import tensorflow as tf
from tqdm import tqdm

RELATIONS = Enum('RELATIONS', (
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
NUM_RELATIONS = len(RELATIONS)
NODE_TYPES = Enum('NODE_TYPES', (
    'CUBE',
    'DECK',
    'CARD',
), start=0, type=int, module=__name__)
NUM_NODE_TYPES = len(NODE_TYPES)

class MultiDense(tf.keras.layers.Layer):
    def __init__(self, dims, out_dims, initializer='glorot_uniform', use_bias=True,
                 activation='linear', **kwargs):
        super(MultiDense, self).__init__(**kwargs)
        self.dims = dims
        self.out_dims = out_dims
        self.initializer = initializer
        self.activation = activation
        self.use_bias = use_bias

    def build(self, input_shape):
        self.weights = self.add_weight(shape=(self.dims, input_shape[1][-1], self.out_dims),
                                       trainable=True, initializer=self.initializer, name='weights_lookup')
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.dims, self.out_dims),
                                        trainable=True, initializer=self.initializer, name='biases_lookup')
        self.activation_layer = tf.keras.layers.Activation(self.activation)

    def call(self, lookups, values, training=False):
        weights = tf.gather(self.weights, lookups)
        mapped = values @ weights
        if self.use_bias:
            biases = tf.gather(self.bias, lookups)
            mapped = mapped + biases
        return self.activation_layer(mapped)

    def get_config(self):
        config = super(MultiDense, self).get_config()
        config.update({
            'dims': self.dims,
            'out_dims': self.out_dims,
            'initializer': self.initializer,
            'activation': self.activation,
            'use_bias': self.use_bias,
        })
        return config


class KGIN(tf.keras.layers.Layer):
    def __init__(self, batch_size, edges, num_entities, num_layers=3, entity_dims=128, normalize=None,
                 relation_dims=64, initializer='glorot_uniform', return_weights=True,
                 relation_activation='selu', message_dropout=0.2, num_intents=8,
                 intent_activation='softmax', message_activation='linear', **kwargs):
        super(KGIN, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.edges = edges
        self.num_entities = num_entities
        self.num_layers = num_layers
        self.entity_dims = entity_dims
        self.normalize = normalize
        self.relation_dims = relation_dims
        self.initializer = initializer
        self.return_weights = return_weights
        self.relation_activation = relation_activation
        self.message_dropout = message_dropout
        self.num_intents = num_intents
        self.intent_activation = intent_activation
        self.message_activation= message_activation

    def get_config(self):
        config = super(KGIN, self).get_config()
        config.update({
            'batch_size': self.batch_size,
            'edges': self.edges,
            'num_entities': self.num_entities,
            'num_layers': self.num_layers,
            'entity_dims': self.entity_dims,
            'normalize': self.normalize,
            'relation_dims': self.relation_dims,
            'initializer': self.initializer,
            'return_weights': self.return_weights,
            'relation_activation': self.relation_activation,
            'message_dropout': self.message_dropout,
            'num_intents': self.num_intents,
            'intent_activation': self.intent_activation,
            'message_activation': self.message_activation,
        })
        return config

    def build(self, input_shape):
        self.entity_embeddings = self.add_weight(shape=(self.num_entities - self.batch_size, self.entity_dims),
                                                 initializer=self.initializer, trainable=True,
                                                 name='entity_embeddings')
        self.default_embeddings = tf.keras.layers.Embedding(NUM_NODE_TYPES, self.entity_dims,
                                                            embeddings_initializer=self.initializer,
                                                            dtype=self.dtype, name='default_embeddings')
        self.relation_embeddings = tf.keras.layers.Embedding(NUM_RELATIONS, self.relation_dims,
                                                             embeddings_initializer=self.initializer,
                                                             dtype=self.dtype, name='relation_embeddings')
        self.transform_by_relation = MultiDense(NUM_RELATIONS, self.relation_dims, initializer=self.initializer,
                                                use_bias=True, activation=self.relation_activation,
                                                dtype=self.dtype, name='transform_by_relation')
        self.message_for_relation = MultiDense(NUM_RELATIONS, self.entity_dims, initializer=self.initializer,
                                               use_bias=False, activation=self.message_activation,
                                               dtype=self.dtype, name='message_for_relation')
        self.intent_maps = MultiDense(NUM_NODE_TYPES, self.num_intents, activation='softmax',
                                      use_bias=True, initializer=self.initializer, dtype=self.dtype,
                                      name='intent_maps')
        self.intents_by_node = MultiDense(NUM_NODE_TYPES, NUM_RELATIONS, activation=self.intent_activation,
                                          use_bias=True, initializer=self.initializer, dtype=self.dtype,
                                          name='intents_by_node')
        self.message_dropout_layer = tf.keras.layers.Dropout(self.message_dropout, name='message_dropout')

    def pass_messages(self, sources, source_types, targets, target_types, relations, edges,
                      entity_embeds, relation_embeds, training=False):
        source_embeds = tf.gather(entity_embeds, sources)
        target_embeds = tf.gather(entity_embeds, targets)
        target_intents = self.intent_maps(target_types, target_embeds, training=training)
        target_relation_weights = self.intents_by_node(target_type, target_intents, training=training)
        intent_weights = tf.gather(target_relation_weights, relations, batch_dims=1)
        source_rel_embeds = self.transform_by_relation(relations, source_embeds, training=training)
        scores = tf.einsum('er,er,e->e', source_rel_embeds, relation_embeds, intent_weights)
        exp_scores = tf.math.exp(scores)
        exp_scores_by_target = tf.math.reciprocal_no_nan(tf.scatter_nd(targets, exp_scores, (self.num_entities,)))
        distribution = exp_scores * tf.gather(exp_scores_by_target, targets)
        edge_weights = self.message_dropout_layer(distribution, training=training)
        messages = edge_weights * self.message_for_relation(relations, source_embeds, training=training)
        new_embeds = tf.scatter_nd(self.targets, messages, entity_embeds.shape)
        if self.normalize is not None:
            new_embeds = tf.linalg.normalize(new_embeds, ord=self.normalize, axis=1)
        if self.return_weights:
            return new_embeds, edge_weights
        else:
            return new_embeds

    def call(self, new_edges, new_node_types, training=False):
        edges = tf.concat([new_edges, self.edges], 0)
        sources, source_types, targets, target_types, relations = tf.unstack(edges, num=5, axis=1)
        relation_embeds = self.relation_embeddings(relations)
        entity_embeds = tf.concat([self.default_embeddings(new_node_types), self.entity_embeddings], 0)
        layer_edge_weights = []
        for message_layer in self.message_layers:
            next_state = self.pass_message(sources, source_types, targets, target_types, relations,
                                           entity_embeds, relation_embeds, training=training)
            if self.return_weights:
                entity_embeds, edge_weights = next_state
                layer_edge_weights.append(edge_weights)
            else:
                entity_embeds = next_state
        if self.return_weights:
            return entity_embeds, layer_edge_weights
        else:
            return entity_embeds

    def predict_links(self, source_embeds, target_embeds, relation, training=False):
        source_rel_embeds = tf.linalg.normalize(self.transform_by_relation(relation, source_embeds,
                                                                           training=training),
                                                ord='euclidean', axis=1)
        target_rel_embeds = tf.linalg.normalize(self.transform_by_relation(relation, target_embeds,
                                                                           training=training),
                                                ord='euclidean', axis=1)
        return tf.einsum('sr,tr->st', source_rel_embeds, target_rel_embeds)


class KGRecommender(tf.keras.Model):
    def __init__(self, cubes, decks, num_cards, batch_size, num_layers=3, entity_dims=128,
                 normalize=None, relation_dims=64, initializer='glorot_uniform', return_weights=True,
                 relation_activation='selu', message_dropout=0.2, num_intents=8,
                 intent_activation='tanh', message_activation='linear', **kwargs):
        super(KGRecommender, self).__init__(**kwargs)
        self._self_setattr_tracking = False
        self.cubes = cubes
        self.decks = decks
        self._self_setattr_tracking = True
        self.num_cards = num_cards
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.entity_dims = entity_dims
        self.normalize = normalize
        self.relation_dims = relation_dims
        self.initializer = initializer,
        self.return_weights = return_weights
        self.relation_activation = relation_activation
        self.message_dropout = message_dropout
        self.num_intents = num_intents
        self.intent_activation = intent_activation
        self.message_activation = message_activation

        self.num_decks = len(decks)
        self.num_cubes = len(cubes)
        self.card_start_index = self.batch_size
        self.cube_start_index = self.card_start_index + self.num_cards
        self.deck_start_index = self.cube_start_index + self.num_cubes
        self.num_entities = self.deck_start_index + self.num_decks
        edges = []
        entity_types = (
            [NODE_TYPES.CUBE for _ in range(self.batch_size)]
            + [NODE_TYPES.CARD for _ in range(self.num_cards)]
            + [NODE_TYPES.CUBE for _ in range(self.num_cubes)]
            + [NODE_TYPES.DECK for _ in range(self.num_decks)]
        )
        for ient, t in tqdm(
                list(zip(enumerate(cubes), ('cube' for _ in cubes)))
                + list(zip(enumerate(decks), ('deck' for _ in decks)))
                + [(i, 'refl') for i in range(self.batch_size, self.num_entities)],
                dynamic_ncols=True, unit_scale=True):
            if t == 'cube':
                i, cube = ient
                cube_id = batch_size + num_cards + i
                self._add_edges(batch_size + num_cards + i, cube, RELATIONS.CUBE_CONTAINS,
                                RELATIONS.IN_CUBE, NODE_TYPES.CUBE, edges)
            elif t == 'deck':
                i, deck = ient
                deck_id = batch_size + num_cards + self.num_cubes + i
                self._add_edges(deck_id, deck['main'], RELATIONS.POOL_CONTAINS, RELATIONS.IN_POOL,
                                NODE_TYPES.DECK, edges)
                self._add_edges(deck_id, deck['main'], RELATIONS.MAIN_CONTAINS, RELATIONS.IN_MAIN,
                                NODE_TYPES.DECK, edges)
                self._add_edges(deck_id, deck['side'], RELATIONS.POOL_CONTAINS, RELATIONS.IN_POOL,
                                NODE_TYPES.DECK, edges)
                self._add_edges(deck_id, deck['side'], RELATIONS.SIDE_CONTAINS, RELATIONS.IN_SIDE,
                                NODE_TYPES.DECK, edges)
            elif t == 'refl':
                edges.append((ient, entity_types[ient], ient, entity_types[ient], RELATIONS.REFLEXIVE))
        self.edges = tf.constant(sorted(edges), dtype=tf.int32)
        self.card_indices = tf.range(self.card_start_index, self.card_start_index + self.num_cards)
        self.batch_indices = tf.range(0, self.batch_size)
        self.reflexive_edges = tf.stack([self.batch_indices, tf.fill(self.batch_indices.shape, NODE_TYPES.CUBE),
                                         self.batch_indices, tf.fill(self.batch_indices.shape, NODE_TYPES.CUBE),
                                         tf.fill(self.batch_indices.shape, RELATIONS.REFLEXIVE)], axis=1)
        self.new_node_types = tf.fill((self.batch_size,), NODE_TYPES.CUBE)

    def _add_edges(self, entity_id, card_ids, out_relation, in_relation, entity_type, edges):
        for card_id in card_ids:
            r_card_id = card_id + self.batch_size
            edges.append((entity_id, entity_type,     r_card_id, NODE_TYPES.CARD, out_relation))
            edges.append((r_card_id, NODE_TYPES.CARD, entity_id, entity_type,      in_relation))

    def get_config(self):
        config = super(KGRecommender, self).get_config()
        config.update({
            'cubes': self.cubes,
            'decks': self.decks,
            'num_cards': self.num_cards,
            'batch_size': self.batch_size,
            'num_layers': self.num_layers,
            'entity_dims': self.entity_dims,
            'normalize': self.normalize,
            'relation_dims': self.relation_dims,
            'initializer': self.initializer,
            'return_weights': self.return_weights,
            'relation_activation': self.relation_activation,
            'message_dropout': self.message_dropout,
            'num_intents': self.num_intents,
            'intent_activation': self.intent_activation,
            'message_activation': self.message_activation,
        })
        return config

    def build(self, input_shape):
        self.kgin = KGIN(batch_size=self.batch_size, edges=self.edges, num_entities=self.num_entities,
                         num_layers=self.num_layers, entity_dims=self.entity_dims, normalize=self.normalize,
                         relation_dims=self.relation_dims, initializer=self.initializer,
                         return_weights=self.return_weights, relation_activation=self.relation_activation,
                         message_dropout=self.message_dropout, num_intents=self.num_intents,
                         intent_activation=self.intent_activation, message_activation=self.message_activation,
                         dtype=self.dtype, name='kg_embeddings')

    def call(self, cubes, training=False):
        cube_indices, card_indices = tf.unstack(tf.where(cubes > 0), num=2, axis=1)
        card_edges = tf.stack([card_indices, tf.fill(card_indices.shape, NODE_TYPES.CARD),
                               cube_indices, tf.fill(cube_indices.shape, NODE_TYPES.CUBE),
                               tf.fill(cube_indices.shape, RELATIONS.IN_CUBE)], axis=1)
        new_edges = tf.concat([card_edges, self.reflexive_edges], axis=0)
        kgin_results = self.kgin(new_edges, self.new_node_types, training=training)
        if self.return_weights:
            entity_embeds, layer_edge_weights = kgin_results
        else:
            entity_embeds = kgin_results
        card_embeds = tf.gather(entity_embeds, self.card_indices)
        batch_cube_embeds = tf.gather(entity_embeds, self.batch_indices)
        return (self.kgin.predict_links(batch_cube_embeds, card_embeds, RELATIONS.CUBE_CONTAINS, training=training) + 1) / 2
