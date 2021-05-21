import contextlib
from sys import meta_path
from typing import Union

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import tensorflow_addons as tfa


def convert_spmatrix_to_sparsetensor(coo: sp.coo_matrix, dtype=tf.float32) -> tf.SparseTensor:
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.sparse.reorder(tf.SparseTensor(indices, tf.convert_to_tensor(coo.data, dtype=dtype), coo.shape))


class MetapathEmbed(tf.keras.layers.Layer):
    def __init__(self, metapath_dims: int, use_sparse=False, l1_weight=0.0, l2_weight=0.005,
                 jit_scope=contextlib.nullcontext, **kwargs):
        super(MetapathEmbed, self).__init__(**kwargs)
        self.metapath_dims = metapath_dims
        self.use_sparse = use_sparse
        self.map_embeddings = tf.keras.layers.Dense(metapath_dims, activation='swish', name=f'{self.name}/map_card_embeddings')
        self.jit_scope = jit_scope
        # self.activity_regularizer = tf.keras.regularizers.l1_l2(l1=l1_weight, l2=l2_weight)

    # def get_config(self):
    #     return super(MetapathEmbed, self).get_config() | {
    #         'metapath_dims': self.metapath_dims,
    #         'use_sparse': self.use_sparse,
    #     }

    @tf.function()
    def call(self, inputs: tuple[tf.SparseTensor, Union[tf.SparseTensor, tf.Tensor], tf.Tensor]):
        with self.jit_scope():
            batch_pools, metapath, card_embeddings = inputs
            transformed_embeddings = self.map_embeddings(card_embeddings)
            if self.use_sparse:
                path_embeddings = tf.sparse.sparse_dense_matmul(metapath, transformed_embeddings, adjoint_a=True, name=f'{self.name}/card_embeddings')
            else:
                path_embeddings = tf.linalg.matmul(metapath, transformed_embeddings, transpose_a=True) # [cards, metapath_dims]
            return tf.sparse.sparse_dense_matmul(batch_pools, path_embeddings, name=f'{self.name}/pool_embeds')


class EmbedSet(tf.keras.layers.Layer):
    def __init__(self, use_sparses: tuple[bool, ...], embed_dims: int, metapath_dims: int, num_heads: int,
                 dropout: float = 0.0, margin=1.0, l1_weight=0.0, l2_weight=0.001,
                 jit_scope=contextlib.nullcontext, **kwargs):
        assert metapath_dims >= num_heads > 0 and metapath_dims % num_heads == 0, 'num_heads must divide metapath_dims.'
        super(EmbedSet, self).__init__(**kwargs)
        self.use_sparses = use_sparses
        self.embed_dims = embed_dims
        self.metapath_dims = metapath_dims
        self.num_heads = num_heads
        self.dropout = dropout
        self.margin = margin
        self.jit_scope = jit_scope
        self.metapath_layers = tuple(MetapathEmbed(metapath_dims=metapath_dims, use_sparse=use_sparse,
                                                   l1_weight=l1_weight, l2_weight=l2_weight, dynamic=False,
                                                   jit_scope=jit_scope, name=f'{self.name}/metapath_{i}')
                                     for i, use_sparse in enumerate(use_sparses))
        self.metapath_attention = tf.keras.layers.MultiHeadAttention(num_heads, metapath_dims // num_heads,
                                                                     dropout=dropout,
                                                                     use_bias=True, bias_initializer='zeros',
                                                                     name=f'{self.name}/metapath_attention')
        self.embed_projection = tf.keras.layers.Dense(metapath_dims, activation='linear', use_bias=True,
                                                      name=f'{self.name}/embed_projection')

    # def get_config(self):
    #     return super(EmbedSet, self).get_config() | {
    #         'use_sparses': self.use_sparses,
    #         'embed_dims': self.embed_dims,
    #         'metapath_dims': self.metapath_dims,
    #         'num_heads': self.num_heads,
    #         'dropout': self.dropout,
    #         'margin': self.margin,
    #     }

    @tf.function()
    def call(self, inputs: tuple[tf.SparseTensor, tuple[Union[tf.SparseTensor, tf.Tensor], ...], tf.Tensor],
             training=False):
        with self.jit_scope():
            pools, metapaths, card_embeddings = inputs
            path_embeds = tf.stack([layer((pools, metapath, card_embeddings), training=training)
                                    for metapath, layer in zip(metapaths, self.metapath_layers)],
                                   axis=1, name=f'{self.name}/path_embeds')
            attended_embeds = tf.reduce_sum(self.metapath_attention(path_embeds, path_embeds, training=training),
                                             axis=1, name=f'{self.name}/attended_path_embeds')
            # pool_embeds = tf.math.l2_normalize(attended_embeds, axis=1, name=f'{self.name}/pool_embeds')
            pool_embeds = attended_embeds
            # projected_cards = tf.math.l2_normalize(self.embed_projection(card_embeddings),
            #                                        axis=1, name=f'{self.name}/projected_cards')
            projected_cards = self.embed_projection(card_embeddings, training=training)
            similarities = tf.einsum('be,ce->bc', pool_embeds, projected_cards, name=f'{self.name}/pool_similarities')
            # return tf.cast((similarities + self.margin) / (1 + self.margin), dtype=tf.float32)
            # return tf.nn.sigmoid(similarities)
            return similarities


class MetapathRecommender(tf.keras.Model):
    def __init__(self, card_metapaths: tuple[sp.spmatrix, ...],
                 embed_dims=128, metapath_dims=64, num_heads=16, dropout=0.0, margin=1.0,
                 decks_weight = 1.0, jit_scope=contextlib.nullcontext, cube_metrics=[],
                 deck_metrics=[], l1_weight=0.01, l2_weight=0.1, **kwargs):
        num_cards = card_metapaths[0].shape[0]
        # inputs = (
        #     tf.keras.Input(shape=(num_cards,), sparse=True),
        #     tf.keras.Input(shape=(num_cards,), sparse=True),
        # )
        # super(MetapathRecommender, self).__init__(inputs=inputs, **kwargs)
        super(MetapathRecommender, self).__init__(**kwargs)
        # self.card_metapaths = card_metapaths
        # self.embed_dims = embed_dims
        # self.metapath_dims = metapath_dims
        # self.num_heads = num_heads
        # self.dropout = dropout
        # self.margin = margin
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.jit_scope = jit_scope
        self.decks_weight = decks_weight
        self.cube_metrics = cube_metrics
        self.deck_metrics = deck_metrics
        self.l2_reg_metric = tf.keras.metrics.Mean(name='l2_reg')
        self.l1_reg_metric = tf.keras.metrics.Mean(name='l1_reg')
        self.cube_loss_metric = tf.keras.metrics.Mean(name='cube_loss')
        self.deck_loss_metric = tf.keras.metrics.Mean(name='deck_loss')
        self.loss_metric = tf.keras.metrics.Mean(name='loss')
        threshold = 2**31 / metapath_dims
        use_sparses = tuple(metapath.getnnz() < threshold for metapath in card_metapaths)
        print('Use Sparse flags: ', use_sparses)
        self.metapaths = tuple(convert_spmatrix_to_sparsetensor(metapath, dtype=self.compute_dtype)
                                   if use_sparse else
                                   tf.convert_to_tensor(metapath.toarray(), dtype=self.compute_dtype)
                               for metapath, use_sparse in zip(card_metapaths, use_sparses))
        self.cube_embeddings = EmbedSet(use_sparses=use_sparses, embed_dims=embed_dims, metapath_dims=metapath_dims,
                                        num_heads=num_heads, dropout=dropout, margin=margin,
                                        l1_weight=l1_weight, l2_weight=l2_weight, jit_scope=jit_scope,
                                        name=f'{self.name}/cube_embeddings')
        self.deck_embeddings = EmbedSet(use_sparses=use_sparses, embed_dims=embed_dims, metapath_dims=metapath_dims,
                                        num_heads=num_heads, dropout=dropout, margin=margin,
                                        l1_weight=l1_weight, l2_weight=l2_weight, jit_scope=jit_scope,
                                        name=f'{self.name}/deck_embeddings')
        self.card_embeddings = self.add_weight(f'{self.name}/card_embeddings', (num_cards, embed_dims),
                                               trainable=True, initializer='glorot_uniform')

    # def get_config(self):
    #     return super(MetapathRecommender, self).get_config() | {
    #         'card_metapaths': self.card_metapaths,
    #         'embed_dims': self.embed_dims,
    #         'metapath_dims': self.metapath_dims,
    #         'num_heads': self.num_heads,
    #         'attention_dropout': self.attention_dropout,
    #         'margin': self.margin,
    #     }

    @tf.function()
    def call(self, inputs: tuple[tf.SparseTensor, tf.SparseTensor], training=False):
        cubes, decks = inputs
        decks_mask = decks > 0
        cubes = cubes / tf.reduce_sum(tf.minimum(cubes, 1), axis=1, keepdims=True)
        decks = decks / tf.reduce_sum(tf.minimum(decks, 1), axis=1, keepdims=True)
        cubes = tf.sparse.from_dense(cubes)
        decks = tf.sparse.from_dense(decks)
        cube_predictions = self.cube_embeddings((cubes, self.metapaths, self.card_embeddings), training=training)
        deck_predictions = self.deck_embeddings((cubes, self.metapaths, self.card_embeddings), training=training)
        return cube_predictions, deck_predictions

    @tf.function()
    def train_step(self, data):
        x, y = data
        cubes, decks = x
        cubes_true, decks_true = y
        flat_decks_mask = tf.reshape(decks, (-1,), name='flat_decks_mask') > 0
        flat_decks_true = tf.boolean_mask(tf.reshape(decks_true, (-1,)), flat_decks_mask, name='masked_deck_true')

        with self.jit_scope():
            with tf.GradientTape() as tape:
                cubes_pred, decks_pred = self(x, training=True)
                cube_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(cubes_true, cubes_pred, from_logits=True))
                flat_decks_pred = tf.boolean_mask(tf.reshape(decks_pred, (-1,)), flat_decks_mask, name='masked_deck_pred')
                deck_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(flat_decks_true, flat_decks_pred, from_logits=True))
                l1_regularization = tf.reduce_mean(tf.norm(tf.cast(self.card_embeddings, dtype=cube_loss.dtype), ord=1, axis=1))
                l2_regularization = tf.reduce_mean(tf.norm(tf.cast(self.card_embeddings, dtype=cube_loss.dtype), ord=2, axis=1))
                loss = cube_loss + self.decks_weight * deck_loss + self.l1_weight * l1_regularization\
                                 + self.l2_weight * l2_regularization

            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            self.loss_metric.update_state(loss)
            self.cube_loss_metric.update_state(cube_loss)
            self.deck_loss_metric.update_state(deck_loss)
            self.l2_reg_metric.update_state(l2_regularization)
            cubes_prob = tf.nn.sigmoid(cubes_pred)
            for metric in self.cube_metrics:
                metric.update_state(cubes_true, cubes_prob)
            flat_decks_prob = tf.nn.sigmoid(flat_decks_pred)
            for metric in self.deck_metrics:
                metric.update_state(flat_decks_true, flat_decks_prob)
        return {
            'loss': self.loss_metric.result(),
            'cube_loss': self.cube_loss_metric.result(),
            'deck_loss': self.deck_loss_metric.result(),
            'l2_reg': self.l2_reg_metric.result(),
            'l1_reg': self.l1_reg_metric.result(),
        } | {
            metric.name: metric.result() for metric in self.cube_metrics
        } | {
            metric.name: metric.result() for metric in self.deck_metrics
        }

    @property
    def metrics(self):
        return [self.loss_metric, self.deck_loss_metric, self.cube_loss_metric, self.l1_reg_metric,
                self.l2_reg_metric] + self.cube_metrics + self.deck_metrics
