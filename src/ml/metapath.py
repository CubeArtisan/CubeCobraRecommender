import contextlib
from sys import meta_path
from typing import Union

import numpy as np
import scipy.sparse as sp
import tensorflow as tf


class MetapathEmbed(tf.keras.layers.Layer):
    def __init__(self, metapath_dims=128,
                 kernel_initializer='glorot_uniform', activation='swish',
                 jit_scope=contextlib.nullcontext, **kwargs):
        super(MetapathEmbed, self).__init__(**kwargs)
        self.metapath_dims = tf.constant(metapath_dims, dtype=tf.int32, name='metapath_dims')
        self.kernel_initializer = kernel_initializer
        self.activation = activation
        self.jit_scope = jit_scope

    # def get_config(self):
    #     return super(MetapathEmbed, self).get_config() | {
    #         'metapath_dims': self.metapath_dims,
    #         'kernel_initializer': self.kernel_initializer,
    #         'activation': self.activation,
    #     }

    def build(self, input_shape):
        embed_dims = input_shape[2][1]
        self.kernel = self.add_weight('kernel', (embed_dims, self.metapath_dims),
                                      initializer=self.kernel_initializer, trainable=True)
        # self.bias = self.add_weight('metapath_biases_1', (self.metapath_dims, 1),
        #                                          initializer='zeros', trainable=True)
        # self.activation_layer = tf.keras.layers.Activation(self.activation, name='activation')
        # super(MetapathEmbed, self).build(input_shape)

    @tf.function()
    def call(self, inputs: tuple[tf.SparseTensor, tf.Tensor]):
        with self.jit_scope():
            batch_pools, metapath, card_embeddings = inputs
            # path_card_embeddings = self.activation_layer(tf.einsum('ce,em->mc', card_embeddings, self.kernel)
            #                                              + self.bias)
            # path_card_matrix = tf.einsum('mc,cd->dm', path_card_embeddings, metapath, name='path_card_matrix')
            # return tf.linalg.matmul(batch_pools, path_card_matrix, b_is_sparse=True, name='path_embeds')
            return tf.einsum('ce,em,cd,bd->bm', card_embeddings, self.kernel, metapath, batch_pools, name='path_embeds')


class MetapathRecommender(tf.keras.Model):
    def __init__(self, card_metapaths: list[tf.Tensor],
                 embed_dims=64, metapath_dims=128, num_heads=16, attention_dropout=0.0, margin=0.5,
                 embedding_initializer='glorot_uniform', metapath_kernel_initializer='glorot_uniform',
                 attention_kernel_initializer='glorot_uniform', pool_kernel_initializer='glorot_uniform',
                 metapath_activation='swish', jit_scope=contextlib.nullcontext, **kwargs):
        super(MetapathRecommender, self).__init__(**kwargs)
        self.num_cards = tf.constant(len(card_metapaths[0]), dtype=tf.int32)
        self.embed_dims = tf.constant(embed_dims, dtype=tf.int32)
        self.metapath_dims = tf.constant(metapath_dims, dtype=tf.int32)
        assert metapath_dims >= num_heads > 0 and metapath_dims % num_heads == 0, 'num_heads must divide metapath_dims.'
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.margin = tf.constant(margin, dtype=self.compute_dtype)
        self.jit_scope = jit_scope
        self.embedding_initializer = embedding_initializer
        self.metapath_kernel_initializer = metapath_kernel_initializer
        self.attention_kernel_initializer = attention_kernel_initializer
        self.pool_kernel_initializer = pool_kernel_initializer
        self.metapath_activation = metapath_activation
        self.card_metapaths = [tf.constant(metapath, dtype=self.compute_dtype) for metapath in card_metapaths]
        self.min_prediction = tf.constant(2**-12, dtype=self.compute_dtype)
        self.max_prediction = tf.constant(1 - 3 / 2**11, dtype=self.compute_dtype)

    # def get_config(self):
    #     return super(MetapathRecommender, self).get_config() | {
    #         'card_metapaths': self.card_metapaths,
    #         'embed_dims': self.embed_dims,
    #         'metapath_dims': self.metapath_dims,
    #         'num_heads': self.num_heads,
    #         'attention_dropout': self.attention_dropout,
    #         'margin': self.margin,
    #         'embedding_initializer': self.embedding_initializer,
    #         'metapath_kernel_initializer': self.metapath_kernel_initializer,
    #         'attention_kernel_initializer': self.attention_kernel_initializer,
    #         'pool_kernel_initializer': self.pool_kernel_initializer,
    #         'metapath_activation': self.metapath_activation,
    #     }

    def build(self, input_shape):
        self.card_embeddings = self.add_weight('card_embeddings', (self.num_cards, self.embed_dims),
                                               initializer=self.embedding_initializer, trainable=True)
        self.cube_metapath_layers = tuple(MetapathEmbed(metapath_dims=self.metapath_dims,
                                                        kernel_initializer=self.metapath_kernel_initializer,
                                                        activation=self.metapath_activation,
                                                        jit_scope=self.jit_scope,
                                                        name=f'cube_metapath_{i}')
                                          for i, metapath in enumerate(self.card_metapaths))
        self.metapath_attention_1 = tf.keras.layers.MultiHeadAttention(self.num_heads, self.metapath_dims // self.num_heads,
                                                                       dropout=self.attention_dropout,
                                                                       kernel_initializer=self.attention_kernel_initializer,
                                                                       use_bias=False, bias_initializer='zeros',
                                                                       name='metapath_attention_1')
        self.pool_kernel_1 = self.add_weight('pool_kernel_1', (self.metapath_dims, self.embed_dims),
                                             initializer=self.pool_kernel_initializer, trainable=True)
        self.pool_bias_1 = self.add_weight('pool_bias_1', (self.embed_dims), initializer='zeros',
                                           trainable=True)
        self.deck_metapath_layers = tuple(MetapathEmbed(metapath_dims=self.metapath_dims,
                                                        kernel_initializer=self.metapath_kernel_initializer,
                                                        activation=self.metapath_activation,
                                                        jit_scope=self.jit_scope,
                                                        name=f'deck_metapath_{i}')
                                          for i, metapath in enumerate(self.card_metapaths))
        self.metapath_attention_2 = tf.keras.layers.MultiHeadAttention(self.num_heads, self.metapath_dims // self.num_heads,
                                                                       dropout=self.attention_dropout,
                                                                       kernel_initializer=self.attention_kernel_initializer,
                                                                       use_bias=False, bias_initializer='zeros',
                                                                       name='metapath_attention_2')
        self.pool_kernel_2 = self.add_weight('pool_kernel_2', (self.metapath_dims, self.embed_dims),
                                           initializer=self.pool_kernel_initializer, trainable=True)
        self.pool_bias_2 = self.add_weight('pool_bias_2', (1, self.embed_dims), initializer='zeros',
                                         trainable=True)
        # super(MetapathRecommender, self).build(input_shape)

    def _inner(self, pools, metapath_layers, attention, pool_kernel, pool_bias, training=False):
        @tf.function()
        def func():
            with self.jit_scope():
                path_embeds = tf.stack([layer([pools, metapath, self.card_embeddings], training=training)
                                        for metapath, layer in zip(self.card_metapaths, metapath_layers)],
                                       axis=1, name='path_embeds')
                atteneded_path_embeds = tf.cast(attention(path_embeds, path_embeds, training=training), dtype=self.compute_dtype)
                pre_pool_embeds = tf.reduce_sum(atteneded_path_embeds, axis=1)
                return tf.einsum('bm,me->be', pre_pool_embeds, tf.cast(pool_kernel, dtype=self.compute_dtype))\
                     + tf.cast(pool_bias, dtype=self.compute_dtype)
        return func()

    def _inner1(self, pools, training=False):
        return lambda: self._inner(pools, self.cube_metapath_layers, self.metapath_attention_1,
                                   self.pool_kernel_1, self.pool_bias_1, training=training)

    def _inner2(self, pools, training=False):
        return lambda: self._inner(pools, self.deck_metapath_layers, self.metapath_attention_1,
                                   self.pool_kernel_2, self.pool_bias_2, training=training)

    @tf.function()
    def call(self, inputs: tuple[tf.SparseTensor, tf.Tensor], training=False):
        with self.jit_scope():
            pools = tf.cast(inputs[0], dtype=self.compute_dtype, name='pools')
            pool_embeds = tf.switch_case(inputs[1], [self._inner1(pools, training),
                                                     self._inner2(pools, training)])
            normalized_card_embeds = tf.math.l2_normalize(tf.cast(self.card_embeddings, dtype=self.compute_dtype),
                                                          axis=1, name='normalized_card_embeds')
            normalized_pool_embeds = tf.math.l2_normalize(pool_embeds, axis=1, name='normalized_pool_embeds')
            return tf.clip_by_value((tf.einsum('be,ce->bc', normalized_pool_embeds, normalized_card_embeds, name='prediction_scores') + self.margin) / (1 + self.margin),
                                    self.min_prediction, self.max_prediction)
