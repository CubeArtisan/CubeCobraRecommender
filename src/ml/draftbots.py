import numpy as np
import tensorflow as tf

from ..non_ml.parse_picks import MAX_IN_PACK, MAX_PICKED, MAX_SEEN, NUM_LAND_COMBS
from .timeseries.timeseries import log_timeseries


def get_mask(tensor):
    tensor._keras_mask = getattr(tensor, '_keras_mask', tf.cast(tf.ones_like(tensor), dtype=tf.bool))
    return tensor._keras_mask


def mask_to_zeros(tensor, name=None):
    mask = get_mask(tensor)
    if len(mask.shape) < len(tensor.shape):
        mask = tf.expand_dims(mask, -1)
    result = tensor * tf.cast(mask, dtype=tensor.dtype, name=name)
    result._keras_mask = get_mask(tensor)
    return result


def cast(tensor, dtype, name=None):
    result = tf.cast(tensor, dtype=dtype, name=name)
    result._keras_mask = get_mask(tensor)
    return result


def normalize(tensor, axis=None, epsilon=1e-04, name=None):
    result = tf.math.l2_normalize(tensor, axis=axis, epsilon=epsilon, name=name)
    result._keras_mask = get_mask(tensor)
    return result


class DraftBot(tf.keras.models.Model):
    def __init__(self, num_cards, temperature,  batch_size, embed_dims=64, num_heads=16, **kwargs):
        kwargs.update({'dynamic': False})
        super(DraftBot, self).__init__(**kwargs)
        self.oracle_weights = self.add_weight('oracle_weight_logits', shape=(3, 15, 6), initializer=
                                              'random_uniform', trainable=True)
        self.ratings_mult = tf.constant([0] + [1 for _ in range(num_cards - 1)], dtype=self.compute_dtype)
        self.ratings = self.add_weight('card_rating_logits', shape=(num_cards,),
                                       initializer='random_uniform', trainable=True)
        self.temperature = self.add_weight('temperature', shape=(), initializer=tf.constant_initializer(temperature),
                                           trainable=False, dtype=tf.float64)
        self.card_embeddings = self.add_weight('card_embeddings', shape=(num_cards, embed_dims),
                                               initializer='random_uniform', trainable=True)
        self.loss_metric = tf.keras.metrics.Mean()
        self.log_loss_metric = tf.keras.metrics.Mean()
        self.accuracy_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(1)
        self.top_2_accuracy_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(2)
        self.top_3_accuracy_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(3)
        self.average_prob_metric = tf.keras.metrics.Mean()
        self.default_target = tf.zeros((batch_size,), dtype=tf.float64)
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads, embed_dims, use_bias=False, name='pool_attention')

    def call(self, inputs, training=False):
        in_pack_cards, seen_cards, seen_counts,\
            picked_cards, picked_counts, coords, coord_weights,\
            prob_seens, prob_pickeds, prob_in_packs = inputs
        
        # with tf.experimental.async_scope():
            # picked_counts = tf.cast(tf.reshape(picked_counts, (-1,)), dtype=self.float_type, name='picked_counts')
            # seen_counts = tf.cast(tf.reshape(seen_counts, (-1,)), dtype=self.float_type, name='seen_counts')
            # coord_weights = tf.cast(coord_weights, dtype=self.float_type, name='coord_weights')
            # prob_seens = tf.cast(prob_seens, dtype=self.float_type, name='prob_seens')
            # prob_pickeds = tf.cast(prob_pickeds, dtype=self.float_type, name='prob_pickeds')
            # prob_in_packs = tf.cast(prob_in_packs, dtype=self.float_type, name='prob_in_packs')
        with tf.experimental.async_scope():
            in_pack_mask = tf.expand_dims(tf.cast(in_pack_cards > 0, dtype=tf.float64), 2)
            ratings = self.ratings_mult * tf.nn.sigmoid(self.ratings)
            oracle_weights = tf.nn.sigmoid(self.oracle_weights)
        with tf.experimental.async_scope():
            oracle_weights_orig = oracle_weights / tf.reduce_sum(oracle_weights, axis=2, keepdims=True)
            # oracle_weights_orig = oracle_weights / 6
            picked_ratings = tf.gather(ratings, picked_cards, name='picked_ratings')
            in_pack_ratings = tf.gather(ratings, in_pack_cards, name='in_pack_ratings')
            seen_ratings = tf.gather(ratings, seen_cards, name='seen_ratings')
            normalized_card_embeddings = tf.math.l2_normalize(self.card_embeddings, axis=1, epsilon=1e-04, name='normalized_card_embeddings')        
        with tf.experimental.async_scope():
            picked_card_embeds = tf.gather(self.card_embeddings, picked_cards, name='picked_card_embeds')
            normalized_in_pack_embeds = tf.gather(normalized_card_embeddings, in_pack_cards, name='normalized_in_pack_embeds')
            normalized_picked_embeds = tf.gather(normalized_card_embeddings, picked_cards, name='normalized_picked_embeds')
            # normalized_seen_embeds = tf.gather(normalized_card_embeddings, seen_cards, name='normalized_seen_embeds')
            oracle_weight_values = tf.gather_nd(oracle_weights_orig, coords, name='oracle_weight_values') # (-1, 4, 6)
        picked_card_embeds._keras_mask = picked_cards > 0
        with tf.experimental.async_scope():
            pool_attentions = self.attention(picked_card_embeds, picked_card_embeds)
            # pool_attentions = tf.cast(tf.math.real(tf.signal.rfft2d(tf.cast(picked_card_embeds, dtype=tf.float32))), dtype=self.compute_dtype)
            oracle_weights = tf.einsum('bxo,bx->bo', oracle_weight_values, coord_weights, name='oracle_weights')
        with tf.experimental.async_scope():
            # pool_embed = tf.math.l2_normalize(tf.einsum('bpe,blp->ble', picked_card_embeds, prob_pickeds, name='pool_embed'), axis=1, epsilon=1e-04, name='pool_embed_normalized')
            pool_embed = tf.math.l2_normalize(tf.einsum('bpe,blp->ble', pool_attentions, prob_pickeds, name='pool_embed'), axis=1, epsilon=1e-04, name='pool_embed_normalized')
            rating_weights, pick_synergy_weights, internal_synergy_weights, seen_synergy_weights,\
                colors_weights, openness_weights = tf.unstack(oracle_weights, num=6, axis=1)
        with tf.experimental.async_scope():
            rating_scores = tf.einsum('blc,bc,b->bcl', prob_in_packs, in_pack_ratings, rating_weights, name='rating_scores')
            pick_synergy_scores = tf.einsum(
                'ble,bce,blc,b->bcl',
                pool_embed,
                normalized_in_pack_embeds,
                prob_in_packs,
                pick_synergy_weights,
                name='pick_synergies_scores'
            )
            internal_synergy_scores = tf.einsum(
                'ble,bpe,blp,b,b->bl',
                pool_embed,
                normalized_picked_embeds,
                prob_pickeds,
                1 / tf.math.maximum(picked_counts, 1),
                internal_synergy_weights,
                name='internal_synergies_scores'
            )
            # seen_synergy_scores = tf.einsum(
                # 'ble,bse,bls,b,b->bl',
                # pool_embed, normalized_seen_embeds,
                # prob_seens,
                # 1 / tf.math.maximum(seen_counts, 1),
                # internal_synergy_weights,
                # name='internal_synergies_scores'
            # )
            seen_synergy_scores = tf.zeros_like(internal_synergies_scores)
            colors_scores = tf.einsum(
                'blp,bp,b,b->bl',
                prob_pickeds, picked_ratings,
                1 / tf.math.maximum(picked_counts, 1),
                colors_weights,
                name='colors_scores'
            )
            openness_scores = tf.einsum(
                'bls,bs,b,b->bl',
                prob_seens, seen_ratings,
                1 / tf.math.maximum(seen_counts, 1),
                openness_weights,
                name='openness_scores'
            )

        raw_scores = rating_scores + pick_synergy_scores \
            + tf.expand_dims(internal_synergy_scores + seen_synergy_scores
                             + colors_scores + openness_scores, 1)
        scores = tf.cast(self.temperature * raw_scores, dtype=tf.float64)
        max_scores = tf.stop_gradient(tf.reduce_max(scores, 2, keepdims=True))
        lse_scores = tf.math.reduce_sum(tf.math.exp(scores - max_scores) * in_pack_mask,
                                        2, name='lse_exp_scores')
        choice_probs = lse_scores / tf.math.reduce_sum(lse_scores, 1, keepdims=True)
        with tf.xla.experimental.jit_scope(compile_ops=False):
            tf.summary.scalar('weights/temperature', self.temperature)
            tf.summary.histogram('weights/card_embeddings', self.card_embeddings)
            tf.summary.histogram('weights/card_ratings', ratings)
            individual_scores = (rating_scores, pick_synergies_scores, internal_synergies_scores, seen_synergy_scores,
                                 colors_scores, openness_scores)
            for i, name in enumerate(('ratings', 'pick_synergy', 'internal_synergy', 'seen_synergy', 'colors', 'openness')):
                log_timeseries(f'weights/{name}_weights', tf.gather(oracle_weights_orig, i, axis=2), start_index=1)
                tf.summary.histogram(f'outputs/{name}_scores', individual_scores)
            tf.summary.histogram('outputs/scores', raw_scores)
            tf.summary.histogram('outputs/scores_with_temp', scores)
            tf.summary.histogram('outputs/max_scores', tf.math.reduce_max(raw_scores, [1, 2]))
            tf.summary.histogram('outputs/max_scores_with_temp', tf.math.reduce_max(scores, [1, 2]))
            tf.summary.histogram('outputs/score_differences', tf.math.reduce_max(raw_scores, 2) - tf.math.reduce_min(raw_scores, 2))
            tf.summary.histogram('outputs/score_differences_with_temp', tf.math.reduce_max(scores, 2) - tf.math.reduce_min(scores, 2))
            tf.summary.histogram('outputs/lse_scores', lse_scores)
            tf.summary.histogram('outputs/correct_lse_scores', lse_scores[:,0])
            tf.summary.histogram('outputs/incorrect_lse_scores', tf.boolean_mask(lse_scores[:,1:], in_pack_cards > 0))
            
            num_cards_in_pack = tf.reduce_sum(in_pack_mask, axis=[1,2])
            incorrect_probs = tf.boolean_mask(choice_probs[:,1:], in_pack_cards > 0)
            tf.summary.histogram('outputs/normalized_prob_correct', num_cards_in_pack * choice_probs[:,0])
            tf.summary.histogram('outputs/normalize_probs_incorrect', num_cards_in_pack * incorrect_probs)
            tf.summary.histogram('outputs/prob_correct', choice_probs[:,0])
            tf.summary.histogram('outputs/probs_incorrect', incorrect_probs)
        return choice_probs

    def train_step(self, data):
        with tf.GradientTape() as tape:
            probs = self(data[1], training=True)
            log_loss = tf.reduce_mean(-tf.math.log(probs[:, 0]))
            loss = log_loss

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        with tf.experimental.async_scope():
            self.loss_metric.update_state(loss)
            self.log_loss_metric.update_state(log_loss)
            self.accuracy_metric.update_state(self.default_target, probs)
            self.top_2_accuracy_metric.update_state(self.default_target, probs)
            self.top_3_accuracy_metric.update_state(self.default_target, probs)
            self.average_prob_metric.update_state(probs[:,0])
        result = {
            'loss': self.loss_metric.result(),
            'log_loss': self.log_loss_metric.result(),
            'accuracy': self.accuracy_metric.result(),
            'top_2_accuracy': self.top_2_accuracy_metric.result(),
            'top_3_accuracy': self.top_3_accuracy_metric.result(),
            'average_prob_correct': self.average_prob_metric.result(),
        }
        return result

    @property
    def metrics(self):
        return [self.loss_metric, self.log_loss_metric, self.accuracy_metric,
                self.top_2_accuracy_metric, self.top_3_accuracy_metric, self.average_prob_metric]
