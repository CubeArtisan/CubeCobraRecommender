import numpy as np
import tensorflow as tf

from ..non_ml.parse_picks import MAX_IN_PACK, MAX_PICKED, MAX_SEEN, NUM_LAND_COMBS
from .timeseries.timeseries import log_timeseries

def get_mask(tensor):
    tensor._keras_mask = getattr(tensor, '_keras_mask', tf.cast(tf.ones_like(tensor), dtype=tf.bool))
    return tensor._keras_mask

def mask_to_zeros(tensor):
    result = tensor * tf.cast(get_mask(tensor), dtype=tensor.dtype)
    result._keras_mask = get_mask(tensor)
    return result
    

def cast(tensor, dtype, name=None):
    result = tf.cast(tensor, dtype=dtype, name=name)
    result._keras_mask = get_mask(tensor)
    return result


def expand_dims(value, dim, name=None):
    result = tf.expand_dims(value, dim, name=name)
    result._keras_mask = tf.expand_dims(get_mask(value), dim)
    return result

    
def squeeze(value, name=None):
    result = tf.squeeze(value, -1, name=name)
    result._keras_mask = get_mask(value)
    return result

    
def add(value1, value2):
    result = value1 + value2
    result._keras_mask = tf.math.logical_and(get_mask(value1), get_mask(value2))
    return result

    
def mult(value1, value2):
    result = value1 * value2
    result._keras_mask = tf.math.logical_and(get_mask(value1), get_mask(value2))
    return result


class CardSynergy(tf.keras.models.Model):
    def __init__(self, num_cards, embed_dims=16, ragged=False, use_xla=False, hidden_activation='selu', regularizer=None):
        super(CardSynergy, self).__init__(name='CardSynergy')
        self.num_cards = num_cards
        self.embed_dims = embed_dims
        self.use_xla = use_xla
        self.ragged = ragged
        self.card_embeddings = tf.keras.layers.Embedding(num_cards, embed_dims, mask_zero=True,
                                                         embeddings_initializer='random_uniform', name='card_embeddings')
        self.upscale_1 = tf.keras.layers.Dense(2 * embed_dims, activation=hidden_activation, use_bias=False,
                                               kernel_initializer='random_uniform', kernel_regularizer=regularizer, name='upscale_1')
        self.upscale_2 = tf.keras.layers.Dense(4 * embed_dims, activation=hidden_activation, use_bias=False,
                                               kernel_initializer='random_uniform', kernel_regularizer=regularizer, name='upscale_2')
        self.downscale_1 = tf.keras.layers.Dense(2 * embed_dims, activation=hidden_activation, use_bias=False,
                                                 kernel_initializer='random_uniform', kernel_regularizer=regularizer, name='downscale_1')
        self.downscale_2 = tf.keras.layers.Dense(embed_dims, activation=hidden_activation, use_bias=False,
                                                 kernel_initializer='random_uniform', kernel_regularizer=regularizer, name='downscale_2')
        final_weights = np.random.uniform(-1, 1, (2 * embed_dims,))
        # self.final_weights = tf.Variable(tf.cast(final_weights, dtype=self.downscale_1.dtype), name='final_weights')
        self.batch_normalization = tf.keras.layers.BatchNormalization()

    def apply_dense_layers(self, x, name='', training=False):
        x = self.upscale_1(x)
        # x = self.upscale_2(x)
        # x = self.downscale_1(x)
        # x = self.downscale_2(x)
        x = tf.einsum('be,e->b', x, cast(self.final_weights, dtype=x.dtype))
        # x = tf.einsum('bcpe,es->bcps', x, self.final_weights)
        # x = tf.reduce_prod(x, -1)
        x_range = tf.stop_gradient(tf.math.reduce_max(x) - tf.math.reduce_min(x))
        x_stddev = tf.math.reduce_std(tf.clip_by_value(x, -5, 5))
        x_abs = tf.math.abs(x)
        loss_for_layer = tf.math.reduce_mean(tf.math.pow(tf.math.maximum(x_abs - 2, 0), 4) + tf.math.pow((x_abs + 1e-04), 1)) / (1 * (1 + 5 * tf.math.minimum(x_stddev, 1 / 5)))
        print(loss_for_layer.shape, loss_for_layer)
        with tf.xla.experimental.jit_scope(compile_ops=False):
            if tf.summary.experimental.get_step() is not None and tf.summary.experimental.get_step() % 25 == 0:
                # tf.summary.histogram(f'values/{name}_final_linear_results', x)
                x_mean = tf.math.reduce_mean(x)
                tf.summary.scalar(f'values/{name}_final_linear_mean', x_mean)
                tf.summary.scalar(f'values/{name}_final_linear_range', x_range)
                tf.summary.scalar(f'values/{name}_final_linear_stddev', x_stddev)
                tf.summary.scalar(f'outputs/{name}_loss_for_layer', loss_for_layer)
        self.add_loss(loss_for_layer)
        return tf.math.tanh(tf.clip_by_value(x, -5, 5))

    def call(self, inputs, training=False, name=None):
        card_indices_a, card_indices_b = inputs
        embeds_a = self.card_embeddings(card_indices_a)
        embeds_b = self.card_embeddings(card_indices_b)
        embeds_a_norm = tf.sqrt(1e-04 + tf.reduce_sum(embeds_a * embeds_a, -1, keepdims=True))
        embeds_b_norm = tf.sqrt(1e-04 + tf.reduce_sum(embeds_b * embeds_b, -1, keepdims=True))
        return tf.einsum('bce,bpe->bcp', embeds_a / embeds_a_norm, embeds_b / embeds_b_norm)
        return tf.reduce_sum(tf.expand_dims(embeds_a / embeds_a_norm, 2) * tf.expand_dims(embeds_b / embeds_b_norm, 1), -1)
        embed_combined = add(expand_dims(embeds_a, 2), expand_dims(embeds_b, 1))
        
        # embed_combined._keras_mask = tf.expand_dims(get_mask(embed_combined), -1)
        # return self.apply_dense_layers(mask_to_zeros(embed_combined))
        embed_combined = tf.ragged.boolean_mask(embed_combined, embed_combined._keras_mask)
        return embed_combined.with_flat_values(self.apply_dense_layers(embed_combined.flat_values, name=name, training=training))
        

class DraftBot(tf.keras.models.Model):
    def __init__(self, num_cards, temperature, float_type,
                 use_xla=False, ragged=False, embed_dims=64):
        super(DraftBot, self).__init__(name='DraftBot')
        self.float_type = float_type
        self.ragged = ragged
        oracle_weights = np.random.uniform(0, 1, (3, 15, 5))
        self.oracle_weights = tf.Variable(
            tf.cast(oracle_weights, dtype=tf.float32),
            dtype=tf.float32, name='oracles_logit_weights'
        )
        self.rating_mult = tf.constant([0] + [1 for _ in range(num_cards - 1)], dtype=float_type)
        card_ratings = np.random.uniform(0, 1, num_cards)
        card_ratings[0] = 0
        self.ratings = tf.Variable(
            tf.cast(tf.reshape(card_ratings, (-1,)), dtype=tf.float32),
            dtype=tf.float32, name='card_ratings',
        )
        self.temperature = tf.Variable(temperature, trainable=False, dtype=tf.float64)
        self.synergy_calculation = CardSynergy(num_cards, use_xla=use_xla,
                                               embed_dims=embed_dims, ragged=False) 
        self.use_xla = use_xla
        self.separate_gradients = True

    def call(self, inputs, training=False):
        with tf.xla.experimental.jit_scope(compile_ops=self.use_xla, separate_compiled_gradients=self.separate_gradients):
            if self.ragged:
                in_pack_cards, seen_cards, picked_cards, coords, coord_weights,\
                prob_seens, prob_pickeds, prob_in_packs = inputs[1]
                picked_counts = tf.math.reduce_sum(tf.ones_like(picked_cards, dtype=self.float_type), 1)
                seen_counts = tf.math.reduce_sum(tf.ones_like(seen_carsd, dtype=self.float_type), 1)
                picked_cards = tf.expand_dims(picked_cards, 2)
                in_pack_cards = tf.expand_dims(in_pack_cards, 2)
                seen_cards = tf.expand_dims(seen_cards, 2)
            else:
                in_pack_cards, seen_cards, seen_counts,\
                    picked_cards, picked_counts, coords, coord_weights,\
                    prob_seens, prob_pickeds, prob_in_packs = inputs
                prob_seens = tf.transpose(prob_seens, [0, 2, 1])
                prob_pickeds = tf.transpose(prob_pickeds, [0, 2, 1])
                prob_in_packs = tf.transpose(prob_in_packs, [0, 2, 1])
            picked_counts = tf.cast(tf.reshape(picked_counts, (-1,)), dtype=self.float_type)
            seen_counts = tf.cast(tf.reshape(seen_counts, (-1,)), dtype=self.float_type)
            coord_weights = tf.cast(coord_weights, dtype=self.float_type, name='coord_weights')
            prob_seens = tf.cast(prob_seens, dtype=self.float_type, name='prob_seens')
            prob_pickeds = tf.cast(prob_pickeds, dtype=self.float_type, name='prob_pickeds')
            prob_in_packs = tf.cast(prob_in_packs, dtype=self.float_type, name='prob_in_packs')
            
            min_rating = tf.stop_gradient(tf.math.reduce_min(self.ratings[1:]))
            # max_rating = tf.stop_gradient(tf.math.reduce_max(self.ratings[1:]))
            #min_rating = -tf.math.reduce_logsumexp(-self.ratings[1:])
            max_rating = tf.math.reduce_logsumexp(self.ratings[1:])
            rating_range = max_rating - min_rating
            oracle_weights = tf.cast(self.oracle_weights - tf.stop_gradient(tf.reduce_min(self.oracle_weights)), dtype=self.float_type) + 1e-04
            # oracle_weights = tf.cast(self.oracle_weights + tf.reduce_logsumexp(-self.oracle_weights), dtype=self.float_type) + 1e-02
            oracle_weights_orig = oracle_weights / tf.reduce_sum(oracle_weights, axis=2, keepdims=True)

            oracle_weight_values = tf.gather_nd(oracle_weights_orig, coords, name='oracle_weight_values') # (-1, 4, 5)
            picked_ratings = cast(tf.where(
                picked_cards > 0,
                (tf.gather(self.ratings, picked_cards) - min_rating) / rating_range,
                tf.zeros_like(picked_cards, dtype=tf.float32),
                name='picked_ratings'
            ), dtype=self.float_type)
            in_pack_ratings = cast(tf.where(
                in_pack_cards > 0,
                (tf.gather(self.ratings, in_pack_cards) - min_rating) / rating_range,
                tf.zeros_like(in_pack_cards, dtype=tf.float32),
                name='in_pack_ratings'
            ), dtype=self.float_type)
            seen_ratings = tf.where(
                seen_cards > 0,
                (tf.gather(self.ratings, seen_cards) - min_rating) / rating_range,
                tf.zeros_like(seen_cards, dtype=tf.float32),
                name='seen_ratings'
            )
            pick_synergies = cast(self.synergy_calculation((in_pack_cards, picked_cards), name='pick'), dtype=self.float_type, name='pick_synergies')
            internal_synergies = cast(self.synergy_calculation((picked_cards, picked_cards), name='internal'), dtype=self.float_type, name='internal_synergies')
            if self.ragged:
                in_pack_ratings = in_pack_ratings.to_tensor(shape=(None, MAX_IN_PACK))
                picked_ratings = picked_ratings.to_tensor(shape=(None, MAX_PICKED))
                seen_ratings = seen_ratings.to_tensor(shape=(None, MAX_SEEN))
            # mask = tf.cast(tf.ones_like(pick_synergies).to_tensor(shape=(None, MAX_IN_PACK, MAX_PICKED)), dtype=tf.bool)
            # pick_synergies = pick_synergies.to_tensor(shape=(None, MAX_IN_PACK, MAX_PICKED), name='pick_synergies')
            # pick_synergies._keras_mask = mask
            #mask = tf.cast(tf.ones_like(internal_synergies).to_tensor(shape=(None, MAX_PICKED, MAX_PICKED)), dtype=tf.bool)
            # internal_synergies = internal_synergies.to_tensor(shape=(None, MAX_PICKED, MAX_PICKED), name='internal_synergies')
            #internal_synergies._keras_mask = mask
            # if ragged:
                # internal_synergies_dense = internal_synergies.to_tensor((None, MAX_PICKED, MAX_PICKED))
                # internal_synergies_dense = internal_synergies_dense * (1 - tf.eye(MAX_PICKED))
                # internal_synergies = tf.ragged.boolean_mask(internal_synergies_dense, tf.cast(tf.ones_like(internal_synergies).to_tensor(internal_synergies_dense.shape), dtype=tf.bool))
            # else:
            eye = 1 - tf.eye(MAX_PICKED)
            mask = tf.logical_and(get_mask(internal_synergies), tf.cast(eye, dtype=tf.bool))
            internal_synergies = internal_synergies * eye
            internal_synergies._keras_mask = mask
            total_probs = tf.reduce_sum(prob_pickeds, 2, name='total_probs')
            oracle_weights = tf.einsum('bxo,bx->bo', oracle_weight_values, coord_weights, name='oracle_weights')
            rating_weights, pick_synergy_weights, internal_synergy_weights,\
                colors_weights, openness_weights = tf.unstack(oracle_weights, num=5, axis=1)

            rating_scores = tf.einsum('blc,bc,b->bcl', prob_in_packs, in_pack_ratings, rating_weights, name='rating_scores')
            pick_synergy_scores = tf.einsum(
                'bcp,blp,blc,b,b->bcl',
                pick_synergies,
                prob_pickeds, prob_in_packs,
                1 / tf.math.maximum(picked_counts, 1),
                pick_synergy_weights,
                name='pick_synergies_scores'
            )
            internal_synergy_scores = tf.einsum(
                'bcp,blp,blc,bl,b,b->bl',
                internal_synergies,
                prob_pickeds, prob_pickeds,
                1 / tf.math.maximum(total_probs, 1),
                1 / tf.math.maximum(picked_counts - 1, 1),
                internal_synergy_weights,
                name='internal_synergies_scores'
            )
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

            scores = self.temperature * tf.cast(rating_scores + pick_synergy_scores
                                                + tf.expand_dims(internal_synergy_scores + colors_scores + openness_scores, 1), dtype=tf.float64)
            #min_scores = tf.math.reduce_min(scores, 2, keepdims=True)
            max_scores = tf.stop_gradient(tf.reduce_max(scores, 2, keepdims=True))
            # scores = tf.ragged.boolean_mask(scores, in_pack_cards > 0)
            # min_scores = tf.ragged.boolean_mask(min_scores, in_pack_cards > 0)
            lse_scores = tf.math.reduce_sum(tf.math.exp(scores - max_scores) * tf.expand_dims(tf.cast(in_pack_cards > 0, dtype=scores.dtype), -1),
                                            2, name='lse_exp_scores')
            choice_probs = lse_scores / tf.math.reduce_sum(lse_scores, 1, keepdims=True)
            tensor_probs = choice_probs
            # tensor_probs = choice_probs.to_tensor(shape=(None, MAX_IN_PACK))
        with tf.xla.experimental.jit_scope(compile_ops=False):
            if (tf.summary.experimental.get_step() % 50) == 0:
                # tf.summary.histogram('weights/oracle_weights', oracle_weights)
                tf.summary.histogram('weights/card_ratings', (self.ratings - min_rating) / rating_range)
                for i, name in enumerate(('ratings', 'pick_synergy', 'internal_synergy', 'colors', 'openness')):
                    log_timeseries(f'weights/{name}_weights', tf.gather(oracle_weights_orig, i, axis=2), start_index=1)
                
                if (tf.summary.experimental.get_step() % 250) == 0:
                    filtered_pick = tf.boolean_mask(
                        tf.reshape(pick_synergies, (-1,)),
                        tf.reshape(tf.math.not_equal(pick_synergies, 0), (-1,)),
                    )
                    # filtered_internal = tf.boolean_mask(
                        # tf.reshape(internal_synergies, (-1,)), 
                        # tf.reshape(tf.not_equal(internal_synergies, 0), (-1,))
                    # )
                    tf.summary.histogram('values/pick_synergies', filtered_pick)
                    # tf.summary.histogram('values/internal_synergies', filtered_internal)
                    # tf.summary.histogram('values/synergies', tf.concat([filtered_internal, filtered_pick], 0))
                    tf.summary.histogram('outputs/scores', scores)#.flat_values)
                    tf.summary.histogram('outputs/max_scores', tf.math.reduce_max(scores, 1))#.flat_values)
                    tf.summary.histogram('outputs/score_differences', (tf.math.reduce_max(scores, 2) - tf.math.reduce_min(scores, 2)))#.flat_values)
                    tf.summary.histogram('outputs/lse_scores', lse_scores)#.flat_values)
                
                # tf.summary.histogram('outputs/choice_probs', tf.boolean_mask(choice_probs, choice_probs > 0))#.flat_values)
                tf.summary.histogram('outputs/prob_correct', tensor_probs[:,0])
                tf.summary.histogram('outputs/probs_incorrect', tf.boolean_mask(choice_probs[:,1:], choice_probs[:,1:] > 0))
               # tf.summary.histogram('outputs/correct_lse_scores', tf.math.log(lse_scores[:,0]))
        return tensor_probs