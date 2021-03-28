import numpy as np
import tensorflow as tf

from ..non_ml.parse_picks import MAX_IN_PACK, MAX_PICKED, MAX_SEEN, NUM_LAND_COMBS
from .timeseries.timeseries import log_timeseries

class DraftBot(tf.keras.models.Model):
    def __init__(self, card_ratings, card_embeddings, temperature,
                 internal_synergy_dropout_rate, pick_synergy_dropout_rate, float_type,
                 debugging=False):
        super(DraftBot, self).__init__(name='DraftBot')
        self.float_type = float_type
        print(self.float_type)
        oracle_weights = np.random.uniform(0, 1, (3, 15, 5))
        oracle_weights = tf.math.divide_no_nan(oracle_weights, tf.reduce_sum(oracle_weights, axis=2, keepdims=True))
        self.oracle_weights = tf.Variable(
            tf.cast(oracle_weights, dtype=tf.float32),
            dtype=tf.float32, name='oracles_logit_weights'
        )
        self.num_cards = len(card_ratings)
        self.rating_mult = tf.constant([0] + [1 for _ in card_ratings[1:]], dtype=float_type)
        card_ratings = np.random.uniform(0, 10, len(card_ratings)),
        card_ratings = tf.reshape(card_ratings, (-1,))
        self.ratings = tf.Variable(
            tf.cast(card_ratings, dtype=tf.float32),
            dtype=tf.float32, name='card_ratings',
        )
        card_embeddings = np.random.uniform(-1, 1, (self.num_cards, len(card_embeddings[0])))
        self.card_embeddings = tf.Variable(
            tf.cast(
                card_embeddings,
                # tf.math.divide_no_nan(card_embeddings, tf.norm(card_embeddings, axis=1, keepdims=True)),
                dtype=tf.float32),
            dtype=tf.float32, name='card_embeddings')
        self.temperature = tf.constant(temperature, dtype=self.float_type)
        self.internal_synergy_dropout = tf.keras.layers.Dropout(internal_synergy_dropout_rate, dtype=self.float_type,
                                                                name='internal_synergy_dropout')
        self.pick_synergy_dropout = tf.keras.layers.Dropout(pick_synergy_dropout_rate, dtype=self.float_type,
                                                            name='pick_synergy_dropout')
        self.debugging = debugging
        self.separate_gradients = True

    def call(self, inputs, training=False):
        in_pack_card_indices, seen_indices, seen_counts,\
            picked_card_indices, picked_counts, coords, coord_weights,\
            prob_seen_matrices,\
            prob_picked_matrices, prob_in_pack_matrices = inputs
        in_pack_card_indices = tf.reshape(tf.cast(in_pack_card_indices, dtype=tf.int32), (-1, MAX_IN_PACK), name='in_packs')
        seen_indices = tf.reshape(tf.cast(seen_indices, dtype=tf.int32), (-1, MAX_SEEN), name='seens')
        seen_counts = tf.reshape(tf.cast(seen_counts, dtype=self.float_type), (-1, 1), name='seen_counts')
        picked_card_indices = tf.reshape(tf.cast(picked_card_indices, dtype=tf.int32), (-1, MAX_PICKED), name='pickeds')
        picked_counts = tf.reshape(tf.cast(picked_counts, dtype=self.float_type), (-1, 1), name='picked_counts')
        coords = tf.reshape(tf.cast(coords, dtype=tf.int32), (-1, 4, 2), name='coords')
        coord_weights = tf.reshape(tf.cast(coord_weights, dtype=self.float_type), (-1, 4), name='coord_weights')

        with tf.xla.experimental.jit_scope(compile_ops=not self.debugging, separate_compiled_gradients=self.separate_gradients):
            prob_seens = tf.reshape(tf.cast(prob_seen_matrices, dtype=self.float_type) / 255,
                                    (-1, NUM_LAND_COMBS, MAX_SEEN), name='prob_seens')
            prob_pickeds = tf.reshape(tf.cast(prob_picked_matrices, dtype=self.float_type) / 255,
                                      (-1, NUM_LAND_COMBS, MAX_PICKED), name='prob_pickeds')
            prob_in_packs = tf.reshape(tf.cast(prob_in_pack_matrices, dtype=self.float_type) / 255,
                                       (-1, NUM_LAND_COMBS, MAX_IN_PACK), name='prob_in_packs')
            embeddings = tf.cast(self.card_embeddings, dtype=self.float_type) * tf.expand_dims(self.rating_mult, 1)
            ratings = tf.cast(self.ratings, dtype=self.float_type, name='ratings')
            # min_rating = tf.stop_gradient(tf.math.reduce_min(ratings[1:]))
            # max_rating = tf.stop_gradient(tf.math.reduce_max(ratings))
            min_rating = tf.stop_gradient(tf.math.reduce_min(ratings[1:]))
            max_rating = tf.stop_gradient(tf.math.reduce_max(ratings))
            ratings = self.rating_mult * (ratings - min_rating) / (max_rating - min_rating)
            oracle_weights = tf.cast(self.oracle_weights, dtype=self.float_type)
            # oracle_weights = tf.cast(tf.math.softplus(self.oracle_weights), dtype=self.float_type)
            oracle_weights = oracle_weights - tf.stop_gradient(tf.reduce_min(oracle_weights, keepdims=True)) + 1e-2
            oracle_weights = tf.math.divide_no_nan(oracle_weights, tf.reduce_sum(oracle_weights, axis=2, keepdims=True))

        tf.summary.histogram('weights/oracle_weights', oracle_weights)
        tf.summary.histogram('weights/card_ratings', ratings)
        for i, name in enumerate(('ratings', 'pick_synergy', 'internal_synergy', 'colors', 'openness')):
            log_timeseries(f'weights/{name}_weights', oracle_weights[:,:14,i], start_index=1)

        oracle_weight_values = tf.gather_nd(oracle_weights, coords) # (-1, 4, 5)
        picked_embeds = tf.gather(embeddings, picked_card_indices, name='picked_embeds')
        in_pack_embeds = tf.gather(embeddings, in_pack_card_indices, name='in_pack_embeds')
        picked_ratings = tf.gather(ratings, picked_card_indices)
        in_pack_ratings = tf.gather(ratings, in_pack_card_indices)
        seen_ratings = tf.gather(ratings, seen_indices)

        with tf.xla.experimental.jit_scope(compile_ops=not self.debugging, separate_compiled_gradients=self.separate_gradients):
            picked_squared_norm = tf.reduce_sum(picked_embeds * picked_embeds, 2, keepdims=True)
            norm_picked_embeds = tf.where(picked_squared_norm > 0, tf.sqrt(picked_squared_norm), tf.zeros_like(picked_squared_norm), name='norm_picked_embeds')
            scaled_picked_embeds = tf.math.divide_no_nan(picked_embeds, norm_picked_embeds)
            in_pack_squared_norm = tf.reduce_sum(in_pack_embeds * in_pack_embeds, 2, keepdims=True)
            norm_in_pack_embeds = tf.sqrt(in_pack_squared_norm)
            norm_in_pack_embeds = tf.where(in_pack_squared_norm > 0, tf.sqrt(in_pack_squared_norm), tf.zeros_like(in_pack_squared_norm), name='norm_in_pack_embeds')

            pick_synergies = tf.reduce_sum(tf.expand_dims(tf.math.divide_no_nan(in_pack_embeds, norm_in_pack_embeds), 2)
                                                      * tf.expand_dims(scaled_picked_embeds, 1), -1)
            pick_synergies = self.pick_synergy_dropout(pick_synergies * (1 - tf.eye(MAX_IN_PACK, num_columns=MAX_PICKED, dtype=self.float_type)), training=training)
            internal_synergies = tf.reduce_sum(tf.expand_dims(scaled_picked_embeds, 2)
                                               * tf.expand_dims(scaled_picked_embeds, 1), -1)
            internal_synergies = self.internal_synergy_dropout(internal_synergies * (1 - tf.eye(MAX_PICKED, dtype=self.float_type)), training=training)
            total_probs = tf.reduce_sum(prob_pickeds, 2, name='total_probs')

        synergies = tf.reshape(tf.concat([internal_synergies, pick_synergies], 1), (-1,))
        synergies = tf.boolean_mask(synergies, tf.math.abs(synergies) > 0)
        tf.summary.histogram('values/synergies', synergies)

        with tf.xla.experimental.jit_scope(compile_ops=not self.debugging, separate_compiled_gradients=self.separate_gradients):
            rating_scores = prob_in_packs * tf.expand_dims(in_pack_ratings, 1)
            pick_synergy_scores = tf.math.divide_no_nan(
                prob_in_packs * tf.reduce_sum(tf.expand_dims(prob_pickeds, 2)
                                              * tf.expand_dims(pick_synergies, 1), 3),
                tf.expand_dims(picked_counts, -1)        # Same as for internal_synergy_oracles
            )
            internal_synergy_scores = tf.expand_dims(tf.math.divide_no_nan(tf.reduce_sum(
                    (tf.expand_dims(prob_pickeds, 3) * tf.expand_dims(prob_pickeds, 2))
                    * tf.expand_dims(internal_synergies, 1),
                    [2, 3], name='internal_synergy_reduction'
                ),
                total_probs * (picked_counts - 1)
            ), 2)
            colors_scores = tf.expand_dims(
                tf.math.divide_no_nan(tf.reduce_sum(tf.expand_dims(picked_ratings, 1) * prob_pickeds, 2), picked_counts), 2
            )
            openness_scores = tf.expand_dims(
                tf.math.divide_no_nan(tf.reduce_sum(tf.expand_dims(seen_ratings, 1) * prob_seens, 2), seen_counts), 2
            )

            oracles_weights = tf.reshape(tf.reduce_sum(
                oracle_weight_values * tf.expand_dims(coord_weights, 2), 1
            ), (-1, 1, 1, 5), name=f'oracles_weights')
            oracle_scores = tf.stack(tuple(scores + tf.zeros_like(rating_scores)
                                           for scores in (rating_scores, pick_synergy_scores,
                                                          internal_synergy_scores, colors_scores,
                                                          openness_scores)),
                                     axis=3, name='oracle_scores')
            # We can multiply by 100 here instead of scaling the weights, ratings, and synergies earlier.
            # We do want to scale them into the [0, 10] range though since that gives much nicer looking scores.
            scores = tf.reduce_sum(oracle_scores * oracles_weights, -1, name='weighted_scores')
            max_scores = tf.reduce_logsumexp(tf.cast(self.temperature * scores, dtype=tf.float32), 1, name='max_scores')
            # tf.summary.histogram('outputs/max_scores', max_scores / self.temperature)
            # tf.summary.histogram('outputs/true_max_scores', tf.reduce_max(max_scores, axis=1) / self.temperature)
            # tf.summary.histogram('outputs/correct_max_scores', max_scores[:,0] / self.temperature)
            choice_probs = tf.nn.softmax(max_scores, 1, name='choice_probs')
            tf.summary.histogram('outputs/prob_correct', choice_probs[:,0])
            tf.summary.histogram('outputs/probs_incorrect', choice_probs[:,1:])
            # max_of_other = tf.reduce_logsumexp(self.temperature * max_scores[:,1:], 1) / self.temperature
            return choice_probs
            # return tf.cast(choice_probs, dtype=self.float_type)
            # return (
            #     choice_probs,
            #     prob_correct,
            #     # prob_correct
            #     # (max_scores[:,0] - max_of_other) / max_scores[:,0]
            # )
