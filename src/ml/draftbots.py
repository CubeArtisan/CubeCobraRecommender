import numpy as np
import tensorflow as tf

from ..non_ml.parse_picks import MAX_IN_PACK, MAX_PICKED, MAX_SEEN, NUM_LAND_COMBS

EMBEDS_DIM = 64

def calculate_synergies(name):
    @tf.function(experimental_compile=True)
    def inner(embeds_a, embeds_b, indices_a, indices_b):
        print(embeds_a.shape)
        norm_a = tf.sqrt(tf.reduce_sum(embeds_a * embeds_a, 2) + 1e-05)
        norm_b = tf.sqrt(tf.reduce_sum(embeds_b * embeds_b, 2) + 1e-05)
        expand_a = tf.expand_dims(embeds_a / tf.expand_dims(norm_a, 2), 2)
        print(expand_a.shape)
        expand_b = tf.expand_dims(embeds_b / tf.expand_dims(norm_b, 2), 1)
        return tf.reduce_sum(expand_a * expand_b, -1)
        # return tf.where(
        #     tf.math.logical_and(tf.expand_dims(indices_a > 0, -1), tf.expand_dims(indices_b > 0, -2)),
        #     similarities / norms,
        #     tf.zeros_like(similarities),
        #     name=f'{name}_synergies',
        # )
    return inner
calc_internal_synergy = calculate_synergies('internal')
calc_picked_synergy = calculate_synergies('picked')

def gather_weights(name):
    @tf.function(experimental_compile=True)
    def inner(coord_weights, weights):
        return tf.reshape(tf.reduce_sum(tf.cast(tf.math.sigmoid(weights), dtype=tf.float16)
                                        * coord_weights, 1),
                          (-1, 1), name=f'{name}_weights')
    return inner
gather_rating_weights = gather_weights('rating')
gather_pick_synergy_weights = gather_weights('pick_synergy')
gather_internal_synergy_weights = gather_weights('internal_synergy')
gather_colors_weights = gather_weights('color')
gather_openness_weights = gather_weights('openness')

def gather_ratings(name):
    @tf.function(experimental_compile=True)
    def inner(ratings):
        return tf.cast(tf.math.tanh(ratings), dtype=tf.float16, name=f'{name}_ratings')
    return inner
gather_pack_ratings = gather_ratings('in_pack')
gather_picked_ratings = gather_ratings('picked')
gather_seen_ratings = gather_ratings('seen')

# @tf.function(experimental_compile=True)
def calc_internal_synergy_score(prob_pickeds, picked_embeds, picked_card_indices, picked_counts, internal_synergy_weights):
    internal_synergy_matrices = calc_internal_synergy(picked_embeds, picked_embeds, picked_card_indices,
                                                           picked_card_indices)
    total_probs = tf.reduce_sum(prob_pickeds, 2, name='total_probs') + tf.constant(1e-05, dtype=tf.float16)
    return internal_synergy_weights * tf.reduce_sum(
        (tf.expand_dims(prob_pickeds, 3) * tf.expand_dims(prob_pickeds, 2))
        * tf.expand_dims(internal_synergy_matrices, 1),
        [2, 3], name='internal_synergy_reduction'
    ) / (total_probs * (picked_counts - 1)) # We guarantee picked_counts is at least 2

def calc_weighted_sum_ratings(name):
    @tf.function(experimental_compile=True)
    def inner(ratings, probs, counts, weights):
        return weights * tf.reduce_sum(tf.expand_dims(ratings, 1) * probs, 2, name=f'{name}_reduction') / counts
    return inner
calc_colors_score = calc_weighted_sum_ratings('colors')
calc_openness_score = calc_weighted_sum_ratings('openness')

@tf.function(experimental_compile=True)
def calc_global_score(internal_synergy_scores, colors_scores, openness_scores):
    return internal_synergy_scores + colors_scores + openness_scores

# @tf.function(experimental_compile=True)
def calc_pick_synergy_score(prob_pickeds, picked_counts, picked_embeds, picked_indices,
                            pack_indices, pack_embeds, pick_synergy_weights):
    picked_synergy_matrices = calc_picked_synergy(pack_embeds, picked_embeds, pack_indices, picked_indices)
    return tf.reduce_sum(tf.expand_dims(prob_pickeds, 2) * tf.expand_dims(picked_synergy_matrices, 1), 3)\
        / tf.expand_dims(picked_counts * pick_synergy_weights, -1) # Same as for internal_synergy_oracles

@tf.function(experimental_compile=True)
def calc_local_scores(pick_synergy_scores, in_pack_ratings, rating_weights, prob_in_packs):
    return prob_in_packs * (tf.expand_dims(in_pack_ratings * rating_weights, 1, name='rating_score')
                            + pick_synergy_scores)

@tf.function(experimental_compile=True)
def calc_choice_probs(local_scores, global_scores, temperature):
    scores = tf.cast(local_scores + tf.expand_dims(global_scores, 2), dtype=tf.float64, name='scores')
    exp_scores = 1 + tf.reduce_sum(tf.exp(temperature * scores, name='exp_land_scores'), 1, name='exp_scores')
    return tf.cast(exp_scores / tf.expand_dims(tf.reduce_sum(exp_scores, 1), 1, name='total_exp_scores'),
                   dtype=tf.float32, name='final_result')

class DraftBot(tf.keras.models.Model):
    def __init__(self, card_ratings, card_embeddings, temperature):
        super(DraftBot, self).__init__()
        self.rating_weights = tf.Variable(
            np.random.uniform(-1, 1, (3, 15)),
            dtype=tf.float32, name='rating_logit_weights',
        )
        self.pick_synergy_weights = tf.Variable(
            np.random.uniform(-1, 1, (3, 15)),
            dtype=tf.float32, name='pick_synergy_logit_weights',
        )
        self.internal_synergy_weights = tf.Variable(
            np.random.uniform(-1, 1, (3, 15)),
            dtype=tf.float32, name='internal_synergy_logit_weights',
        )
        self.colors_weights = tf.Variable(
            np.random.uniform(-1, 1, (3, 15)),
            dtype=tf.float32, name='colors_logit_weights',
        )
        self.openness_weights = tf.Variable(
            np.random.uniform(-1, 1, (3, 15)),
            dtype=tf.float32, name='openness_logit_weights',
        )
        self.rating_mult = tf.constant([0] + [1 for _ in card_ratings[1:]], dtype=tf.float32)
        self.ratings = tf.Variable(
            tf.math.atanh(tf.minimum(card_ratings, 8) / 10),
            dtype=tf.float32, name='card_ratings',
        )
        self.card_embeddings = tf.Variable(card_embeddings, dtype=tf.float32, name='card_embeddings')
        # We can multiply by 100 here instead of scaling the weights, ratings, and synergies later.
        self.temperature = tf.constant(temperature * 100, dtype=tf.float64)

    def call(self, inputs):
        in_pack_card_indices, seen_indices, seen_counts,\
            picked_card_indices, picked_counts, coords, coord_weights,\
            prob_seen_matrices,\
            prob_picked_matrices, prob_in_pack_matrices = inputs
        in_pack_card_indices = tf.reshape(tf.cast(in_pack_card_indices, dtype=tf.int32), (-1, MAX_IN_PACK), name='in_packs')
        seen_indices = tf.reshape(tf.cast(seen_indices, dtype=tf.int32), (-1, MAX_SEEN), name='seens')
        seen_counts = tf.reshape(tf.cast(seen_counts, dtype=tf.float16), (-1, 1), name='seen_counts')
        picked_card_indices = tf.reshape(tf.cast(picked_card_indices, dtype=tf.int32), (-1, MAX_PICKED), name='pickeds')
        picked_counts = tf.reshape(tf.cast(picked_counts, dtype=tf.float16), (-1, 1), name='picked_counts')
        coords = tf.reshape(tf.cast(coords, dtype=tf.int32), (-1, 4, 2), name='coords')
        coord_weights = tf.reshape(tf.cast(coord_weights, dtype=tf.float16), (-1, 4), name='coord_weights')
        prob_seens = tf.reshape(tf.cast(prob_seen_matrices, dtype=tf.float16) / 255,
                                (-1, NUM_LAND_COMBS, MAX_SEEN), name='prob_seens')
        prob_pickeds = tf.reshape(tf.cast(prob_picked_matrices, dtype=tf.float16) / 255,
                                  (-1, NUM_LAND_COMBS, MAX_PICKED), name='prob_pickeds')
        prob_in_packs = tf.reshape(tf.cast(prob_in_pack_matrices, dtype=tf.float16) / 255,
                                   (-1, NUM_LAND_COMBS, MAX_IN_PACK), name='prob_in_packs')
        ratings = self.ratings * self.rating_mult
        embeddings = self.card_embeddings * tf.expand_dims(self.rating_mult, 1)
        picked_embeds = tf.cast(tf.gather(embeddings, picked_card_indices), name='picked_embeds', dtype=tf.float16)
        in_pack_embeds = tf.cast(tf.gather(embeddings, in_pack_card_indices), name='in_pack_embeds', dtype=tf.float16)
        in_pack_ratings = gather_pack_ratings(tf.gather(ratings, in_pack_card_indices))
        picked_ratings = gather_picked_ratings(tf.gather(ratings, picked_card_indices))
        seen_ratings = gather_seen_ratings(tf.gather(ratings, seen_indices))
        rating_weights = gather_rating_weights(coord_weights, tf.gather_nd(self.rating_weights, coords))
        pick_synergy_weights = gather_pick_synergy_weights(coord_weights, tf.gather_nd(self.pick_synergy_weights, coords))
        internal_synergy_weights = gather_internal_synergy_weights(coord_weights, tf.gather_nd(self.internal_synergy_weights, coords))
        colors_weights = gather_colors_weights(coord_weights, tf.gather_nd(self.colors_weights, coords))
        openness_weights = gather_openness_weights(coord_weights, tf.gather_nd(self.openness_weights, coords))

        internal_synergy_score = calc_internal_synergy_score(prob_pickeds, picked_embeds, picked_card_indices,
                                                             picked_counts, internal_synergy_weights)
        colors_score = calc_colors_score(picked_ratings, prob_pickeds, picked_counts, colors_weights)
        openness_score = calc_openness_score(seen_ratings, prob_seens, seen_counts, openness_weights)
        global_scores = calc_global_score(internal_synergy_score, colors_score, openness_score)
        pick_synergy_score = calc_pick_synergy_score(prob_pickeds, picked_counts,
                                                     picked_embeds, picked_card_indices, in_pack_card_indices, in_pack_embeds,
                                                     pick_synergy_weights)
        local_scores = calc_local_scores(pick_synergy_score, in_pack_ratings, rating_weights, prob_in_packs)
        return calc_choice_probs(local_scores, global_scores, self.temperature)
