import numpy as np
import tensorflow as tf

EMBEDS_DIM = 64


class DraftBot(tf.keras.models.Model):
    def __init__(self, card_ratings, card_embeddings, temperature, num_land_combs):
        super(DraftBot, self).__init__()
        self.num_land_combs = num_land_combs
        self.rating_weights = tf.Variable(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ],
            constraint=lambda x: tf.clip_by_value(x, 0, 10),
            dtype=tf.float32,
            name='rating_weights',
        )
        self.colors_weights = tf.Variable(
            [
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            ],
            constraint=lambda x: tf.clip_by_value(x, 0, 10),
            dtype=tf.float32,
            name='colors_weights',
        )
        self.internal_synergy_weights = tf.Variable(
            [
                [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            ],
            constraint=lambda x: tf.clip_by_value(x, 0, 10),
            dtype=tf.float32,
            name='internal_synergy_weights',
        )
        self.pick_synergy_weights = tf.Variable(
            [
                [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            ],
            constraint=lambda x: tf.clip_by_value(x, 0, 10),
            dtype=tf.float32,
            name='pick_synergy_weights',
        )
        self.openness_weights = tf.Variable(
            [
                [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            ],
            constraint=lambda x: tf.clip_by_value(x, 0, 10),
            dtype=tf.float32,
            name='openness_weights',
        )
        rating_max = tf.constant([0] + [10 for _ in card_ratings[1:]], dtype=tf.float32)
        rating_min = tf.constant([0] + [-10 for _ in card_ratings[1:]], dtype=tf.float32)
        self.ratings = tf.Variable(
            card_ratings,
            constraint=lambda x: tf.clip_by_value(x, rating_min, rating_max),
            dtype=tf.float32,
            name='card_ratings',
        )
        # self.card_synergies = tf.Variable(tf.fill((len(card_ratings) * (len(card_ratings) + 1) // 2,), 0.0),
        #                                   dtype=tf.float32)
        self.card_embeddings = tf.Variable(card_embeddings, dtype=tf.float32, name='card_embeddings')
        # self.card_embeddings = tf.Variable(np.random.uniform(-1, 1, (len(card_ratings), EMBEDS_DIM)), dtype=tf.float32)
        self.temperature = tf.constant(temperature, dtype=tf.float32)
        self.synergy = tf.keras.layers.Dot(3, normalize=True)

    def gather_weights(self, weights, coords, coord_weights):
        return tf.reshape(tf.reduce_sum(tf.gather_nd(weights, coords) * coord_weights, 1), (-1, 1))

    def call(self, inputs, training=None, mask=None):
        in_pack_card_indices, seen_indices, seen_counts,\
            picked_card_indices, picked_counts, coords, coord_weights,\
            prob_seen_matrices,\
            prob_picked_matrices, prob_in_pack_matrices = inputs

        # Have to do this because you can't have a constraint on an embedding.
        ratings = self.ratings * 1
        in_pack_ratings = tf.reshape(tf.gather(ratings, in_pack_card_indices, name='in_pack_ratings'), (-1, 16))
        seen_ratings = tf.reshape(tf.gather(ratings, seen_indices, name='seen_ratings'), (-1, 360))
        picked_ratings = tf.reshape(tf.gather(ratings, picked_card_indices, name='picked_ratings'), (-1, 48))

        picked_embeds = tf.gather(self.card_embeddings, picked_card_indices)
        picked_embed_a = tf.reshape(picked_embeds, (-1, 48, 1, EMBEDS_DIM))
        picked_embed_b = tf.reshape(picked_embeds, (-1, 1, 48, EMBEDS_DIM))
        internal_synergy_matrices = tf.reshape(10 * self.synergy([picked_embed_a, picked_embed_b]), (-1, 48, 48))
        in_pack_embeds = tf.reshape(tf.gather(self.card_embeddings, in_pack_card_indices), (-1, 16, 1, EMBEDS_DIM))
        picked_synergy_matrices = tf.reshape(10 * self.synergy([in_pack_embeds, picked_embed_b]), (-1, 16, 48))

        # picked_pairs_a = tf.reshape(picked_card_indices, (-1, 48, 1))
        # picked_pairs_b = tf.reshape(picked_card_indices, (-1, 1, 48))
        # internal_pairs_idx1 = picked_pairs_a * (picked_pairs_a + 1) // 2 + picked_pairs_b
        # internal_pairs_idx2 = picked_pairs_b * (picked_pairs_b + 1) // 2 + picked_pairs_a
        # internal_pairs = tf.maximum(internal_pairs_idx1, internal_pairs_idx2)
        # internal_synergy_matrices = tf.gather(self.card_synergies, internal_pairs)
        # in_pack_pairs = tf.reshape(in_pack_card_indices, (-1, 16, 1))
        # picked_pairs_idx1 = in_pack_pairs * (in_pack_pairs + 1) // 2 + picked_pairs_b
        # picked_pairs_idx2 = picked_pairs_b * (picked_pairs_b + 1) // 2 + in_pack_pairs
        # picked_pairs = tf.maximum(picked_pairs_idx1, picked_pairs_idx2)
        # picked_synergy_matrices = tf.gather(self.card_synergies, picked_pairs)

        rating_weights = self.gather_weights(self.rating_weights, coords, coord_weights)
        pick_synergy_weights = self.gather_weights(self.pick_synergy_weights, coords, coord_weights)
        internal_synergy_weights = self.gather_weights(self.internal_synergy_weights, coords, coord_weights)
        openness_weights = self.gather_weights(self.openness_weights, coords, coord_weights)
        colors_weights = self.gather_weights(self.colors_weights, coords, coord_weights)

        prob_seens = tf.reshape(tf.cast(prob_seen_matrices, dtype=tf.float32) / 255, (-1, self.num_land_combs, 360))
        prob_pickeds = tf.reshape(tf.cast(prob_picked_matrices, dtype=tf.float32) / 255, (-1, self.num_land_combs, 48))
        prob_in_packs = tf.reshape(tf.cast(prob_in_pack_matrices, dtype=tf.float32) / 255, (-1, self.num_land_combs, 16))

        total_probs = tf.reshape((tf.reduce_sum(prob_pickeds, 2)), (-1, self.num_land_combs)) + 1e-06
        picked_counts = tf.reshape(picked_counts, (-1, 1))
        internal_synergy_oracles = tf.reduce_sum(
            (tf.expand_dims(prob_pickeds, 3) @ tf.expand_dims(prob_pickeds, 2))
            * tf.expand_dims(internal_synergy_matrices, 1),
            [2, 3]
        ) / total_probs / (picked_counts - 1) # We guarantee picked_counts is greater than 1

        # Same as for internal_synergy_oracles
        colors_oracles = tf.reduce_sum(tf.expand_dims(picked_ratings, 1) * prob_pickeds, 2) / picked_counts

        openness_oracles = tf.reduce_sum(tf.expand_dims(seen_ratings, 1) * prob_seens, 2)\
            / tf.reshape(seen_counts, (-1, 1)) # We guarantee seen_counts is greater than 0

        global_scores = (
            internal_synergy_oracles * internal_synergy_weights
            + colors_oracles * colors_weights
            + openness_oracles * openness_weights
        ) # [batch_size, self.num_land_combs]

        picked_synergy_matrices = tf.reshape(picked_synergy_matrices, (-1, 16, 48))
        picked_counts = tf.reshape(picked_counts, (-1, 1, 1))
        pick_synergy_oracles =\
            tf.reduce_sum(tf.expand_dims(prob_pickeds, 2)
                          * tf.expand_dims(picked_synergy_matrices, 1), 3)\
            / picked_counts # Same as for internal_synergy_oracles
        local_scores = prob_in_packs * (tf.expand_dims(in_pack_ratings, 1) * tf.expand_dims(rating_weights, 2)
                                        + pick_synergy_oracles * tf.expand_dims(pick_synergy_weights, 2))

        scores = local_scores + tf.expand_dims(global_scores, 2)
        # return tf.reduce_max(scores * self.temperature, 1)
        exp_scores = tf.reduce_sum(tf.exp(tf.cast(self.temperature * scores, dtype=tf.float64)), 1)
        return exp_scores / tf.expand_dims(tf.reduce_sum(exp_scores, 1), 1)
