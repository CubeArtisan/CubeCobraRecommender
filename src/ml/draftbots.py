import tensorflow as tf

PACK_SIZE = 15
PACKS = 3


class DraftBot(tf.keras.models.Model):
    def __init__(self, num_cards, card_embeddings, prob_to_play, land_requirements, card_colors,
                 is_land, is_fetch, has_basic_land_types):
        super(DraftBot, self).__init__()
        self.num_cards = num_cards
        self.ratings = tf.keras.layers.Embedding(num_cards + 1, 1, mask_zero=True)
        self.rating_weights = tf.Variable([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ])
        self.colors_weights = tf.Variable([
            [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
            [40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
            [60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60],
        ], dtype=tf.float32)
        self.fixing_weights = tf.Variable([
            [0.1, 0.3, 0.6, 0.8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ], dtype=tf.float32)
        self.internal_synergy_weights = tf.Variable([
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        ], dtype=tf.float32)
        self.external_synergy_weights = tf.Variable([
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        ], dtype=tf.float32)
        self.openness_weights = tf.Variable([
            [4, 12, 12.3, 12.6, 13, 13.4, 13.7, 14, 15, 14.6, 14.2, 13.8, 13.4, 13, 12.6],
            [13, 12.6, 12.2, 11.8, 11.4, 11, 10.6, 10.2, 9.8, 9.4, 9, 8.6, 8.2, 7.8, 7],
            [8, 7.5, 7, 6.5, 6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1],
        ], dtype=tf.float32)
        self.similarity_clip = tf.Variable(0.7, dtype=tf.float32,
                                           constraint=lambda x: tf.minimum(tf.maximum(x, 0), 0.99))
        self.prob_to_include = tf.Variable(0.67, dtype=tf.float32,
                                           constraint=lambda x: tf.minimum(tf.maximum(x, 0), 1))
        self.card_embeddings = tf.constant(card_embeddings)
        self.prob_to_play = tf.constant(prob_to_play)
        self.land_requirements = tf.constant(land_requirements)
        self.card_colors = tf.constant(card_colors)
        self.is_land = tf.constant(is_land, dtype=tf.bool)
        self.is_fetch = tf.constant(is_fetch, dtype=tf.bool)
        self.has_basic_land_types = tf.constant(has_basic_land_types, dtype=tf.bool)

    def call(self, inputs, training=None, mask=None):
        card, picked, seen, pick_num, pack_num, packs, pack_size = inputs

        def get_weights(weights):
            xCoordinate = tf.divide(pack_num, packs)
            yCoordinate = tf.divide(pick_num, pack_size)
            xIndex = PACKS * xCoordinate
            yIndex = PACK_SIZE * yCoordinate
            floorXIndex = tf.floor(xIndex)
            floorYIndex = tf.floor(yIndex)
            ceilXIndex = tf.math.ceil(xIndex)
            ceilYIndex = tf.math.ceil(yIndex)
            xIndexModOne = tf.subtract(xIndex, floorXIndex)
            yIndexModOne = tf.subtract(yIndex, floorYIndex)
            InvXIndexModOne = 1 - xIndexModOne
            InvYIndexModOne = 1 - yIndexModOne
            XY = tf.multiply(xIndexModOne, yIndexModOne)
            Xy = tf.multiply(xIndexModOne, InvYIndexModOne)
            xY = tf.multiply(InvXIndexModOne, yIndexModOne)
            xy = tf.multiply(InvXIndexModOne, InvYIndexModOne)
            XYWeight = tf.gather_nd(weights, (ceilXIndex, ceilYIndex))
            XyWeight = tf.gather_nd(weights, (ceilXIndex, floorYIndex))
            xYWeight = tf.gather_nd(weights, (floorXIndex, ceilYIndex))
            xyWeight = tf.gather_nd(weights, (floorXIndex, floorYIndex))
            XY = tf.multiply(XY, XYWeight)
            Xy = tf.multiply(Xy, XyWeight)
            xY = tf.multiply(xY, xYWeight)
            xy = tf.multiply(xy, xyWeight)
            return tf.add_n((XY, Xy, xY, xy))

        # This is a wildly inaccurate approximation, but works for a first draft
        def get_casting_probability(lands, card_index):
            lands_for_card = tf.gather(self.land_requirements, card_index)
            total_devotion = tf.reduce_sum(lands_for_card)
            probability = tf.constant(1)
            total_lands = tf.constant(0)
            for i in range(5):
                on_color_lands = tf.gather(lands, i)
                devotion = tf.gather(lands_for_card, i)
                probability = probability * tf.gather_nd(self.prob_to_play, (devotion,
                                                                             on_color_lands))
                total_lands = total_lands + tf.cond(devotion > 0, lambda: on_color_lands,
                                                    lambda: tf.constant(0))
            return probability * tf.gather_nd(self.prob_to_play, (total_devotion, total_lands))

        def get_rating(lands, card_index):
            rating = tf.reshape(self.ratings(card_index), ())
            return get_casting_probability(lands, tf.gather(self.land_requirements, card_index))\
                * rating

        def transform_similarity(similarity):
            similarity_multiplier = 1 / (1 - self.similarity_clip)
            scaled = similarity_multiplier\
                * tf.minimum(tf.maximum(0, similarity - self.similarity_clip),
                             1 - self.similarity_clip)
            transformed = -tf.math.log(1 - scaled) * 5
            return tf.map_fn(lambda value: tf.cond(tf.math.is_inf(value), lambda: tf.constant(10),
                                                   lambda: value), transformed)

        def rating_oracle(lands, batch_index1, batch_index2):
            card_index = tf.gather_nd(card, (batch_index1, batch_index2))
            return get_rating(lands, card_index)

        def pick_synergy_oracle(lands, batch_index1, batch_index2):
            card_index = tf.gather_nd(card, (batch_index1, batch_index2))
            card_embedding = tf.gather(self.card_embeddings, card_index)
            card_embedding = tf.tile(tf.expand_dims(card_embedding, axis=0), [self.num_cards, 1])
            similarities = tf.reshape(tf.keras.layers.dot([card_embedding, self.card_embeddings],
                                                          normalize=True), (-1,))
            synergies = transform_similarity(similarities)
            probabilities = tf.map_fn(lambda ci: tf.cond(tf.equal(ci, card_index), lambda: 0,
                                                         lambda: get_casting_probability(lands, ci)),
                                      tf.range(self.num_cards))
            weighted_synergies = tf.multiply(synergies, probabilities)
            our_picked = tf.gather(picked, (batch_index1, batch_index2))
            return tf.keras.layers.dot([weighted_synergies, our_picked])

        def fixing_oracle(lands, batch_index1, batch_index2):
            card_index = tf.gather_nd(card, (batch_index1, batch_index2))

            def calculate_fixing():
                combination = tf.cast(tf.greater(lands, 3), "float32")
                colors = tf.gather(self.card_colors, card_index)
                overlap = 2 * tf.keras.layers.dot(combination, colors)
                overlap = tf.cond(tf.gather(self.is_fetch, card_index), lambda: overlap,
                                  lambda: tf.cond(tf.gather(self.has_basic_land_types, card_index),
                                                  lambda: 0.75 * overlap,
                                                  lambda: 0.5 * overlap))
            return tf.cond(tf.gather(self.is_land, card_index), calculate_fixing,
                           lambda: tf.constant(0))

        def internal_synergy_oracle(lands, batch_index1, batch_index2):
            def get_card_internal_synergy(card_index):
                card_probability = get_casting_probability(lands, card_index)

                def inner():
                    def synergy_with_card(other_index):
                        card_embedding = tf.gather(self.card_embeddings, card_index)
                        other_probability = get_casting_probability(lands, other_index)
                        other_count = tf.gather_nd(picked, (batch_index1, batch_index2, other_index))
                        other_count = tf.cond(tf.equal(other_index, card_index),
                                              lambda: other_count - 1,
                                              lambda: other_count)

                        def inner2():
                            other_embedding = tf.gather(self.other_embedding, other_index)
                            similarity = tf.keras.layers.dot([card_embedding, other_embedding],
                                                             nomralize=True)
                            synergy = transform_similarity(similarity)
                            return other_count * synergy * other_probability * card_probability
                        return tf.cond(tf.logical_and(tf.greater(other_count, 0),
                                                      tf.greater_equal(other_probability,
                                                                       self.prob_to_include)),
                                       inner2, lambda: 0)
                    return tf.reduce_sum(tf.map_fn(synergy_with_card, tf.range(self.num_cards)))
                seen_count = tf.gather_nd(picked, (batch_index1, batch_index2, card_index))
                tf.cond(tf.logical_and(tf.greater(seen_count, 0),
                                       tf.greater_equal(card_probability, self.prob_to_include)),
                                       inner, lambda: 0)
            return tf.reduce_sum(tf.map_fn(get_card_internal_synergy, tf.range(self.num_cards)))

        def openness_oracle(lands, batch_index1, batch_index2):
            def get_card_openness(card_index):
                card_probability = get_casting_probability(lands, card_index)
                card_count = tf.gather_nd(seen, (batch_index1, batch_index2, card_index))
                return tf.cond(tf.logical_and(tf.greater(card_count, 0),
                                              tf.greater_equal(card_probability,
                                                               self.prob_to_include)),
                               lambda: card_count * get_rating(lands, card_index),
                               lambda: 0)
            return tf.reduce_sum(tf.map_fn(get_card_openness, tf.range(self.num_cards)))

        def color_oracle(lands, batch_index1, batch_index2):
            def get_card_contribution(card_index):
                card_probability = get_casting_probability(lands, card_index)
                card_count = tf.gather_nd(picked, (batch_index1, batch_index2, card_index))
                return tf.cond(tf.logical_and(tf.greater(card_count, 0),
                                              tf.greater_equal(card_probability,
                                                               self.prob_to_include)),
                               lambda: card_count * get_rating(lands, card_index),
                               lambda: 0)
            return tf.reduce_sum(tf.map_fn(get_card_contribution, tf.range(self.num_cards)))

        oracles = [rating_oracle, pick_synergy_oracle, fixing_oracle, internal_synergy_oracle,
                   openness_oracle, color_oracle]
        weights = [self.rating_weights, self.pick_synergy_weights, self.fixing_weights,
                   self.internal_synergy_weights, self.openness_weights, self.colors_weights]

        def get_score(lands, batch_index1, batch_index2):
            return tf.reduce_sum([oracle(lands, batch_index1, batch_index2) * get_weights(weight)
                                  for oracle, weight in zip(oracles, weights)])

        def do_climb(batch_index1, batch_index2):
            def climb_cond(prev_value, current_value, _):
                return tf.greater(current_value, prev_value)

            def try_climb(_, current_value, lands):
                final_value = current_value
                final_lands = lands
                for remove_index in range(5):
                    for add_index in range(5):
                        if remove_index == add_index:
                            continue
                        new_lands = tf.map_fn(lambda i: tf.cond(tf.equal(remove_index, i),
                                                                lambda: tf.gather(lands, i) - 1,
                                                                lambda: tf.cond(tf.equal(add_index, i),
                                                                                lambda: tf.gather(lands, i) + 1,
                                                                                lambda: tf.gather(lands, i))))
                        new_value = tf.cond(tf.greater(tf.gather(lands, remove_index), 0),
                                            lambda: get_score(new_lands, batch_index1, batch_index2),
                                            lambda: 0)
                        final_lands = tf.cond(tf.greater(new_value, final_value), lambda: new_lands,
                                              lambda: final_lands)
                        final_value = tf.maximum(new_value, final_value)
                return current_value, final_value, final_lands

            _, score, _ = tf.while_loop(climb_cond, try_climb, [tf.constant(0), tf.constant(1),
                                                                    tf.constant([4, 4, 3, 3, 3])])
            return score

        batch_size = tf.shape(card)[0]
        pack_size = tf.shape(card)[1]
        return tf.map_fn(lambda b1: tf.nn.softmax(tf.map_fn(lambda b2: do_climb(b1, b2),
                                                            tf.range(pack_size))),
                         tf.range(batch_size))