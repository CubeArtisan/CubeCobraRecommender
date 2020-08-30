import tensorflow as tf

PACK_SIZE = 15
PACKS = 3
MAX_PACK_SIZE = 32
MAX_SEEN = 512
MAX_PICKED = 128


class DraftBot(tf.keras.models.Model):
    def __init__(self, num_cards, card_embeddings, prob_to_play, land_requirements, card_colors,
                 is_land, is_fetch, has_basic_land_types, temperature):
        super(DraftBot, self).__init__()
        self.num_cards = num_cards
        self.rating_weights = tf.Variable([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ], dtype=tf.float32)
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
        self.pick_synergy_weights = tf.Variable([
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
        self.ratings = tf.keras.layers.Embedding(num_cards + 1, 1, mask_zero=True)
        self.card_embeddings = tf.constant(card_embeddings)
        self.prob_to_play = tf.constant(prob_to_play)
        self.land_requirements = tf.constant(land_requirements)
        self.card_colors = tf.constant(card_colors)
        self.is_land = tf.constant(is_land, dtype=tf.bool)
        self.is_fetch = tf.constant(is_fetch, dtype=tf.bool)
        self.has_basic_land_types = tf.constant(has_basic_land_types, dtype=tf.bool)
        self.temperature = tf.constant(temperature, dtype=tf.float32)

    def call(self, inputs, training=None, mask=None):
        cards, picked, seen, pick_num, pack_num, packs, pack_size = inputs
        cards = tf.cast(cards, 'int32')
        cards_rating = tf.reshape(self.ratings(cards), tf.shape(cards))
        cards_embeddings = tf.gather(self.card_embeddings, cards)
        cards_land_requirements = tf.gather(self.land_requirements, cards)
        cards_colors = tf.gather(self.card_colors, cards)
        cards_is_land = tf.gather(self.is_land, cards)
        cards_is_fetch = tf.gather(self.is_fetch, cards)
        cards_has_basic_land_types = tf.gather(self.has_basic_land_types, cards)
        picked = tf.cast(picked, 'int32')
        picked_rating = tf.reshape(self.ratings(picked), tf.shape(picked))
        picked_embeddings = tf.gather(self.card_embeddings, picked)
        picked_land_requirements = tf.gather(self.land_requirements, picked)
        seen = tf.cast(seen, 'int32')
        seen_rating = tf.reshape(self.ratings(seen), tf.shape(seen))
        seen_land_requirements = tf.gather(self.land_requirements, seen)
        pack_num = tf.cast(pack_num, "float32")
        pick_num = tf.cast(pick_num, "float32")
        packs = tf.cast(packs, "float32")
        pack_size = tf.cast(pack_size, "float32")

        def get_weights(weights, batch_index1):
            pn = tf.gather(pack_num, batch_index1)
            px = tf.gather(packs, batch_index1)
            pin = tf.gather(pick_num, batch_index1)
            ps = tf.gather(pack_size, batch_index1)
            xCoordinate = tf.divide(pn, px)
            yCoordinate = tf.divide(pin, ps)
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
            floorYIndex = tf.cast(floorYIndex, 'int32')
            floorXIndex = tf.cast(floorXIndex, 'int32')
            ceilXIndex = tf.cast(ceilXIndex, 'int32')
            ceilYIndex = tf.cast(ceilYIndex, 'int32')
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
        def get_casting_probability(lands, lands_for_card):
            total_devotion = tf.reduce_sum(lands_for_card)
            probability = tf.constant(1, dtype=tf.float32)
            total_lands = tf.constant(0)
            for i in range(5):
                on_color_lands = tf.cast(tf.gather(lands, i), 'int32')
                devotion = tf.cast(tf.gather(lands_for_card, i), "int32")
                probability = probability * tf.gather_nd(self.prob_to_play, (devotion,
                                                                             on_color_lands))
                total_lands = total_lands + tf.cond(devotion > 0, lambda: on_color_lands,
                                                    lambda: tf.constant(0))
            return probability * tf.gather_nd(self.prob_to_play, (total_devotion, total_lands))

        def transform_similarity(similarity):
            similarity_multiplier = 1 / (1 - self.similarity_clip)
            scaled = similarity_multiplier\
                * tf.minimum(tf.maximum(tf.constant(0, dtype=tf.float32),
                                        similarity - self.similarity_clip),
                             1 - self.similarity_clip)
            transformed = tf.squeeze(-tf.math.log(1 - scaled) * 5)
            return tf.cond(tf.math.is_inf(transformed), lambda: tf.constant(10, dtype=tf.float32),
                           lambda: transformed)

        def rating_oracle(lands, batch_index1, batch_index2):
            rating = tf.gather_nd(cards_rating, (batch_index1, batch_index2))
            land_requirements = tf.gather_nd(cards_land_requirements, (batch_index1, batch_index2))
            return rating * get_casting_probability(lands, land_requirements)

        def pick_synergy_oracle(lands, batch_index1, batch_index2):
            card_embedding = tf.expand_dims(tf.gather_nd(cards_embeddings, (batch_index1, batch_index2)), 0)
            land_requirements = tf.gather_nd(cards_land_requirements, (batch_index1, batch_index2))
            card_probability = get_casting_probability(lands, land_requirements)

            def get_card_pick_synergy(other_index):
                other_index = tf.cast(other_index, 'int32')
                other_real_index = tf.gather_nd(picked, (batch_index1, other_index))

                def inner2():
                    other_embedding = tf.expand_dims(tf.gather_nd(picked_embeddings, (batch_index1, other_index)), 0)
                    similarity = tf.reshape(tf.keras.layers.dot([card_embedding, other_embedding],
                                                                normalize=True, axes=1), ())
                    synergy = transform_similarity(similarity)
                    other_land_requiremtns = tf.gather_nd(picked_land_requirements, (batch_index1, other_index))
                    other_probability = get_casting_probability(lands, other_land_requiremtns)
                    return tf.cond(tf.greater(other_probability, self.prob_to_include),
                                   lambda: synergy * other_probability,
                                   lambda: tf.constant(0, dtype=tf.float32))
                return tf.cond(tf.greater(other_index, 0), inner2, lambda: tf.constant(0, dtype=tf.float32))
            return card_probability * tf.reduce_sum(tf.map_fn(get_card_pick_synergy,
                                                              tf.range(MAX_PICKED, dtype=tf.float32),
                                                              parallel_iterations=MAX_PICKED))

        def fixing_oracle(lands, batch_index1, batch_index2):
            def calculate_fixing():
                combination = tf.expand_dims(tf.cast(tf.greater(lands, 2), "float32"), 0)
                colors = tf.expand_dims(tf.cast(tf.gather_nd(cards_colors, (batch_index1, batch_index2)), 'float32'), 0)
                overlap = 2.0 * tf.reshape(tf.keras.layers.dot([combination, colors], axes=1), ())
                return tf.cond(tf.gather_nd(cards_is_fetch, (batch_index1, batch_index2)), lambda: overlap,
                                  lambda: tf.cond(tf.gather_nd(cards_has_basic_land_types, (batch_index1, batch_index2)),
                                                  lambda: 0.75 * overlap,
                                                  lambda: 0.5 * overlap))
                return overlap
            return tf.cond(tf.gather_nd(cards_is_land, (batch_index1, batch_index2)),
                           calculate_fixing,
                           lambda: tf.constant(0, dtype=tf.float32))

        def internal_synergy_oracle(lands, batch_index1, batch_index2):
            def card_probability(card_index):
                card_index = tf.cast(card_index, 'int32')
                card_real_index = tf.gather_nd(picked, (batch_index1, card_index))
                land_requirements = tf.gather_nd(picked_land_requirements, (batch_index1, card_index))
                return tf.cond(tf.greater(card_real_index, 0), lambda: get_casting_probability(lands, land_requirements),
                               lambda: tf.constant(0, dtype=tf.float32))
            probabilities = tf.map_fn(card_probability, tf.range(MAX_PICKED, dtype=tf.float32),
                                      parallel_iterations=MAX_PICKED)

            def get_card_internal_synergy(card_index):
                card_index = tf.cast(card_index, 'int32')
                card_real_index = tf.gather_nd(picked, (batch_index1, card_index))
                card_probability = tf.gather(probabilities, card_index)

                def inner():
                    card_embedding = tf.expand_dims(tf.gather_nd(cards_embeddings, (batch_index1, card_index)), 0)

                    def synergy_with_card(other_index):
                        other_index = tf.cast(other_index, 'int32')
                        other_real_index = tf.gather_nd(picked, (batch_index1, other_index))
                        other_probability = tf.gather(probabilities, other_index)

                        def inner2():
                            other_embedding = tf.expand_dims(tf.gather_nd(cards_embeddings, (batch_index1, card_index)), 0)
                            similarity = tf.keras.layers.dot([card_embedding, other_embedding],
                                                             normalize=True, axes=1)
                            synergy = transform_similarity(similarity)
                            return tf.squeeze(synergy * other_probability)
                        return tf.cond(tf.logical_and(tf.greater_equal(other_probability,
                                                                       self.prob_to_include),
                                                      tf.greater(other_real_index, 0)),
                                       inner2, lambda: tf.constant(0, dtype=tf.float32))
                    return card_probability * tf.reduce_sum(tf.map_fn(synergy_with_card,
                                                                      tf.range(MAX_PICKED, dtype=tf.float32),
                                                                      parallel_iterations=16))
                return tf.cond(tf.logical_and(tf.greater(card_real_index, 0),
                                              tf.greater_equal(card_probability, self.prob_to_include)),
                               inner, lambda: tf.constant(0, dtype=tf.float32))
            synergy_scores = tf.map_fn(get_card_internal_synergy, tf.range(MAX_PICKED, dtype=tf.float32),
                                       parallel_iterations=16)
            return tf.reduce_sum(synergy_scores)

        def openness_oracle(lands, batch_index1, batch_index2):
            def get_card_openness(card_index):
                card_index = tf.cast(card_index, 'int32')
                card_real_index = tf.gather_nd(seen, (batch_index1, card_index))
                land_requirements = tf.gather_nd(seen_land_requirements, (batch_index1, card_index))
                card_probability = get_casting_probability(lands, land_requirements)
                card_rating = tf.gather_nd(seen_rating, (batch_index1, card_index))
                return tf.cond(tf.logical_and(tf.greater(card_real_index, 0),
                                              tf.greater_equal(card_probability,
                                                               self.prob_to_include)),
                               lambda: card_probability * card_rating,
                               lambda: tf.constant(0, dtype=tf.float32))
            return tf.reduce_sum(tf.map_fn(get_card_openness, tf.range(MAX_SEEN, dtype=tf.float32)))

        def color_oracle(lands, batch_index1, batch_index2):
            def get_card_contribution(card_index):
                card_index = tf.cast(card_index, 'int32')
                card_real_index = tf.gather_nd(picked, (batch_index1, card_index))
                land_requirements = tf.gather_nd(picked_land_requirements, (batch_index1, card_index))
                card_probability = get_casting_probability(lands, land_requirements)
                card_rating = tf.gather_nd(picked_rating, (batch_index1, card_index))
                return tf.cond(tf.logical_and(tf.greater(card_real_index, 0),
                                              tf.greater_equal(card_probability,
                                                               self.prob_to_include)),
                               lambda: card_probability * card_rating,
                               lambda: tf.constant(0, dtype=tf.float32))
            return tf.reduce_sum(tf.map_fn(get_card_contribution, tf.range(MAX_PICKED, dtype=tf.float32)))

        oracles = [rating_oracle, pick_synergy_oracle, fixing_oracle, internal_synergy_oracle,
                   openness_oracle, color_oracle]
        weights = [self.rating_weights, self.pick_synergy_weights, self.fixing_weights,
                   self.internal_synergy_weights, self.openness_weights, self.colors_weights]

        def get_score(lands, batch_index1, batch_index2):
            score = tf.squeeze(tf.cast(rating_oracle(lands, batch_index1, batch_index2), 'float32'))\
                * tf.squeeze(tf.cast(get_weights(self.rating_weights, batch_index1), 'float32'))
            score = score + tf.squeeze(tf.cast(pick_synergy_oracle(lands, batch_index1, batch_index2), 'float32'))\
                * tf.squeeze(tf.cast(get_weights(self.pick_synergy_weights, batch_index1), 'float32'))
            score = score + tf.squeeze(tf.cast(fixing_oracle(lands, batch_index1, batch_index2), 'float32'))\
                * tf.squeeze(tf.cast(get_weights(self.fixing_weights, batch_index1), 'float32'))
            score = score + tf.squeeze(tf.cast(internal_synergy_oracle(lands, batch_index1, batch_index2), 'float32'))\
                * tf.squeeze(tf.cast(get_weights(self.internal_synergy_weights, batch_index1), 'float32'))
            score = score + tf.squeeze(tf.cast(openness_oracle(lands, batch_index1, batch_index2), 'float32'))\
                * tf.squeeze(tf.cast(get_weights(self.openness_weights, batch_index1), 'float32'))
            score = score + tf.squeeze(tf.cast(color_oracle(lands, batch_index1, batch_index2), 'float32'))\
                * tf.squeeze(tf.cast(get_weights(self.colors_weights, batch_index1), 'float32'))
            return score

        def do_climb(batch_index1, batch_index2):
            batch_index1 = tf.cast(batch_index1, 'int32')
            batch_index2 = tf.cast(batch_index2, 'int32')
            card_index = tf.gather_nd(cards, (batch_index1, batch_index2))

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
                                                                                lambda: tf.gather(lands, i))),
                                              tf.range(5))
                        new_value = tf.cond(tf.greater(tf.gather(lands, remove_index), 0),
                                            lambda: get_score(new_lands, batch_index1, batch_index2),
                                            lambda: tf.constant(0, dtype=tf.float32))
                        final_lands = tf.cond(tf.greater(new_value, final_value), lambda: new_lands,
                                              lambda: final_lands)
                print(current_value, final_value, final_lands)
                return current_value, final_value, final_lands

            def do_while_loop():
                _, score, _ = tf.while_loop(climb_cond, try_climb, [tf.constant(-1, dtype=tf.float32),
                                                                    tf.constant(0, dtype=tf.float32),
                                                                    tf.constant([4, 4, 3, 3, 3])],
                                            parallel_iterations=1)
                return score
            return tf.cond(tf.greater(card_index, 0), do_while_loop, lambda: tf.constant(0, dtype=tf.float32))

        batch_size = tf.shape(cards)[0]
        pack_count = tf.shape(cards)[1]
        scores = tf.map_fn(lambda b1: tf.map_fn(lambda b2: do_climb(b1, b2),
                                                tf.range(pack_count, dtype=tf.float32),
                                                parallel_iterations=MAX_PACK_SIZE),
                           tf.range(batch_size, dtype=tf.float32),
                           parallel_iterations=16)
        return tf.nn.softmax(scores / temperature)
