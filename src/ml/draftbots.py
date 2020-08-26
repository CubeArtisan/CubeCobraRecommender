import tensorflow as tf

PACK_SIZE = 15
PACKS = 3


class DraftBot(tf.keras.models.Model):
    def __init__(self, card_counts, card_embeddings):
        super(DraftBot, self).__init__()
        self.card_count = card_counts
        self.ratings = tf.keras.layers.Embedding(card_counts + 1, 1, mask_zero=True)
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
        ])
        self.prob_to_include = tf.Variable(0.67, dtype=tf.float32)
        self.card_embeddings = card_embeddings

    def call(self, inputs, training=None, mask=None):
        card, picked, seen, pickNum, packNum, packs, packSize = inputs

        def get_weights(weights):
            xCoordinate = tf.divide(packNum, packs)
            yCoordinate = tf.divide(pickNum, packSize)
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

        def get_rating(lands):
            rating = tf.reshape(self.ratings(card_index), (-1,))
            tf.one_hot(card_index, self.card_count)