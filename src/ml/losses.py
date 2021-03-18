import tensorflow as tf

# class CustomCrossEntropy(tf.keras.losses.Loss):
#     def __init__(self, **kwargs):
#         super(CustomCrossEntropy, self).__init__(**kwargs)

#     def call(self, y_true, y_pred):
#         y_pred = tf.convert_to_tensor(y_pred)


class TripletLoss(tf.keras.losses.Loss):
    def __init__(self, margin, **kwargs):
        super(TripletLoss, self).__init__(**kwargs)
        self.margin = margin

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        a, b, c = tf.unstack(y_pred, num=3, axis=1)
        ab = tf.reshape(tf.norm(tf.subtract(a, b), axis=1), (-1,))
        ac = tf.reshape(tf.norm(tf.subtract(a, c), axis=1), (-1,))
        return tf.reduce_mean(tf.maximum(tf.subtract(ab, ac) + self.margin, 0))


class ContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, example_count, temperature, **kwargs):
        super(ContrastiveLoss, self).__init__(**kwargs)
        self.example_count = example_count
        self.temperature = temperature

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        a, examples = tf.split(y_pred, [1, self.example_count], axis=1)
        # a is [batch_size, 1, CARD_EMBED_SIZE]
        a = tf.tile(a, (1, self.example_count, 1))
        similarities = tf.reshape(tf.keras.layers.dot([a, examples], axes=2, normalize=True),
                                  (-1, self.example_count)) / self.temperature
        exp_similarities = tf.math.exp(similarities)
        pos, _ = tf.split(exp_similarities, [1, self.example_count - 1], 1)
        summed_exp = tf.reshape(tf.reduce_sum(exp_similarities, 1), (-1,))
        pos = tf.reshape(pos, (-1,))
        logs = -tf.math.log(tf.divide(pos, summed_exp))
        return tf.reduce_mean(logs)

