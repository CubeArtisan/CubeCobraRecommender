import tensorflow as tf


class FilteredBinaryAccuracy(tf.keras.metrics.Metric):
    def __init__(self, filter_value, name, **kwargs):
        super(FilteredBinaryAccuracy, self).__init__(name=name, **kwargs)
        self.filter_value = filter_value
        self.filtered_count = self.add_weight(name='filtered_count', initializer='zeros')
        self.filtered_accurate = self.add_weight(name='filtered_accurate', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        if self.filter_value:
            expectations = tf.greater(y_true, 0.5)
            trues_pred = tf.greater(y_pred, 0.5)
            result = tf.logical_and(expectations, trues_pred)
        else:
            expectations = tf.less(y_true, 0.5)
            falses_pred = tf.less(y_pred, 0.5)
            result = tf.logical_and(expectations, falses_pred)
        expectations = tf.cast(expectations, "float32")
        result = tf.cast(result, "float32")
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            expectations = tf.multiply(expectations, sample_weight)
            result = tf.multiply(result, sample_weight)
        self.filtered_count.assign_add(tf.reduce_sum(expectations))
        self.filtered_accurate.assign_add(tf.reduce_sum(result))

    def result(self):
        return tf.divide(self.filtered_accurate, self.filtered_count)


class TripletFilteredAccuracy(tf.keras.metrics.Metric):
    def __init__(self, filter_value, name, **kwargs):
        super(TripletFilteredAccuracy, self).__init__(name=name, **kwargs)
        self.count = self.add_weight(name='count', initializer='zeros')
        self.filtered_accurate = self.add_weight(name='filtered_accurate', initializer='zeros')
        self.filter_value = filter_value

    def update_state(self, y_true, y_pred, sample_weight=None):
        a, b, c = tf.unstack(y_pred, num=3, axis=1)
        if self.filter_value:
            ab = tf.reshape(tf.norm(tf.subtract(a, b), axis=1), (-1,))
            counts = tf.ones_like(ab, dtype=tf.float32)
            expectations = tf.less(ab, 0.2)
        else:
            ac = tf.reshape(tf.norm(tf.subtract(a, c), axis=1), (-1,))
            counts = tf.ones_like(ac, dtype=tf.float32)
            expectations = tf.greater(ac, 0.2)
        expectations = tf.cast(expectations, "float32")
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            expectations = tf.multiply(expectations, sample_weight)
            counts = tf.multiply(counts, sample_weight)
        self.count.assign_add(tf.reduce_sum(counts))
        self.filtered_accurate.assign_add(tf.reduce_sum(expectations))

    def result(self):
        return tf.divide(self.filtered_accurate, self.count)


class TripletAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name, **kwargs):
        super(TripletAccuracy, self).__init__(name=name, **kwargs)
        self.count = self.add_weight(name='count', initializer='zeros')
        self.accurate = self.add_weight(name='accurate', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        a, b, c = tf.unstack(y_pred, num=3, axis=1)
        ab = tf.reshape(tf.norm(tf.subtract(a, b), axis=1), (-1,))
        ac = tf.reshape(tf.norm(tf.subtract(a, c), axis=1), (-1,))
        pos_expectations = tf.less(ab, 0.5)
        neg_expectations = tf.greater(ac, 0.5)
        pos_expectations = tf.cast(pos_expectations, "float32")
        neg_expectations = tf.cast(neg_expectations, "float32")
        counts = tf.fill(tf.shape(ab), 2.0)
        counts = tf.cast(counts, "float32")
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            pos_expectations = tf.multiply(pos_expectations, sample_weight)
            neg_expectations = tf.multiply(neg_expectations, sample_weight)
            counts = tf.multiply(counts, sample_weight)
        self.count.assign_add(tf.reduce_sum(counts))
        self.accurate.assign_add(tf.reduce_sum(pos_expectations))
        self.accurate.assign_add(tf.reduce_sum(neg_expectations))

    def result(self):
        return tf.divide(self.accurate, self.count)


class ContrastiveFilteredAccuracy(tf.keras.metrics.Metric):
    def __init__(self, filter_value, example_count, name, **kwargs):
        super(ContrastiveFilteredAccuracy, self).__init__(name=name, **kwargs)
        self.count = self.add_weight(name='count', initializer='zeros')
        self.filtered_accurate = self.add_weight(name='filtered_accurate', initializer='zeros')
        self.filter_value = filter_value
        self.example_count = example_count

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.convert_to_tensor(y_pred)
        a, examples = tf.split(y_pred, [1, self.example_count], axis=1)
        a = tf.tile(a, (1, self.example_count, 1))
        # a is [batch_size, 1, CARD_EMBED_SIZE]
        similarities = tf.reshape(tf.keras.layers.dot([a, examples], axes=2, normalize=True),
                                  (-1, self.example_count))
        pos, neg = tf.split(similarities, [1, self.example_count - 1], axis=1)
        pos_expectation = tf.greater(pos, 0.5)
        neg_expectation = tf.less(neg, 0.5)
        if self.filter_value is None:
            expectations = tf.concat([pos_expectation, neg_expectation], 1)
        elif self.filter_value:
            expectations = pos_expectation
        else:
            expectations = neg_expectation
        counts = tf.ones_like(expectations, dtype=tf.float32)
        expectations = tf.cast(expectations, "float32")
        # if sample_weight is not None:
        #     sample_weight = tf.cast(sample_weight, "float32")
        #     sample_weight = tf.tile(tf.reshape(sample_weight, (-1, 1)), [1, tf.shape()])
        #     expectations = tf.multiply(expectations, sample_weight)
        #     counts = tf.multiply(counts, sample_weight)
        counts = tf.reshape(counts, (-1,))
        expectations = tf.reshape(expectations, (-1,))
        self.count.assign_add(tf.reduce_sum(counts))
        self.filtered_accurate.assign_add(tf.reduce_sum(expectations))

    def result(self):
        return tf.divide(self.filtered_accurate, self.count)