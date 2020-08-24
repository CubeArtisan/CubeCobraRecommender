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