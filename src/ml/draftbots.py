import tensorflow as tf

from src.ml.timeseries.timeseries import log_timeseries


class SetEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_cards, embed_dims, final_dims=None, item_dropout_rate=0.0, dense_dropout_rate=0.0, **kwargs):
        super(SetEmbedding, self).__init__(**kwargs)
        self.num_cards = num_cards
        self.embed_dims = embed_dims
        self.item_dropout_rate = item_dropout_rate
        self.dense_dropout_rate = dense_dropout_rate
        self.final_dims = final_dims or embed_dims
        
    def get_config(self):
        config = super(SetEmbedding, self).get_config()
        config.update({
            "num_cards": self.num_cards,
            "embed_dims": self.embed_dims,
            "item_dropout_rate": self.item_dropout_rate,
            "dense_dropout_rate": self.dense_dropout_rate,
            "final_dims": self.final_dims,
        })
        return config

    def build(self, input_shape):
        self.embeddings = tf.keras.layers.Embedding(self.num_cards, self.embed_dims, mask_zero=True, input_length=input_shape[1], name='item_embeddings')
        self.upcast_2x = tf.keras.layers.Dense(2 * self.embed_dims, activation='tanh', use_bias=True, name='upcast_2x')
        self.upcast_4x = tf.keras.layers.Dense(4 * self.embed_dims, activation='tanh', use_bias=True, name='upcast_4x')
        self.downcast_2x = tf.keras.layers.Dense(2 * self.embed_dims, activation='tanh', use_bias=True, name='downcast_2x')
        self.downcast_final = tf.keras.layers.Dense(self.final_dims, activation='linear', use_bias=True, name='downcast_final')
        self.item_dropout = tf.keras.layers.Dropout(rate=self.item_dropout_rate, noise_shape=(input_shape[0], input_shape[1], 1))
        self.dense_dropout = tf.keras.layers.Dropout(rate=self.dense_dropout_rate)
        
    def call(self, inputs, training=False, mask=None):
        item_embeds = self.item_dropout(self.embeddings(inputs), training=training)
        summed_embeds = tf.math.reduce_sum(item_embeds * tf.expand_dims(tf.cast(inputs > 0, dtype=self.compute_dtype), 2), 1)
        upcast_2x = self.dense_dropout(self.upcast_2x(summed_embeds), training=training)
        upcast_4x = self.dense_dropout(self.upcast_4x(upcast_2x), training=training)
        downcast_2x = self.dense_dropout(self.downcast_2x(upcast_4x), training=training)
        return self.downcast_final(downcast_2x)
        

class DraftBot(tf.keras.models.Model):
    def __init__(self, num_cards, embed_dims=64, summary_period=1024, seen_dims=16, picked_dims=32,
                 rating_l2_loss_weight=0.0, rating_l1_loss_weight=0.0, oracle_stddev_loss_weight=0.0,
                 dropout_picked_rate=0.0, dropout_seen_rate=0.0, dropout_dense_rate=0.0, **kwargs):
        kwargs.update({'dynamic': False})
        super(DraftBot, self).__init__(**kwargs)
        self.num_cards = num_cards
        self.embed_dims = embed_dims
        self.summary_period = summary_period
        
        self.loss_weights = {k: tf.constant(v, dtype=tf.float32) for k, v in
            (('log_loss', 1), ('rating_l1_loss', rating_l1_loss_weight), ('rating_l2_loss', rating_l2_loss_weight),
             ('oracle_stddev_loss', oracle_stddev_loss_weight))}
        self.mean_metrics = {k: tf.keras.metrics.Mean() for k in
            ('loss', 'log_loss', 'rating_l2_loss', 'rating_l1_loss', 'embedding_l2_loss', 'pick_synergy_l2_loss',
             'seen_synergy_l2_loss', 'oracle_stddev_loss')}
        self.accuracy_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(1)
        self.top_2_accuracy_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(2)
        self.top_3_accuracy_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(3)
        self.dropout_picked_rate = dropout_picked_rate
        self.dropout_seen_rate = dropout_seen_rate
        self.dropout_dense_rate = dropout_dense_rate
        self.seen_dims = seen_dims
        self.picked_dims = picked_dims
        
    def build(self, input_shape):
        self.oracle_weights = self.add_weight('oracle_weights', shape=(3, 15, 3), initializer=tf.constant_initializer(8), trainable=True)
        self.in_pack_rating_logits = self.add_weight('in_pack_rating_logits', shape=(self.num_cards,), initializer=tf.constant_initializer(0), trainable=True)
        self.card_embeddings = tf.keras.layers.Embedding(self.num_cards, self.embed_dims, mask_zero=True, input_length=input_shape[0][1])
        self.pool_embedding = SetEmbedding(self.num_cards, self.picked_dims, final_dims=self.embed_dims, item_dropout_rate=self.dropout_picked_rate, dense_dropout_rate=self.dropout_dense_rate, name='pool_set_embedding')
        self.seen_embedding = SetEmbedding(self.num_cards, self.seen_dims, final_dims=self.embed_dims, item_dropout_rate=self.dropout_seen_rate, dense_dropout_rate=self.dropout_dense_rate, name='seen_set_embedding')
        
        self.embed_mult = tf.constant([[0 for _ in range(self.embed_dims)]] + [[1 for _ in range(self.embed_dims)] for _ in range(self.num_cards - 1)], dtype=self.compute_dtype)
        self.default_target = tf.constant(0, shape=(input_shape[0][0],), dtype=tf.int32)

    def get_config(self):
        config = super(DraftBot, self).get_config()
        config.update({
            "num_cards": self.num_cards,
            "batch_size": self.batch_size,
            "embed_dims": self.embed_dims,
            "summary_period": self.summary_period,
            "rating_l2_loss_weight": self.rating_l2_loss_weight.numpy(),
            "rating_l1_loss_weight": self.rating_l1_loss_weight.numpy(),
            "dropout_picked_rate": self.dropout_picked_rate,
            "dropout_seen_rate": self.dropout_seen_rate,
            "dropout_dense_rate": self.dropout_dense_rate,
        })
        return config
    
    def _get_ratings(self, logits, training=False, name=None):
        return tf.nn.sigmoid(tf.constant(8, dtype=self.compute_dtype) * logits, name=name)
        
    def _get_weights(self, training=False):
        return tf.math.softplus(tf.constant(1, dtype=self.compute_dtype) * self.oracle_weights)
        
    def call(self, inputs, training=False, mask=None):
        if len(inputs) == 10:
            return 0
        in_pack_cards, seen_cards, picked_cards, coords, coord_weights = inputs
        # Ratings are in the range (0, 1) so we use a sigmoid activation.
        in_pack_card_ratings = self._get_ratings(self.in_pack_rating_logits, training=training, name='in_pack_card_ratings')
        in_pack_ratings = tf.gather(in_pack_card_ratings, in_pack_cards, name='in_pack_ratings')
        in_pack_embeds = self.card_embeddings(in_pack_cards)
        pool_embed = self.pool_embedding(picked_cards, training=training)
        seen_embed = self.seen_embedding(seen_cards, training=training)
        # We calculate the weight for each oracle as the linear interpolation of the weights on a 2d (3 x 15) grid.
        # There are 3 oracles so we can group them into one variable here for simplicity. The input coords are 4 points
        # on the 2d grid that we'll interpolate between. coord_weights is the weight for each of the four points.
        # Negative oracle weights don't make sense so we apply softplus here to ensure they are positive.
        oracle_weights_orig = self._get_weights(training=training)
        oracle_weight_values = tf.gather_nd(oracle_weights_orig, coords, name='oracle_weight_values')  # (-1, 4, 3)
        oracle_weights = tf.einsum('bxo,bx->bo', oracle_weight_values, coord_weights, name='oracle_weights')

        pick_synergy_scores = tf.nn.sigmoid(tf.einsum('be,bce->bc', pool_embed, in_pack_embeds, name='pick_synergy_scores'))
        seen_synergy_scores = tf.nn.sigmoid(tf.einsum('be,bce->bc', seen_embed, in_pack_embeds, name='seen_synergy_scores'))
        rating_scores = in_pack_ratings
        oracle_scores = tf.stack([rating_scores, pick_synergy_scores, seen_synergy_scores], axis=2)
        scores = tf.einsum('bco,bo->bc', oracle_scores, oracle_weights, name='in_pack_scores')
        # This is all logging for tensorboard. It can't easily be factored into a separate function since it uses so many
        # local variables.
        if training and tf.summary.experimental.get_step() % self.summary_period == 0:        
            in_pack_mask = tf.cast(in_pack_cards > 0, dtype=tf.float32)
            inv_in_pack_mask = 1 - in_pack_mask
            num_in_pack = tf.math.reduce_sum(in_pack_mask, 1)
            choice_probs = tf.nn.softmax(scores, axis=1)
            max_probs = tf.math.reduce_max(choice_probs, 1)
            max_score = tf.math.reduce_max(scores, 1)
            min_score = tf.math.reduce_min(scores + tf.reduce_max(max_score) * inv_in_pack_mask, 1)
            max_diff = max_score - min_score
            min_correct_prob = tf.math.reduce_min(choice_probs[:, 0])
            max_correct_prob = tf.math.reduce_max(choice_probs[:, 0])
            temperatures = tf.math.reduce_sum(oracle_weights_orig, axis=2)
            relative_oracle_weights = oracle_weights_orig / tf.expand_dims(temperatures, 2)
            in_top_1 = tf.cast(tf.math.in_top_k(self.default_target, choice_probs, 1), tf.float32)
            in_top_2 = tf.cast(tf.math.in_top_k(self.default_target, choice_probs, 2), tf.float32)
            in_top_3 = tf.cast(tf.math.in_top_k(self.default_target, choice_probs, 3), tf.float32)

            def to_timeline(key, values, **kwargs):
                tiled_values = tf.expand_dims(values, 1) * coord_weights
                total_values = tf.tensor_scatter_nd_add(tf.zeros((3, 15), dtype=tf.float32), coords, tiled_values)
                count_values = tf.tensor_scatter_nd_add(tf.zeros((3, 15), dtype=tf.float32), coords, coord_weights)
                log_timeseries(key, total_values / count_values, **kwargs)

            with tf.xla.experimental.jit_scope(compile_ops=False):
                to_timeline(f'outputs/accuracy/timeline', in_top_1, start_index=1)
                to_timeline(f'outputs/accuracy_top_2/timeline', in_top_2, start_index=1)
                to_timeline(f'outputs/accuracy_top_3/timeline', in_top_3, start_index=1)
                to_timeline(f'outputs/adjusted_accuracy/timeline', in_top_1 * num_in_pack, start_index=1)
                to_timeline(f'outputs/adjusted_accuracy_top_2/timeline', in_top_2 * num_in_pack, start_index=1)
                to_timeline(f'outputs/adjusted_accuracy_top_3/timeline', in_top_3 * num_in_pack, start_index=1)
                log_timeseries(f'weights/oracles/temperature', temperatures, start_index=1)
                oracle_details = (
                    ('rating', 0, rating_scores),
                    ('pick_synergy', 1, pick_synergy_scores),
                    ('seen_synergy', 2, seen_synergy_scores)
                )
                for name, idx, values in oracle_details:
                    log_timeseries(f'weights/oracles/multiplier/{name}', oracle_weights_orig[:, :-1, idx], start_index=1)
                    log_timeseries(f'weights/oracles/relative/{name}', relative_oracle_weights[:, :, idx],
                                   start_index=1)
                    diffs = tf.reduce_max(values, 1) - tf.reduce_min(values + inv_in_pack_mask, 1)
                    values_with_temp = values[:, 0] * oracle_weights[:, idx]
                    diffs_with_temp = diffs * oracle_weights[:, idx]
                    to_timeline(f'outputs/oracles/values/weighted/correct/{name}', values_with_temp, start_index=1)
                    tf.summary.histogram(f'outputs/oracles/values/correct/{name}', values[:, 0])
                    to_timeline(f'outputs/oracles/diffs/weighted/correct/{name}', diffs_with_temp, start_index=1)
                    tf.summary.histogram(f'outputs/oracles/diffs/{name}', diffs)
                tf.summary.scalar('probs/correct/min', min_correct_prob)
                tf.summary.scalar('probs/correct/max/from_one', 1 - max_correct_prob)
                tf.summary.histogram('probs/correct', choice_probs[:, 0])
                to_timeline(f'outputs/probs/correct/timeline', choice_probs[:, 0], start_index=1)
                tf.summary.histogram('probs/max', max_probs)
                tf.summary.histogram('probs/max_adjusted', max_probs * num_in_pack)
                tf.summary.histogram(f'outputs/scores/correct', scores[:, 0])
                tf.summary.histogram(f'outputs/scores/diffs', max_diff)
                to_timeline(f'outputs/scores/diffs/timeline', max_diff, start_index=1)
                tf.summary.histogram('weights/ratings', in_pack_card_ratings)
        if training:
            return scores, oracle_scores
        else:
            return scores

    def _update_metrics(self, mean_metrics, probs):
        for key, value in mean_metrics.items():
            self.mean_metrics[key].update_state(value)
        self.accuracy_metric.update_state(self.default_target, probs)
        self.top_2_accuracy_metric.update_state(self.default_target, probs)
        self.top_3_accuracy_metric.update_state(self.default_target, probs)
        result = {
            'accuracy': self.accuracy_metric.result(),
            'accuracy_top_2': self.top_2_accuracy_metric.result(),
            'accuracy_top_3': self.top_3_accuracy_metric.result(),
        }
        result.update({k: v.result() for k, v in self.mean_metrics.items() if k in mean_metrics})
        return result

    def calculate_loss(self, data, training=False):
        if training:
            scores, oracle_scores = self(data, training=True)
            in_pack_mask = tf.cast(data[0] > 0, dtype=tf.float32)
            num_in_pack = tf.math.reduce_sum(in_pack_mask, 1)
            total_in_packs = tf.math.reduce_sum(in_pack_mask)
            oracle_scores = oracle_scores * tf.expand_dims(in_pack_mask, 2)
            mean_oracle_scores = tf.math.reduce_sum(oracle_scores, [0, 1], keepdims=True) / tf.reshape(total_in_packs, (-1, 1, 1))
            oracle_scores_shifted_mean = oracle_scores - mean_oracle_scores * tf.expand_dims(in_pack_mask, 2)
            oracle_scores_shifted_05 = (oracle_scores - 0.5) * tf.expand_dims(in_pack_mask, 2)
            tf.summary.histogram('oracle_scores_shifted_mean', oracle_scores_shifted_mean)
            
            losses = {
                'rating_l1_loss': tf.math.reduce_sum(oracle_scores_shifted_05 * oracle_scores_shifted_05) / total_in_packs,
                'rating_l2_loss': tf.math.reduce_mean(self.in_pack_rating_logits * self.in_pack_rating_logits),
                'oracle_stddev_loss': -tf.math.reduce_sum(oracle_scores_shifted_mean * oracle_scores_shifted_mean) / total_in_packs,
            }
        else:
            scores = self(data, training=False)
            losses = {}
        losses['log_loss'] = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.default_target, logits=scores)
        losses['loss'] = sum(weight * losses[k] for k, weight in self.loss_weights.items() if k in losses)
        return losses, scores

    def train_step(self, data):
        filtered_data = [data[0][0], data[0][1], data[0][3], data[0][5], data[0][6]]
        with tf.GradientTape() as tape:
            losses, scores = self.calculate_loss(filtered_data, training=True)
        gradients = tape.gradient(losses['loss'], self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return self._update_metrics(losses, scores)

    def test_step(self, data):
        filtered_data = [data[0][0], data[0][1], data[0][3], data[0][5], data[0][6]]
        losses, scores = self.calculate_loss(filtered_data, training=False)
        return self._update_metrics(losses, scores)

    @property
    def metrics(self):
        return list(self.mean_metrics.values()) + [self.accuracy_metric, self.top_2_accuracy_metric, self.top_3_accuracy_metric]
