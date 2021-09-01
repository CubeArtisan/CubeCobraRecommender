import tensorflow as tf

from src.ml.timeseries.timeseries import log_timeseries


class DraftBot(tf.keras.models.Model):
    def __init__(self, card_ratings, initial_embeddings, batch_size, embed_dims=64, num_heads=16, summary_period=1024,
                 l2_loss_weight=0.0, l1_loss_weight=0.0, **kwargs):
        kwargs.update({'dynamic': False})
        super(DraftBot, self).__init__(**kwargs)
        num_cards = len(card_ratings)
        self.summary_period = summary_period
        self.batch_size = batch_size
        self.l2_loss_weight = l2_loss_weight
        self.l1_loss_weight = l1_loss_weight
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.loss_metric = tf.keras.metrics.Mean()
        self.synergy_loss_metric = tf.keras.metrics.Mean()
        self.prob_loss_metric = tf.keras.metrics.Mean()
        self.log_loss_metric = tf.keras.metrics.Mean()
        self.l2_loss_metric = tf.keras.metrics.Mean()
        self.l1_loss_metric = tf.keras.metrics.Mean()
        self.accuracy_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(1)
        self.top_2_accuracy_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(2)
        self.top_3_accuracy_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(3)
        self.average_prob_metric = tf.keras.metrics.Mean()
        # Our preprocessing guarantees that the human choice is always at index 0. Our calculations are permutation
        # invariant so this does not introduce any bias.
        self.rating_mult = tf.constant(embed_dims, dtype=self.compute_dtype)
        self.default_target = tf.zeros((batch_size,), dtype=tf.int32)
        self.oracle_weights = self.add_weight('oracle_weights', shape=(3, 15, 6), initializer=tf.constant_initializer([[[8, 8, 8, 1, 1, 1] for _ in range(15)] for _ in range(3)]),
                                              trainable=True)
        self.in_pack_rating_logits = self.add_weight('in_pack_rating_logits', shape=(num_cards,),
                                                  initializer=tf.constant_initializer(card_ratings),
                                                  trainable=True)
        self.picked_rating_logits = self.add_weight('picked_rating_logits', shape=(num_cards,),
                                                  initializer=tf.constant_initializer(card_ratings),
                                                  trainable=True)
        self.seen_rating_logits = self.add_weight('seen_rating_logits', shape=(num_cards,),
                                                  initializer=tf.constant_initializer(card_ratings),
                                                  trainable=True)
        self.card_embeddings = self.add_weight('card_embeddings', shape=(num_cards, embed_dims),
                                               initializer=tf.constant_initializer(initial_embeddings), trainable=True)
        self.card_pool_embeddings = self.add_weight('card_pool_embeddings', shape=(num_cards, embed_dims),
                                                    initializer=tf.constant_initializer(initial_embeddings), trainable=True)
        # self.card_seen_embeddings = self.add_weight('card_seen_embeddings', shape=(num_cards, embed_dims),
        #                                             initializer=tf.constant_initializer(initial_embeddings / 10), trainable=True)
        self.dense_1_weight = self.add_weight('dense_1_weight', shape=(embed_dims, 2 * embed_dims), initializer='random_normal', trainable=True)
        self.dense_1_bias = self.add_weight('dense_1_bias', shape=(embed_dims * 2,), initializer='zeros', trainable=True)
        # self.dense_2_weight = self.add_weight('dense_2_weight', shape=(embed_dims * 2, 4 * embed_dims), initializer='random_normal', trainable=True)
        # self.dense_2_bias = self.add_weight('dense_2_bias', shape=(embed_dims * 4,), initializer='zeros', trainable=True)
        # self.dense_3_weight = self.add_weight('dense_3_weight', shape=(embed_dims * 4, 2 * embed_dims), initializer='random_normal', trainable=True)
        # self.dense_3_bias = self.add_weight('dense_3_bias', shape=(embed_dims * 2,), initializer='zeros', trainable=True)
        self.dense_4_weight = self.add_weight('dense_4_weight', shape=(embed_dims * 2, embed_dims), initializer='random_normal', trainable=True)
        self.dense_5_weight = self.add_weight('dense_5_weight', shape=(embed_dims, 2 * embed_dims), initializer='random_normal', trainable=True)
        self.dense_5_bias = self.add_weight('dense_5_bias', shape=(embed_dims * 2,), initializer='zeros', trainable=True)
        # self.dense_6_weight = self.add_weight('dense_6_weight', shape=(embed_dims * 2, 4 * embed_dims), initializer='random_normal', trainable=True)
        # self.dense_6_bias = self.add_weight('dense_6_bias', shape=(embed_dims * 4,), initializer='zeros', trainable=True)
        # self.dense_7_weight = self.add_weight('dense_7_weight', shape=(embed_dims * 4, 2 * embed_dims), initializer='random_normal', trainable=True)
        # self.dense_7_bias = self.add_weight('dense_7_bias', shape=(embed_dims * 2,), initializer='zeros', trainable=True)
        self.dense_8_weight = self.add_weight('dense_8_weight', shape=(embed_dims * 2, embed_dims), initializer='random_normal', trainable=True)
        # self.dense_square = self.add_weight('dense_square_weight', shape=(embed_dims, embed_dims), initializer='random_normal', trainable=True)
        self.dropout_picked = tf.keras.layers.Dropout(rate=0.5, noise_shape=(batch_size, 48, 1))
        self.dropout_seen = tf.keras.layers.Dropout(rate=0.8, noise_shape=(batch_size, 400, 1))
        self.dropout_dense = tf.keras.layers.Dropout(rate=1 / embed_dims)
        # self.self_attention = tf.keras.layers.MultiHeadAttention(num_heads, 2 * embed_dims // num_heads, name='self_attention')
        embed_mult = [[0 for _ in range(self.embed_dims)]] + [[1 for _ in range(self.embed_dims)] for _ in range(num_cards - 1)]
        self.embed_mult = tf.constant(embed_mult, dtype=self.compute_dtype)

    def get_config(self):
        config = super(DraftBot, self).get_config()
        config.update({
            "num_cards": self.num_cards,
            "batch_size": self.batch_size,
            "embed_dims": self.embed_dims,
            "num_heads": self.num_heads,
            "summary_period": self.summary_period,
            "l2_loss_weight": self.l2_loss_weight,
            "l1_loss_weight": self.l1_loss_weight,
            "temp_weight": self.temp_weight,
        })
        return config
        
    def _get_linear_higher_pool_embedding(self, picked_card_embeds, picked_probs, training=False):
        higher_embeddings = tf.nn.bias_add(tf.einsum('bpe,ef->bpf', picked_card_embeds, self.summed_weights, name='higher_embeddings'),
                                           self.higher_embed_bias, name='higher_embeddings_biased')
        regular_embeddings = tf.einsum('bpf,blp,fe->ble', higher_embeddings, picked_probs, self.project_back, name='regular_embeddings')
        return tf.math.l2_normalize(tf.nn.bias_add(regular_embeddings, self.card_embeddings[0], name='biased_regular_embeddings'),
                                    axis=2, epsilon=1e-04, name='pool_embed')

    def _get_linear_pool_embedding(self, picked_card_embeds, picked_probs, training=False):
        summed_embeddings = tf.einsum('bpe,blp->ble', picked_card_embeds, picked_probs, name='summed_embeddings')
        return tf.math.l2_normalize(tf.nn.bias_add(summed_embeddings, self.card_pool_embeddings[0]), axis=2, epsilon=1e-04, name='pool_embed')
        
    def _get_nonlinear_pool_embedding(self, picked_card_embeds, picked_probs, training=False):
        summed_embeddings = tf.einsum('bpe,blp->ble', picked_card_embeds, picked_probs, name='summed_embeddings')
        upcast_1 = tf.math.tanh(tf.nn.bias_add(tf.einsum('ble,ef->blf', summed_embeddings, self.dense_1_weight, name='dense_1_weighted'), self.dense_1_bias, name='dense_1'))
        upcast_2 = tf.math.tanh(tf.nn.bias_add(tf.einsum('blf,fg->blg', upcast_1, self.dense_2_weight, name='dense_2_weighted'), self.dense_2_bias, name='dense_2'))
        downcast_1 = tf.math.tanh(tf.nn.bias_add(tf.einsum('blg,gf->blf', upcast_2, self.dense_3_weight, name='dense_3_weighted'), self.dense_3_bias, name='dense_3'))
        downcast_2 = tf.nn.bias_add(tf.einsum('blf,fe->ble', downcast_1, self.dense_4_weight, name='dense_4_weighted'), self.card_pool_embeddings[0], name='dense_4')
        return tf.math.l2_normalize(downcast_2, axis=2, epsilon=1e-04, name='pool_embed')

    def _get_attention_pool_embedding(self, picked_cards, picked_probs, training=False):
        picked_card_embeds = tf.gather(self.card_embeddings, picked_cards, name='picked_card_embeds')
        # We don't care about the positions without cards so we mask them out here.
        picked_card_embeds._keras_mask = picked_cards > 0
        attention_mask = tf.logical_and(tf.expand_dims(picked_cards > 0, 1), tf.expand_dims(picked_cards > 0, 2))
        # We use self-attention to model higher-order interactions in the pool of picked cards
        pool_attentions = self.self_attention(picked_card_embeds, picked_card_embeds, attention_mask=attention_mask)
        # We sum weighted by the casting probabilities to collapse down to a single embedding then normalize for cosine similarity.
        return tf.einsum('bpe,blp->ble', pool_attentions, picked_probs, name='unnormalized_pool_embed')
        
    def _get_pool_embedding_no_probs(self, picked_cards, training=False):
        picked_card_embeds = tf.gather(self.card_pool_embeddings * self.embed_mult, picked_cards, name='picked_card_embeds')
        summed_embeddings = tf.reduce_sum(self.dropout_picked(picked_card_embeds, training=training), axis=1)
        upcast_1 = self.dropout_dense(tf.math.tanh(tf.nn.bias_add(tf.einsum('be,ef->bf', summed_embeddings, self.dense_1_weight, name='pick_dense_1_weighted'), self.dense_1_bias, name='pick_dense_1')), training=training)
        # upcast_2 = self.dropout_dense(tf.math.tanh(tf.nn.bias_add(tf.einsum('bf,fg->bg', upcast_1, self.dense_2_weight, name='pick_dense_2_weighted'), self.dense_2_bias, name='pick_dense_2')), training=training)
        # downcast_1 = self.dropout_dense(tf.math.tanh(tf.nn.bias_add(tf.einsum('bg,gf->bf', upcast_2, self.dense_3_weight, name='pick_dense_3_weighted'), self.dense_3_bias, name='pick_dense_3')), training=training)
        downcast_2 = tf.nn.bias_add(tf.einsum('bf,fe->be', upcast_1, self.dense_4_weight, name='pick_dense_4_weighted'), self.card_pool_embeddings[0], name='pick_dense_4')
        return downcast_2
        
    def _get_seen_embedding_no_probs(self, seen_cards, training=False):
        seen_card_embeds = tf.gather(self.card_pool_embeddings * self.embed_mult, seen_cards, name='seen_card_embeds')
        summed_embeddings = self.dropout_dense(tf.reduce_sum(self.dropout_seen(seen_card_embeds, training=training), axis=1), training=training)
        upcast_1 = self.dropout_dense(tf.math.tanh(tf.nn.bias_add(tf.einsum('be,ef->bf', summed_embeddings, self.dense_5_weight, name='seen_dense_5_weighted'), self.dense_5_bias, name='seen_dense_5')), training=training)
        # upcast_2 = self.dropout_dense(tf.math.tanh(tf.nn.bias_add(tf.einsum('bf,fg->bg', upcast_1, self.dense_6_weight, name='seen_dense_2_weighted'), self.dense_6_bias, name='seen_dense_2')), training=training)
        # downcast_1 = self.dropout_dense(tf.math.tanh(tf.nn.bias_add(tf.einsum('bg,gf->bf', upcast_2, self.dense_7_weight, name='seen_dense_3_weighted'), self.dense_7_bias, name='seen_dense_3')), training=training)
        downcast_2 = tf.nn.bias_add(tf.einsum('bf,fe->be', upcast_1, self.dense_8_weight, name='seen_dense_4_weighted'), self.card_embeddings[0], name='seen_dense_4')
        return downcast_2

    def _get_pool_embedding(self, picked_cards, picked_probs, training=False):
        picked_card_embeds = tf.gather(self.card_pool_embeddings, picked_cards, name='picked_card_embeds')
        return self._get_nonlinear_pool_embedding(picked_card_embeds, picked_probs, training=training)
        
        # reciprocal_total_picked_prob = 1 / (tf.math.reduce_sum(picked_probs, axis=2) + 1e-04)
        # linear_pool_embeddings = tf.einsum('ble,ef->blf', self._get_linear_pool_embedding(picked_card_embeds, picked_probs, reciprocal_total_picked_prob), self.summed_weights)
        # nonlinear_pool_embeddings = tf.einsum('ble,ef->blf', self._get_tanh_pool_embedding(picked_card_embeds, picked_probs, reciprocal_total_picked_prob), self.nonlinear_weights)
        # higher_embedding = tf.nn.bias_add(linear_pool_embeddings + nonlinear_pool_embeddings, self.embed_bias)
        # combined_embedding = tf.einsum('blf,fe->ble', higher_embedding, self.project_back)
        # return tf.math.l2_normalize(combined_embedding, axis=2, epsilon=1e-04, name='pool_embedding')
    
    def _get_ratings(self, logits, training=False, name=None):
        return tf.nn.sigmoid(tf.constant(8, dtype=self.compute_dtype) * logits, name=name)
        
    def _get_weights(self, training=False):
        return tf.math.softplus(tf.constant(1, dtype=self.compute_dtype) * self.oracle_weights)
        
    def call(self, inputs, training=False, mask=None):
        if len(inputs) == 10:
            return 0
        in_pack_cards, seen_cards, picked_cards, coords, coord_weights = inputs
        # Ratings are in the range (0, 1) so we use a sigmoid activation.
        # We gather before the sigmoid to make the gradient sparse over the ratings which helps with LazyAdam.
        in_pack_card_ratings = self._get_ratings(self.in_pack_rating_logits, training=training, name='in_pack_card_ratings')
        in_pack_ratings = tf.gather(in_pack_card_ratings, in_pack_cards, name='in_pack_ratings')
        # We normalize here to allow computing the cosine similarity.
        in_pack_embeds = tf.gather(self.card_embeddings, in_pack_cards, name='in_pack_embeds')
        pool_embed = self._get_pool_embedding_no_probs(picked_cards, training=training)
        seen_embed = self._get_seen_embedding_no_probs(seen_cards, training=training)
        # We calculate the weight for each oracle as the linear interpolation of the weights on a 2d (3 x 15) grid.
        # There are 6 oracles so we can group them into one variable here for simplicity. The input coords are 4 points
        # on the 2d grid that we'll interpolate between. coord_weights is the weight for each of the four points.
        # Negative oracle weights don't make sense so we apply softplus here to ensure they are positive.
        oracle_weights_orig = self._get_weights(training=training)
        oracle_weight_values = tf.gather_nd(oracle_weights_orig, coords, name='oracle_weight_values')  # (-1, 4, 6)
        oracle_weights = tf.einsum('bxo,bx->bo', oracle_weight_values, coord_weights, name='oracle_weights')
        # These are the per-card oracles to choose between cards in the pack given a lands configuration.
        # The pick synergy oracle for each card is the cosine similarity between its embedding and the pools embedding
        # times the cards casting probability.
        
        pick_synergy_scores = tf.nn.sigmoid(tf.einsum('be,bce->bc', pool_embed, in_pack_embeds, name='pick_synergy_scores'))
        seen_synergy_scores = tf.nn.sigmoid(tf.einsum('be,bce->bc', seen_embed, in_pack_embeds, name='seen_synergy_scores'))
        rating_scores = in_pack_ratings
        scores = tf.einsum('bco,bo->bc', tf.stack([rating_scores, pick_synergy_scores, seen_synergy_scores], axis=2),
                           oracle_weights[:, 0:3], name='in_pack_scores')
        scores = tf.expand_dims(scores, 2)
        # Here we compute softmax(max(scores, axis=2)) with the operations broken apart to allow optimizing the calculation.
        # Since logsumexp and softmax are translation invariant we shrink the scores so the max score is 0 to reduce numerical instability.
        max_scores = tf.stop_gradient(tf.reduce_max(scores, [1, 2], keepdims=True, name='max_scores'))
        exp_scores = tf.reduce_sum(tf.math.exp(scores - max_scores, name='exp_scores_pre_sum'), 2, name='exp_scores')
        # # Since the first operation of softmax is exp and the last of logsumexp is log we can combine them into a no-op.
        choice_probs = exp_scores / tf.math.reduce_sum(exp_scores, 1, keepdims=True, name='total_exp_scores')

        synergy_scores = pick_synergy_scores[:, 0] + seen_synergy_scores[:, 0]
        # This is all logging for tensorboard. It can't easily be factored into a separate function since it uses so many
        # local variables.
        if training and tf.summary.experimental.get_step() % self.summary_period == 0:
            # num_cards_in_pack = tf.reduce_sum(in_pack_mask, 1)
            max_probs = tf.math.reduce_max(choice_probs, 1)
            max_score = tf.math.reduce_max(scores, [1, 2])
            min_score = tf.math.reduce_min(scores, [1, 2])
            max_diff = max_score - min_score
            min_correct_prob = tf.math.reduce_min(choice_probs[:, 0])
            max_correct_prob = tf.math.reduce_max(choice_probs[:, 0])
            temperatures = tf.math.reduce_sum(oracle_weights_orig[:, :, 0:3], axis=2)
            relative_oracle_weights = oracle_weights_orig[:, :, 0:3] / tf.expand_dims(temperatures, 2)
            in_top_1 = tf.cast(tf.math.in_top_k(self.default_target, choice_probs, 1), tf.float32)
            in_top_2 = tf.cast(tf.math.in_top_k(self.default_target, choice_probs, 2), tf.float32)
            in_top_3 = tf.cast(tf.math.in_top_k(self.default_target, choice_probs, 3), tf.float32)

            def to_timeline(key, values, **kwargs):
                tiled_values = tf.expand_dims(values, 1) * coord_weights
                total_values = tf.tensor_scatter_nd_add(tf.zeros((3, 15), dtype=tf.float32), coords, tiled_values)
                count_values = tf.tensor_scatter_nd_add(tf.zeros((3, 15), dtype=tf.float32), coords, coord_weights)
                log_timeseries(key, total_values / count_values, **kwargs)

            with tf.xla.experimental.jit_scope(compile_ops=False):
                tf.summary.histogram('weights/card_ratings/in_pack', in_pack_card_ratings)
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
                    diffs = tf.reduce_max(values, 1) - tf.math.reduce_min(values, 1)
                    diffs_with_temp = diffs * oracle_weights[:, idx]
                    relative_diffs = diffs_with_temp / max_diff
                    to_timeline(f'outputs/oracles/weighted_diffs/timeline/{name}', diffs_with_temp)
                    tf.summary.histogram(f'outputs/oracles/diffs/values/{name}', diffs)
                    to_timeline(f'outputs/oracles/relative/diffs/timeline/{name}', relative_diffs)
                    tf.summary.histogram(f'outputs/oracles/values/{name}/correct', values[:, 0])
                tf.summary.histogram(f'outputs/synergy_scores', synergy_scores)
                tf.summary.histogram(f'outputs/scores/diffs/correct', tf.reduce_max(scores[:, 0], 1) - min_score)
                tf.summary.histogram('outputs/scores/diffs', max_diff)
                to_timeline('outputs/scores/diffs/timeline', max_diff)
                tf.summary.histogram('outputs/probs/correct', choice_probs[:, 0])
                tf.summary.scalar('outputs/probs/correct/min', min_correct_prob)
                tf.summary.scalar('outputs/probs/correct/max_from_one', 1 - max_correct_prob)
                tf.summary.histogram('outputs/probs/max', max_probs)
                to_timeline(f'outputs/probs/correct/timeline', choice_probs[:, 0], start_index=1)
                to_timeline(f'outputs/accuracy/timeline', in_top_1, start_index=1)
                to_timeline(f'outputs/accuracy_top_2/timeline', in_top_2, start_index=1)
                to_timeline(f'outputs/accuracy_top_3/timeline', in_top_3, start_index=1)
        return choice_probs, synergy_scores

    def _update_metrics(self, loss, log_loss, l2_loss, l1_loss, synergy_loss, prob_loss, probs):
        self.loss_metric.update_state(loss)
        self.log_loss_metric.update_state(log_loss)
        self.prob_loss_metric.update_state(prob_loss)
        self.synergy_loss_metric.update_state(synergy_loss)
        self.l2_loss_metric.update_state(l2_loss)
        self.l1_loss_metric.update_state(l1_loss)
        self.accuracy_metric.update_state(self.default_target, probs)
        self.top_2_accuracy_metric.update_state(self.default_target, probs)
        self.top_3_accuracy_metric.update_state(self.default_target, probs)
        self.average_prob_metric.update_state(probs[:, 0])
        return {
            'loss': self.loss_metric.result(),
            'log_loss': self.log_loss_metric.result(),
            'prob_loss': self.prob_loss_metric.result(),
            'synergy_loss': self.synergy_loss_metric.result(),
            'l2_loss': self.l2_loss_metric.result(),
            'l1_loss': self.l1_loss_metric.result(),
            'accuracy': self.accuracy_metric.result(),
            'accuracy_top_2': self.top_2_accuracy_metric.result(),
            'accuracy_top_3': self.top_3_accuracy_metric.result(),
            'average_prob_correct': self.average_prob_metric.result(),
        }

    def calculate_loss(self, data, training=False):
        probs, synergy_scores = self(data, training=training)
        num_cards_in_pack = tf.reduce_sum(tf.cast(data[0] > 0, dtype=self.compute_dtype), 1)
        log_loss = tf.reduce_mean(-tf.math.log(probs[:, 0] + 1e-16) * num_cards_in_pack)
        prob_loss = -tf.reduce_mean(probs[:, 0] * num_cards_in_pack)
        synergy_loss = -tf.reduce_mean(synergy_scores)
        oracle_weights = self._get_weights(training=training)
        oracle_weights = oracle_weights[:, :, 0:2]
        l1_loss = tf.math.reduce_mean(tf.math.abs(self.in_pack_rating_logits)
                                       + tf.math.abs(self.picked_rating_logits)
                                       + tf.math.abs(self.seen_rating_logits))
        l2_loss = tf.math.reduce_mean(self.in_pack_rating_logits * self.in_pack_rating_logits
                                       + self.picked_rating_logits * self.picked_rating_logits
                                       + self.seen_rating_logits * self.seen_rating_logits)
        loss = log_loss + self.l2_loss_weight * l2_loss + self.l1_loss_weight * l1_loss
        return loss, log_loss, l2_loss, l1_loss, synergy_loss, prob_loss, probs

    def train_step(self, data):
        filtered_data = [data[0][0], data[0][1], data[0][3], data[0][5], data[0][6]]
        with tf.GradientTape() as tape:
            loss, log_loss, l2_loss, l1_loss, synergy_loss, prob_loss, probs = self.calculate_loss(filtered_data, training=True)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return self._update_metrics(loss, log_loss, l2_loss, l1_loss, synergy_loss, prob_loss, probs)

    def test_step(self, data):
        filtered_data = [data[0][0], data[0][1], data[0][3], data[0][5], data[0][6]]
        loss, log_loss, l2_loss, l1_loss, synergy_loss, prob_loss, probs = self.calculate_loss(filtered_data, training=False)
        return self._update_metrics(loss, log_loss, l2_loss, l1_loss, synergy_loss, prob_loss, probs)

    @property
    def metrics(self):
        return [self.loss_metric, self.log_loss_metric, self.l2_loss_metric, self.l1_loss_metric, self.synergy_loss_metric,
                self.accuracy_metric, self.top_2_accuracy_metric, self.top_3_accuracy_metric, self.average_prob_metric, self.prob_loss_metric]
