import tensorflow as tf

from .timeseries.timeseries import log_timeseries


def get_mask(tensor):
    tensor._keras_mask = getattr(tensor, '_keras_mask', tf.cast(tf.ones_like(tensor), dtype=tf.bool))
    # noinspection PyProtectedMember
    return tensor._keras_mask


def mask_to_zeros(tensor, name=None):
    mask = get_mask(tensor)
    if len(mask.shape) < len(tensor.shape):
        mask = tf.expand_dims(mask, -1)
    result = tensor * tf.cast(mask, dtype=tensor.dtype, name=name)
    result._keras_mask = get_mask(tensor)
    return result


def cast(tensor, dtype, name=None):
    result = tf.cast(tensor, dtype=dtype, name=name)
    result._keras_mask = get_mask(tensor)
    return result


def normalize(tensor, axis=None, epsilon=1e-04, name=None):
    result = tf.math.l2_normalize(tensor, axis=axis, epsilon=epsilon, name=name)
    result._keras_mask = get_mask(tensor)
    return result


class DraftBot(tf.keras.models.Model):
    def __init__(self, num_cards, batch_size, embed_dims=64, num_heads=16, summary_period=1024,
                 l2_loss_weight=0.0, l1_loss_weight=0.0, **kwargs):
        kwargs.update({'dynamic': False})
        super(DraftBot, self).__init__(**kwargs)
        self.summary_period = summary_period
        self.batch_size = batch_size
        self.l2_loss_weight = l2_loss_weight
        self.l1_loss_weight = l1_loss_weight
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.oracle_weights = self.add_weight('oracle_weights', shape=(3, 15, 6), initializer='ones', trainable=True)
        self.card_rating_logits = self.add_weight('card_rating_logits', shape=(num_cards,), initializer='random_uniform',
                                                  trainable=True)
        self.card_embeddings = self.add_weight('card_embeddings', shape=(num_cards, embed_dims),
                                               initializer='random_uniform', trainable=True)
        self.self_attention = tf.keras.layers.MultiHeadAttention(num_heads, embed_dims // num_heads, name='self_attention')
        self.loss_metric = tf.keras.metrics.Mean()
        self.log_loss_metric = tf.keras.metrics.Mean()
        self.l2_loss_metric = tf.keras.metrics.Mean()
        self.l1_loss_metric = tf.keras.metrics.Mean()
        self.accuracy_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(1)
        self.top_2_accuracy_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(2)
        self.top_3_accuracy_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(3)
        self.average_prob_metric = tf.keras.metrics.Mean()
        # Our preprocessing guarantees that the human choice is always at index 0. Our calculations are permutation
        # invariant so this does not introduce any bias.
        self.default_target = tf.zeros((batch_size,), dtype=tf.int32)

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

    def call(self, inputs, training=False, mask=None):
        in_pack_cards, seen_cards, seen_counts, picked_cards, picked_counts, coords, coord_weights, prob_seens,\
            prob_pickeds, prob_in_packs = inputs
        # We precalculate 8 land combinations that are heuristically verified to be likely candidates for highest scores
        # and guarantee these combinations are diverse. To speed up computation we store the probabilities as unsigned
        # 8-bit fixed-point integers this converts back to float
        prob_seens = tf.cast(prob_seens, dtype=self.compute_dtype) / tf.constant(255, dtype=self.compute_dtype)
        prob_pickeds = tf.cast(prob_pickeds, dtype=self.compute_dtype) / tf.constant(255, dtype=self.compute_dtype)
        prob_in_packs = tf.cast(prob_in_packs, dtype=self.compute_dtype) / tf.constant(255, dtype=self.compute_dtype)
        # Ratings are in the range (0, 1) so we use a sigmoid activation.
        # We gather before the sigmoid to make the gradient sparse over the ratings which helps with LazyAdam.
        picked_ratings = tf.nn.sigmoid(tf.gather(self.card_rating_logits, picked_cards), name='picked_ratings')
        in_pack_ratings = tf.nn.sigmoid(tf.gather(self.card_rating_logits, in_pack_cards), name='in_pack_ratings')
        seen_ratings = tf.nn.sigmoid(tf.gather(self.card_rating_logits, seen_cards), name='seen_ratings')
        # We normalize here to allow computing the cosine similarity.
        normalized_in_pack_embeds = tf.math.l2_normalize(tf.gather(self.card_embeddings, in_pack_cards), axis=2, name='normalized_in_pack_embeds')
        picked_card_embeds = tf.gather(self.card_embeddings, picked_cards, name='picked_card_embeds')
        normalized_picked_embeds = tf.math.l2_normalize(picked_card_embeds, axis=2, name='normalized_picked_embeds')
        normalized_seen_embeds = tf.math.l2_normalize(tf.gather(self.card_embeddings, seen_cards), axis=2, name='normalized_seen_embeds')
        # We calculate the weight for each oracle as the linear interpolation of the weights on a 2d (3 x 15) grid.
        # There are 6 oracles so we can group them into one variable here for simplicity. The input coords are 4 points
        # on the 2d grid that we'll interpolate between. coord_weights is the weight for each of the four points.
        # Negative oracle weights don't make sense so we apply softplus here to ensure they are positive.
        oracle_weights_orig = tf.math.softplus(self.oracle_weights)
        oracle_weight_values = tf.gather_nd(oracle_weights_orig, coords, name='oracle_weight_values')  # (-1, 4, 6)
        oracle_weights = tf.einsum('bxo,bx->bo', oracle_weight_values, coord_weights, name='oracle_weights')
        # We don't care about the positions without cards so we mask them out here.
        picked_card_embeds._keras_mask = picked_cards > 0
        attention_mask = tf.logical_and(tf.expand_dims(picked_cards > 0, 1), tf.expand_dims(picked_cards > 0, 2))
        # We use self-attention to model higher-order interactions in the pool of picked cards
        pool_attentions = self.self_attention(picked_card_embeds, picked_card_embeds, attention_mask=attention_mask)
        # We sum weighted by the casting probabilities to collapse down to a single embedding then normalize for cosine similarity.
        unnormalized_pool_embed = tf.einsum('bpe,blp->ble', pool_attentions, prob_pickeds, name='pool_embed')
        pool_embed = tf.math.l2_normalize(unnormalized_pool_embed, axis=2, epsilon=1e-04, name='pool_embed_normalized')
        # These are the per-card oracles to choose between cards in the pack given a lands configuration.
        # The rating oracle for each card is its rating times its casting probability.
        rating_scores = tf.einsum('blc,bc->bcl', prob_in_packs, in_pack_ratings, name='rating_scores')
        # The pick synergy oracle for each card is the cosine similarity between its embedding and the pools embedding
        # times the cards casting probability.
        pick_synergy_scores = tf.einsum('ble,bce,blc->bcl', pool_embed, normalized_in_pack_embeds, prob_in_packs, name='pick_synergy_scores')
        # These are per-land-configuration oracles that help choose between different land configurations.
        # The colors oracle for a land configuration is the mean rating times casting probability of the cards in the pool.
        colors_scores = tf.einsum('blp,bp,b->bl', prob_pickeds, picked_ratings, 1 / tf.math.maximum(picked_counts, 1), name='colors_scores')
        # The openness oracle for a land configuration is the mean rating times casting probability of the cards that
        # have been seen this draft with cards that have been seen multiple times included multiple times.
        openness_scores = tf.einsum('bls,bs,b->bl', prob_seens, seen_ratings, 1 / tf.math.maximum(seen_counts, 1), name='openness_scores')
        # The internal synergy oracle for a land configuration is the mean cosine similarity times casting probability
        # of the cards that have been picked this draft.
        internal_synergy_scores = tf.einsum('ble,bpe,blp,b->bl', pool_embed, normalized_picked_embeds, prob_pickeds,
                                            1 / tf.math.maximum(picked_counts, 1), name='internal_synergies_scores')
        # The seen synergy oracle for a land configuration is the mean cosine similarity times casting probability
        # of the cards that have been seen this draft with cards that have been seen multiple times included multiple times.
        seen_synergy_scores = tf.einsum('ble,bse,bls,b->bl',pool_embed, normalized_seen_embeds, prob_seens,
                                        1 / tf.math.maximum(seen_counts, 1), name='seen_synergies_scores')
        # Combine the oracle scores linearly according to the oracle weights to get a score for every card/land-configuration pair.
        concat_option_scores = tf.stack([rating_scores, pick_synergy_scores], axis=3)
        concat_lands_scores = tf.stack([internal_synergy_scores, seen_synergy_scores, colors_scores, openness_scores], axis=2)
        option_scores = tf.einsum('bclo,bo->bcl', concat_option_scores, oracle_weights[:, :2])
        lands_scores = tf.einsum('blo,bo->bl', concat_lands_scores, oracle_weights[:, 2:])
        scores = option_scores + tf.expand_dims(lands_scores, 1)
        # Here we compute softmax(logsumexp(scores, 2)) with the operations broken apart to allow optimizing the calculation.
        # Since logsumexp and softmax are translation invariant we shrink the scores so the max score is 0 to reduce numerical instability.
        max_scores = tf.stop_gradient(tf.reduce_max(scores, [1, 2], keepdims=True))
        # This is needed to allow masking out the positions without cards so they don't participate in the softmax or logsumexp computations.
        in_pack_mask = tf.expand_dims(tf.cast(in_pack_cards > 0, dtype=tf.float32), 2)
        lse_scores = tf.math.reduce_sum(tf.math.exp(scores - max_scores) * in_pack_mask, 2, name='lse_exp_scores')
        # Since the first operation of softmax is exp and the last of logsumexp is log we can combine them into a no-op.
        choice_probs = lse_scores / tf.math.reduce_sum(lse_scores, 1, keepdims=True)

        # This is all logging for tensorboard. It can't easily be factored into a separate function since it uses so many
        # local variables.
        if training and tf.summary.experimental.get_step() % self.summary_period == 0:
            rating_diffs = tf.reduce_max(rating_scores, [1, 2]) - tf.math.reduce_min(rating_scores, [1, 2])
            pick_synergy_diffs = tf.reduce_max(pick_synergy_scores, [1, 2]) - tf.math.reduce_min(pick_synergy_scores, [1, 2])
            internal_synergy_diffs = tf.math.reduce_max(internal_synergy_scores, 1) - tf.math.reduce_min(internal_synergy_scores, 1)
            seen_synergy_diffs = tf.math.reduce_max(seen_synergy_scores, 1) - tf.math.reduce_min(seen_synergy_scores, 1)
            colors_diffs = tf.reduce_max(colors_scores, 1) - tf.math.reduce_min(colors_scores, 1)
            openness_diffs = tf.reduce_max(openness_scores, 1) - tf.math.reduce_min(openness_scores, 1)
            num_cards_in_pack = tf.reduce_sum(in_pack_mask, [1, 2])
            ratings_orig = tf.nn.sigmoid(self.card_rating_logits[1:])
            max_probs = tf.math.reduce_max(choice_probs, 1)
            max_score = tf.math.reduce_max(scores, [1, 2])
            min_score = tf.math.reduce_min(scores, [1, 2])
            max_diff = max_score - min_score
            min_correct_prob = tf.math.reduce_min(choice_probs[:, 0])
            max_correct_prob = tf.math.reduce_max(choice_probs[:, 0])
            temperatures = tf.math.reduce_sum(oracle_weights_orig, axis=2)
            in_top_1 = tf.cast(tf.math.in_top_k(self.default_target, choice_probs, 1), tf.float32)
            in_top_2 = tf.cast(tf.math.in_top_k(self.default_target, choice_probs, 2), tf.float32)
            in_top_3 = tf.cast(tf.math.in_top_k(self.default_target, choice_probs, 3), tf.float32)

            def to_timeline(key, values, **kwargs):
                tiled_values = tf.expand_dims(values, 1) * coord_weights
                total_values = tf.tensor_scatter_nd_add(tf.zeros((3, 15), dtype=tf.float32), coords, tiled_values)
                count_values = tf.tensor_scatter_nd_add(tf.zeros((3, 15), dtype=tf.float32), coords, coord_weights)
                log_timeseries(key, total_values / count_values, **kwargs)
            with tf.xla.experimental.jit_scope(compile_ops=False):
                tf.summary.histogram('weights/card_ratings', ratings_orig)
                log_timeseries(f'weights/oracles/temperatures', temperatures, start_index=1)
                log_timeseries(f'weights/oracles/rating', oracle_weights_orig[:, :, 0], start_index=1)
                log_timeseries(f'weights/oracles/rating/relative', oracle_weights_orig[:, :, 0] / temperatures, start_index=1)
                log_timeseries(f'weights/oracles/pick_synergy', oracle_weights_orig[:, :, 1], start_index=1)
                log_timeseries(f'weights/oracles/pick_synergy/relative', oracle_weights_orig[:, :, 1] / temperatures, start_index=1)
                log_timeseries(f'weights/oracles/internal_synergy', oracle_weights_orig[:, :, 2], start_index=1)
                log_timeseries(f'weights/oracles/internal_synergy/relative', oracle_weights_orig[:, :, 2] / temperatures, start_index=1)
                log_timeseries(f'weights/oracles/seen_synergy', oracle_weights_orig[:, :, 3], start_index=1)
                log_timeseries(f'weights/oracles/seen_synergy/relative', oracle_weights_orig[:, :, 3] / temperatures, start_index=1)
                log_timeseries(f'weights/oracles/colors', oracle_weights_orig[:, :, 4], start_index=1)
                log_timeseries(f'weights/oracles/colors/relative', oracle_weights_orig[:, :, 4] / temperatures, start_index=1)
                log_timeseries(f'weights/oracles/openness', oracle_weights_orig[:, :, 5], start_index=1)
                log_timeseries(f'weights/oracles/openness/relative', oracle_weights_orig[:, :, 5] / temperatures, start_index=1)
                tf.summary.histogram(f'outputs/oracles/ratings/diffs', rating_diffs)
                to_timeline(f'outputs/oracles/ratings/diffs/timeline', rating_diffs)
                tf.summary.histogram(f'outputs/oracles/ratings/correct', rating_scores[:, 0])
                tf.summary.histogram(f'outputs/oracles/pick_synergy/diffs', pick_synergy_diffs)
                to_timeline(f'outputs/oracles/pick_synergy/diffs/timeline', pick_synergy_diffs)
                tf.summary.histogram(f'outputs/oracles/pick_synergy/correct', tf.reduce_max(pick_synergy_scores[:, 0], 1))
                tf.summary.histogram(f'outputs/oracles/internal_synergy', internal_synergy_scores)
                tf.summary.histogram(f'outputs/oracles/internal_synergy/diffs', internal_synergy_diffs)
                to_timeline(f'outputs/oracles/internal_synergy/diffs/timeline', internal_synergy_diffs)
                tf.summary.histogram(f'outputs/oracles/seen_synergy', seen_synergy_scores)
                tf.summary.histogram(f'outputs/oracles/seen_synergy/diffs', seen_synergy_diffs)
                to_timeline(f'outputs/oracles/seen_synergy/diffs/timeline', seen_synergy_diffs)
                tf.summary.histogram(f'outputs/oracles/colors', colors_scores)
                tf.summary.histogram(f'outputs/oracles/colors/diffs', colors_diffs)
                to_timeline(f'outputs/oracles/colors/diffs/timeline', colors_diffs)
                tf.summary.histogram(f'outputs/oracles/openness', openness_scores)
                tf.summary.histogram(f'outputs/oracles/openness/diffs', openness_diffs)
                to_timeline(f'outputs/oracles/openness/diffs/timeline', openness_diffs)
                tf.summary.histogram(f'outputs/scores/correct', scores[:, 0])
                tf.summary.histogram('outputs/scores/diffs', max_diff)
                to_timeline('outputs/scores/diffs/timeline', max_diff)
                tf.summary.histogram('outputs/scores/max', max_score)
                tf.summary.histogram('outputs/scores/min', min_score)
                tf.summary.histogram('outputs/scores/lse/correct', lse_scores[:, 0])
                tf.summary.histogram('outputs/probs/correct/normalized', num_cards_in_pack * choice_probs[:, 0])
                tf.summary.histogram('outputs/probs/correct', choice_probs[:, 0])
                tf.summary.scalar('outputs/probs/correct/min', min_correct_prob)
                tf.summary.scalar('outputs/probs/correct/one_minus_max', 1 - max_correct_prob)
                tf.summary.histogram('outputs/probs/max', max_probs)
                tf.summary.histogram('outputs/probs/max/normalized', num_cards_in_pack * max_probs)
                to_timeline(f'outputs/probs/timeline', choice_probs[:, 0], start_index=1)
                to_timeline(f'outputs/accuracy/timeline', in_top_1, start_index=1)
                to_timeline(f'outputs/accuracy_top_2/timeline', in_top_2, start_index=1)
                to_timeline(f'outputs/accuracy_top_3/timeline', in_top_3, start_index=1)
        return choice_probs

    def _update_metrics(self, loss, log_loss, l2_loss, l1_loss, probs):
        self.loss_metric.update_state(loss)
        self.log_loss_metric.update_state(log_loss)
        self.l2_loss_metric.update_state(l2_loss)
        self.l1_loss_metric.update_state(l1_loss)
        self.accuracy_metric.update_state(self.default_target, probs)
        self.top_2_accuracy_metric.update_state(self.default_target, probs)
        self.top_3_accuracy_metric.update_state(self.default_target, probs)
        self.average_prob_metric.update_state(probs[:, 0])
        return {
            'loss': self.loss_metric.result(),
            'log_loss': self.log_loss_metric.result(),
            'l2_loss': self.l2_loss_metric.result(),
            'l1_loss': self.l1_loss_metric.result(),
            'accuracy': self.accuracy_metric.result(),
            'accuracy_top_2': self.top_2_accuracy_metric.result(),
            'accuracy_top_3': self.top_3_accuracy_metric.result(),
            'average_prob_correct': self.average_prob_metric.result(),
        }

    def calculate_loss(self, data, training=False):
        probs = self(data, training=training)
        log_loss = tf.reduce_mean(-tf.math.log(probs[:, 0] + 1e-16))
        oracle_weights = tf.math.softplus(self.oracle_weights)
        l2_loss = tf.reduce_sum(tf.cast(oracle_weights * oracle_weights, dtype=tf.float32))
        l1_loss = tf.math.reduce_sum(tf.cast(oracle_weights, dtype=tf.float32))
        loss = log_loss + self.l2_loss_weight * l2_loss + self.l1_loss_weight * l1_loss
        return loss, log_loss, l2_loss, l1_loss, probs

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss, log_loss, l2_loss, l1_loss, probs = self.calculate_loss(data[0], training=True)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return self._update_metrics(loss, log_loss, l2_loss, l1_loss, probs)

    def test_step(self, data):
        loss, log_loss, l2_loss, l1_loss, probs = self.calculate_loss(data[0], training=False)
        return self._update_metrics(loss, log_loss, l2_loss, l1_loss, probs)

    @property
    def metrics(self):
        return [self.loss_metric, self.log_loss_metric, self.l2_loss_metric, self.l1_loss_metric,
                self.accuracy_metric, self.top_2_accuracy_metric, self.top_3_accuracy_metric, self.average_prob_metric]
