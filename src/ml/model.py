import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

if __name__ == "__main__":
    import sys
    from os.path import dirname as dir
    sys.path.append(dir(sys.path[0]))

from ml.ml_utils import generate_paths, MAX_PATH_LENGTH, NUM_INPUT_PATHS
from ml.keras_attention_layer import AttentionLayer

"""
Below is what was used for training the most recent version of the model:

Optimizer: Adagrad, with default hyperparameters

Loss: Binary Crossentropy + MSE(adj_mtx,decoded_for_reg)
    - adj_mtx is the adjacency matrix created by create_mtx.py
    and then updated such that each row sums to 1.
    - decoded_for_reg is an output of the model

Epochs: 100

Batch Size: 64
"""
CNN_SIZE = 32
CNN_WINDOW_SIZE = 4
EMBED_SIZE = 64
CATEGORICAL_EMBED_SIZE = 256
CONTINUOUS_EMBED_SIZE = 64
PATH_EMBED_SIZE = 512
CARD_EMBED_SIZE = 256


class CardEncoder(Model):
    """
    Encode cards to vectors
    """
    def __init__(self, name, vocab_dict, max_paths, max_path_length, width,
                 continuous_feature_count, categorical_feature_count):
        super().__init__()
        self.assigned_name = name
        self.max_paths = max_paths
        self.max_path_length = max_path_length
        self.width = width
        self.continuous_feature_count = continuous_feature_count
        self.categorical_feature_count = categorical_feature_count
        self.cnn = tf.keras.layers.Conv2D(CNN_SIZE, (CNN_WINDOW_SIZE, 300))
        print(len(vocab_dict))
        embedding_tensor = [0 for _ in range(len(vocab_dict))]
        for index, embedding in vocab_dict.values():
            embedding_tensor[index] = embedding.astype(np.float32)
        embedding_tensor = tf.convert_to_tensor(embedding_tensor)
        self.continuous_dropout = tf.keras.layers.Dropout(0.5, name='continuous_dropout')
        self.dropout = tf.keras.layers.Dropout(0.25, name='dropout')
        self.flatten = tf.keras.layers.Flatten()
        self.embedding_tensor = tf.Variable(embedding_tensor)
        self.embed_paths = Dense(PATH_EMBED_SIZE, activation="tanh")
        self.embed_dense = Dense(CARD_EMBED_SIZE, activation="tanh")
        self.continuous_dense1 = Dense(CONTINUOUS_EMBED_SIZE*2, activation="tanh")
        self.continuous_dense2 = Dense(CONTINUOUS_EMBED_SIZE, activation="tanh")
        self.categorical_dense = Dense(CATEGORICAL_EMBED_SIZE, activation="tanh")
        self.attention_layer = AttentionLayer(name=self.assigned_name + "_attention")
        print('preprocessed cards')

    def call(self, x, training=None, mask=None):
        paths, continuous_features, categorical_features = x
        batch_size = tf.shape(paths)[0]
        categorical_features_embed = tf.gather(self.embedding_tensor, categorical_features) # batch_size, width, feature_count, 300
        categorical_features_flat = tf.reshape(categorical_features_embed,
                                               (batch_size, self.width,
                                                300 * self.categorical_feature_count))
        categorical_features_dense = self.categorical_dense(categorical_features_flat)
        continuous_features_flat = tf.reshape(continuous_features, (batch_size, self.width,
                                                               self.continuous_feature_count))
        continuous_features_embed1 = self.continuous_dense1(continuous_features_flat)
        continuous_features_embed = self.continuous_dense2(continuous_features_embed1)
        continuous_features_embed = self.continuous_dropout(continuous_features_embed, training=training)
        all_paths_embed = tf.gather(self.embedding_tensor, paths) # batch_size, width, max_paths, max_path_length, 300
        flat_paths_embed = tf.reshape(all_paths_embed, [batch_size * self.max_paths * self.width,
                                                        self.max_path_length, 300, 1])
        final_cnn_state = self.cnn(flat_paths_embed)
        flattened_cnn_state = self.flatten(final_cnn_state)
        flattened_dense = self.embed_paths(flattened_cnn_state)
        by_card = tf.reshape(flattened_dense,
                             (batch_size * self.width, self.max_paths, PATH_EMBED_SIZE))
        code_vectors, attention_weights = self.attention_layer([by_card])
        code_vectors = tf.reshape(code_vectors, (batch_size, self.width, PATH_EMBED_SIZE))
        with_features = tf.concat([code_vectors, continuous_features_embed,
                                   categorical_features_dense], 2)
        with_features = self.dropout(with_features, training=training)
        embedded = self.embed_dense(with_features)
        embedded = tf.math.l2_normalize(embedded, 2)
        return embedded


class Encoder(Model):
    """
    Encoder part of the model -> compress dimensionality
    """
    def __init__(self, name, cards, max_cube_size):
        super().__init__()
        self.assigned_name = name
        self.encoded_1 = Dense(EMBED_SIZE * 8, activation='relu', name=name + "_e1")
        self.encoded_2 = Dense(EMBED_SIZE * 4, activation='relu', name=name + "_e2")
        self.encoded_3 = Dense(EMBED_SIZE * 2, activation='relu', name=name + "_e3")
        self.bottleneck = Dense(EMBED_SIZE, activation='relu',
                                name=name + "_bottleneck")

    def call(self, x, training=False, mask=None):
        encoded = self.encoded_1(x)
        encoded = self.encoded_2(encoded)
        encoded = self.encoded_3(encoded)
        return self.bottleneck(encoded)


class Decoder(Model):
    """
    Decoder part of the model -> expand from compressed latent
        space back to the input space
    """
    def __init__(self, name, output_dim, output_act):
        super().__init__()
        self.decoded_1 = Dense(EMBED_SIZE * 2, activation='relu', name=name + "_d1")
        self.decoded_2 = Dense(EMBED_SIZE * 4, activation='relu', name=name + "_d2")
        self.decoded_3 = Dense(EMBED_SIZE * 8, activation='relu', name=name + "_d3")
        self.reconstruct = Dense(output_dim, activation=output_act,
                                 name=name + "_reconstruction")

    def call(self, x, **kwargs):
        decoded = self.decoded_1(x)
        decoded = self.decoded_2(decoded)
        decoded = self.decoded_3(decoded)
        return self.reconstruct(decoded)


class CardEncoderWrapper(Model):
    """
    AutoEncoder build as a recommender system based on the following idea:

        If our input is a binary vector where 1 represents the presence of an
        item in a collection, then an autoencoder trained
    """
    def __init__(self, *args):
        super().__init__()
        self.card_encoder = CardEncoder("card_encoder", *args)
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def summary(self, **kwargs):
        super().summary(**kwargs)
        self.card_encoder.summary(**kwargs)

    def call(self, input, **kwargs):
        """
        input contains two things:
            input[0] = the binary vectors representing the collections
            input[1] = a diagonal matrix of size (self.N X self.N)

        We run the same encoder for each type of input, but with different
        decoders. This is because the goal is to make sure that the compression
        for collections still does a reasonable job compressing individual
        items. So a penalty term (regularization) is added to the model in the
        ability to reconstruct the probability distribution (adjacency matrix)
        on the item level from the encoding.

        The hope is that this regularization enforces this conditional
        probability to be embedded in the recommendations. As the individual
        items must pull towards items represented strongly within the graph.
        """
        a, b = tf.unstack(input, 2, 1)
        a_embed = self.card_encoder(a)
        b_embed = self.card_encoder(b)
        a_embed = tf.reshape(a_embed, (-1, CARD_EMBED_SIZE))
        b_embed = tf.reshape(b_embed, (-1, CARD_EMBED_SIZE))
        similarity = tf.keras.layers.dot([a_embed, b_embed], axes=1, normalize=True)
        similarity = tf.reshape(similarity, (-1, 1))
        return self.dense(similarity)


class CC_Recommender(Model):
    """
    AutoEncoder build as a recommender system based on the following idea:

        If our input is a binary vector where 1 represents the presence of an
        item in a collection, then an autoencoder trained
    """
    def __init__(self, cards, max_cube_size, card_embeddings):
        super().__init__()
        self.N = len(cards)
        self.encoder = Encoder("encoder", cards, max_cube_size)
        # sigmoid because input is a binary vector we want to reproduce
        self.decoder = Decoder("main", self.N, output_act='sigmoid')
        # softmax because the graph information is probabilities
        self.decoder_for_reg = Decoder("reg", self.N, output_act='softmax')
        all_paths = generate_paths(cards)
        our_paths = []
        for a in all_paths:
            a += [[0 for _ in range(MAX_PATH_LENGTH)] for _ in range(len(a), NUM_INPUT_PATHS)]
            a = np.random.choice(a, NUM_INPUT_PATHS)
            our_paths.append(a)
        self.card_embeddings = tf.constant(card_embeddings)


    def call(self, input, **kwargs):
        """
        input contains two things:
            input[0] = the binary vectors representing the collections
            input[1] = a diagonal matrix of size (self.N X self.N)

        We run the same encoder for each type of input, but with different
        decoders. This is because the goal is to make sure that the compression
        for collections still does a reasonable job compressing individual
        items. So a penalty term (regularization) is added to the model in the
        ability to reconstruct the probability distribution (adjacency matrix)
        on the item level from the encoding.

        The hope is that this regularization enforces this conditional
        probability to be embedded in the recommendations. As the individual
        items must pull towards items represented strongly within the graph.
        """
        flatten = tf.keras.layers.Flatten()
        x, identity = input
        x = flatten(tf.gather(self.card_embeddings, x))
        identity = flatten(tf.gather(self.card_embedding, identity))
        encoded = self.encoder(x)
        reconstruction = self.decoder(encoded)
        encode_for_reg = self.encoder(identity)
        decoded_for_reg = self.decoder_for_reg(encode_for_reg)
        return reconstruction, decoded_for_reg
