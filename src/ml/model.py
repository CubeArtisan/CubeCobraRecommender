import random
from collections import defaultdict

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

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
VOCAB_SIZE = 32
RNN_SIZE = 64
EMBED_SIZE = 64
CARD_EMBED_SIZE = 64


class CardEncoder(Model):
    """
    Encode cards to vectors
    """
    def __init__(self, name, vocab_count, max_paths, max_path_length):
        super().__init__()
        self.assigned_name = name
        self.max_paths = max_paths
        self.max_path_length = max_path_length
        self.embedding = tf.keras.layers.Embedding(vocab_count, VOCAB_SIZE,
                                                   name=self.assigned_name + "_vocab_embedding",
                                                   mask_zero=True)
        self.rnn_cell = tf.keras.layers.LSTMCell(RNN_SIZE // 2)
        self.rnn = tf.keras.layers.Bidirectional(
            layer=tf.keras.layers.RNN(self.rnn_cell, return_state=True),
            backward_layer=tf.keras.layers.RNN(self.rnn_cell, go_backwards=True,
                                               return_state=True),
            merge_mode="concat",
            name=self.assigned_name + "_bidirectional",
            dtype=tf.float32)
        # rnn_cell = tf.keras.layers.LSTMCell(RNN_SIZE)
        # self.rnn = tf.keras.layers.RNN(rnn_cell, dtype=tf.float32, return_state=True)
        self.embed_dense = Dense(CARD_EMBED_SIZE, activation="tanh")
        print('preprocessed cards')

    def call(self, x, training=None, mask=None):
        all_paths_embed = self.embedding(x)
        flat_paths_embed = tf.reshape(all_paths_embed, [-1, self.max_path_length, VOCAB_SIZE])
        _, state_fw, _, state_bw, _ = self.rnn(inputs=flat_paths_embed)
        final_rnn_state = tf.concat([state_fw, state_bw], -1)

        concated = tf.reshape(final_rnn_state, [-1, self.max_paths * RNN_SIZE])
        if self.max_path_length * RNN_SIZE == CARD_EMBED_SIZE:
            return concated
        else:
            return self.embed_dense(concated)


class Encoder(Model):
    """
    Encoder part of the model -> compress dimensionality
    """
    def __init__(self, name, cards, max_cube_size):
        super().__init__()
        self.assigned_name = name
        self.card_encoder = CardEncoder(name + "_card_encoder", cards, max_cube_size)
        self.flatten = tf.keras.layers.Flatten(name=name + "_flatten")
        self.encoded_1 = Dense(EMBED_SIZE * 8, activation='relu', name=name + "_e1")
        self.encoded_2 = Dense(EMBED_SIZE * 4, activation='relu', name=name + "_e2")
        self.encoded_3 = Dense(EMBED_SIZE * 2, activation='relu', name=name + "_e3")
        self.bottleneck = Dense(EMBED_SIZE, activation='relu',
                                name=name + "_bottleneck")

    def call(self, x, training=False, mask=None):
        print(x.shape)
        encoded = self.card_encoder(x)
        print(x.shape)
        encoded = self.encoded_1(encoded)
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
        self.activation = tf.keras.layers.Activation("sigmoid")
        self.dot = tf.keras.layers.Dot(1, normalize=True)

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
        return self.activation(self.dot([a_embed, b_embed]))


class CC_Recommender(Model):
    """
    AutoEncoder build as a recommender system based on the following idea:

        If our input is a binary vector where 1 represents the presence of an
        item in a collection, then an autoencoder trained
    """
    def __init__(self, cards, max_cube_size):
        super().__init__()
        self.N = len(cards)
        self.encoder = Encoder("encoder", cards, max_cube_size)
        # sigmoid because input is a binary vector we want to reproduce
        self.decoder = Decoder("main", self.N, output_act='sigmoid')
        # softmax because the graph information is probabilities
        self.decoder_for_reg = Decoder("reg", self.N, output_act='softmax')

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
        x, identity = input
        encoded = self.encoder(x)
        print(tf.shape(encoded))
        reconstruction = self.decoder(encoded)
        encode_for_reg = self.encoder(identity)
        decoded_for_reg = self.decoder_for_reg(encode_for_reg)
        return reconstruction, decoded_for_reg
