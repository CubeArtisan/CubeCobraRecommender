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
RNN_SIZE = 512
EMBED_SIZE = 64
CARD_EMBED_SIZE = 256


class CardEncoder(Model):
    """
    Encode cards to vectors
    """
    def __init__(self, name, vocab_dict, max_paths, max_path_length):
        super().__init__()
        self.assigned_name = name
        self.max_paths = max_paths
        self.max_path_length = max_path_length
        embedding_tensor = [0 for _ in range(len(vocab_dict))]
        for index, embedding in vocab_dict.values():
            embedding_tensor[index] = embedding.astype(np.float32)
        embedding_tensor = tf.convert_to_tensor(embedding_tensor)
        self.embedding_tensor = tf.Variable(embedding_tensor)
        self.rnn_cell = tf.keras.layers.LSTMCell(RNN_SIZE // 2)
        self.rnn = tf.keras.layers.Bidirectional(
            layer=tf.keras.layers.RNN(self.rnn_cell, return_state=True),
            backward_layer=tf.keras.layers.RNN(self.rnn_cell, go_backwards=True,
                                               return_state=True),
            merge_mode="concat",
            name=self.assigned_name + "_bidirectional",
            dtype=tf.float32)
        self.embed_dense = Dense(CARD_EMBED_SIZE, activation="tanh")
        self.time_distributed = tf.keras.layers.TimeDistributed(self.embed_dense,
                                                                name=self.assigned_name
                                                                     + "_time_distributed")
        self.attention_layer = AttentionLayer(name=self.assigned_name + "_attention")
        print('preprocessed cards')

    def call(self, x, training=None, mask=None):
        all_paths_embed = tf.gather(self.embedding_tensor, x)
        flat_paths_embed = tf.reshape(all_paths_embed, [-1, self.max_path_length, 300])
        _, state_fw, _, state_bw, _ = self.rnn(inputs=flat_paths_embed)
        final_rnn_state = tf.concat([state_fw, state_bw], -1)
        by_card = tf.reshape(final_rnn_state, [-1, self.max_paths, RNN_SIZE])
        by_card_dense = self.time_distributed(by_card)
        code_vectors, attention_weights = self.attention_layer([by_card_dense])
        return code_vectors


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
