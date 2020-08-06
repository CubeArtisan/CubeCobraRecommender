from typing import Dict, List, Union

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


class Encoder(Model):
    """
    Encoder part of the model -> compress dimensionality
    """
    def __init__(self, name, cards):
        super().__init__()
        self.__preprocess_cards(cards)
        # self.input_drop = Dropout(0.2)
        self.encoded_1 = Dense(512, activation='relu', name=name + "_e1")
        # self.e1_drop = Dropout(0.5)
        self.encoded_2 = Dense(256, activation='relu', name=name + "_e2")
        # self.e2_drop = Dropout(0.5)
        self.encoded_3 = Dense(128, activation='relu', name=name + "_e3")
        # self.e3_drop = Dropout(0.2)
        self.bottleneck = Dense(64, activation='relu',
                                name=name + "_bottleneck")

    def __preprocess_cards(self, cards):
        def convert_structure(structure: Union[list, dict, int, str, bool],
                              key: str, vocab_dict: Dict[str, int],
                              children: List[List[int]],
                              node_labels: List[int]) -> int:

            if isinstance(structure, list):
                if key in vocab_dict:
                    vocab = vocab_dict[key]
                else:
                    vocab = len(vocab_dict)
                    vocab_dict[key] = vocab
                our_children = []
                for index, child in enumerate(structure):
                    child_index = convert_structure(child, str(index),
                                                    vocab_dict, children,
                                                    node_labels)
                    our_children.append(child_index)
                for index in range(len(structure), len(children)):
                    our_children.append(-1)
                our_index = len(node_labels)
                for index, child_index in enumerate(our_children):
                    if len(children) <= index:
                        children.append([-1 for _ in node_labels])
                    if len(children[index]) <= our_index:
                        children[index].append(-1)
                    else:
                        children[index][our_index] = -1
                node_labels.append(vocab)
                return our_index
            elif isinstance(structure, dict):
                our_children = []
                if key in vocab_dict:
                    vocab = vocab_dict[key]
                else:
                    vocab = len(vocab_dict)
                    vocab_dict[key] = vocab
                for key, child in structure.items():
                    child_index = convert_structure(child, key, vocab_dict,
                                                    children, node_labels)
                    our_children.append(child_index)
                for _ in range(len(structure), len(children)):
                    our_children.append(-1)
                our_index = len(node_labels)
                for index, child_index in enumerate(our_children):
                    if len(children) <= index:
                        children.append([-1 for _ in node_labels])
                    if len(children[index]) <= our_index:
                        children[index].append(-1)
                    else:
                        children[index][our_index] = -1
                node_labels.append(vocab)
                return our_index
            else:
                key = f'{key}.{structure}'
                if key in vocab_dict:
                    vocab = vocab_dict[key]
                else:
                    vocab = len(vocab_dict)
                    vocab_dict[key] = vocab
                our_index = len(node_labels)
                node_labels.append(vocab)
                for index in range(len(children)):
                    if len(children[index]) <= our_index:
                        children[index].append(-1)
                    else:
                        children[index][our_index] = -1
                return our_index

        vocab_dict = {}
        children = []
        node_labels = []
        card_indices = []
        for card in cards:
            card_index = convert_structure(card, "", vocab_dict, children,
                                           node_labels)
            card_indices.append(card_index)
        children_count = len(children)
        node_count = len(node_labels)
        children = tf.constant([[child[i] for child in children]
                                for i in range(len(node_labels))])
        node_labels = tf.constant(node_labels)
        card_indices = tf.constant(card_indices)
        embedding = tf.Variable(tf.zeros((len(vocab_dict), 64)),
                                name="vocab_embedding")
        W = tf.Variable(tf.zeros((len(children) * 64, 64)), name="W")
        tensor_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True,
                                      clear_after_read=False,
                                      infer_shape=False)

        def true_cond(_, __):
            return tf.constant(True)

        def loop_body(tensor_array, i):
            node_label = tf.gather(node_labels, i)
            our_children = tf.gather(children, i)

            def get_child_array(i):
                child_index = tf.gather(our_children, i)
                child_tensor = tf.cond(tf.less(child_index, 0),
                                       tf.zeros(64),
                                       tf.gather(tensor_array, child_index))
                return child_tensor

            child_array = tf.concat([get_child_tensor(i)
                                     for i in range(children_count)], 0)

            vocab = tf.expand_dims(tf.gather(embedding, node_label), 0)
            node_tensor = tf.nn.relu(tf.add(vocab,
                                            tf.matmul(child_array.concat(),
                                                      W)))
            tensor_array = tensor_array.write(i, node_tensor)
            i = tf.add(i, 1)
            return tensor_array, i

        tensor_array, _ = tf.while_loop(true_cond, loop_body,
                                        [tensor_array, 0],
                                        parallel_iterations=1,
                                        maximum_iterations=node_count)
        self.num_cards = len(cards)
        self.card_tensors = tf.gather(tensor_array.concat(), card_indices)

    def call(self, x, training=None):
        def loop_cond(tensor_array, i):
            return tf.less(i, tf.squeeze(tf.shape(x)))

        def loop_body(tensor_array, i):
            original_row = tf.gather(x, i)
            row = tf.concat([tf.multiply(tf.gather(original_row, j),
                                         tf.gather(self.card_tensors, j))
                             for j in range(self.num_cards)], 0)
            tensor_array = tensor_array.write(row, i)
            i = tf.add(i, 1)
            return tensor_array, i

        tensor_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True,
                                      clear_after_read=False,
                                      infer_shape=False)
        tensor_array, _ = tf.while_loop(loop_cond, loop_body,
                                        [tensor_array, 0],
                                        parallel_iterations=1)
        x = tensor_array.concat()
        encoded = self.encoded_1(x)
        # encoded = self.e1_drop(encoded)
        encoded = self.encoded_2(encoded)
        # encoded = self.e2_drop(encoded)
        encoded = self.encoded_3(encoded)
        # encoded = self.e3_drop(encoded)
        return self.bottleneck(encoded)

    # def call_for_reg(self, x):
    #     encoded = self.encoded_1(x)
    #     encoded = self.encoded_2(encoded)
    #     encoded = self.encoded_3(encoded)
    #     return self.bottleneck(encoded)


class Decoder(Model):
    """
    Decoder part of the model -> expand from compressed latent
        space back to the input space
    """
    def __init__(self, name, output_dim, output_act):
        super().__init__()
        # self.bottleneck_drop = Dropout(0.2)
        self.decoded_1 = Dense(128, activation='relu', name=name + "_d1")
        # self.d1_drop = Dropout(0.4)
        self.decoded_2 = Dense(256, activation='relu', name=name + "_d2")
        # self.d2_drop = Dropout(0.4)
        self.decoded_3 = Dense(512, activation='relu', name=name + "_d3")
        # self.d3_drop = Dropout(0.2)
        self.reconstruct = Dense(output_dim, activation=output_act,
                                 name=name + "_reconstruction")

    def call(self, x, training=None):
        decoded = self.decoded_1(x)
        decoded = self.decoded_2(decoded)
        decoded = self.decoded_3(decoded)
        return self.reconstruct(decoded)

    # def call_for_reg(self, x):
    #     x = self.bottleneck_drop(x)
    #     decoded = self.decoded_1(x)
    #     decoded = self.d1_drop(decoded)
    #     decoded = self.decoded_2(decoded)
    #     decoded = self.d2_drop(decoded)
    #     decoded = self.decoded_3(decoded)
    #     decoded = self.d3_drop(decoded)
    #     return self.reconstruct(decoded)


class CC_Recommender(Model):
    """
    AutoEncoder build as a recommender system based on the following idea:

        If our input is a binary vector where 1 represents the presence of an
        item in a collection, then an autoencoder trained
    """
    def __init__(self, cards):
        super().__init__()
        self.N = len(cards)
        self.__preprocess_cards(cards)
        self.encoder = Encoder("encoder")
        # sigmoid because input is a binary vector we want to reproduce
        self.decoder = Decoder("main", self.N, output_act='sigmoid')
        # softmax because the graph information is probabilities
        # self.input_noise = Dropout(0.5)
        # self.latent_noise = Dropout(0.2)
        self.decoder_for_reg = Decoder("reg", self.N, output_act='softmax')

    def call(self, input, training=None):
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
        # x = self.input_noise(x)
        encoded = self.encoder(x)
        # latent_for_reconstruct = self.latent_noise(encoded)
        reconstruction = self.decoder(encoded)
        encode_for_reg = self.encoder(identity)
        # latent_for_reg = self.latent_noise(encode_for_reg)
        decoded_for_reg = self.decoder_for_reg(encode_for_reg)
        return reconstruction, decoded_for_reg
