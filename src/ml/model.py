from typing import Dict, List, Tuple, Union

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
VOCAB_SIZE = 4

class Encoder(Model):
    """
    Encoder part of the model -> compress dimensionality
    """
    def __init__(self, name, cards, max_cube_size, batch_size):
        super().__init__()
        self.assigned_name = name
        self.max_cube_size = max_cube_size
        self.batch_size = batch_size
        self.__preprocess_cards(cards)
        self.flatten = tf.keras.layers.Flatten(name=name + "_flatten")
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
                              node_labels: List[int],
                              node_depths: List[int]) -> Tuple[int, int]:

            if isinstance(structure, list):
                if key in vocab_dict:
                    vocab = vocab_dict[key]
                else:
                    vocab = len(vocab_dict)
                    vocab_dict[key] = vocab
                our_children = []
                max_child_depth = 0
                for index, child in enumerate(structure):
                    child_index, depth = convert_structure(child, str(index),
                                                           vocab_dict, children,
                                                           node_labels, node_depths)
                    our_children.append(child_index)
                    max_child_depth = max(max_child_depth, depth)
                for index in range(len(structure), len(children)):
                    our_children.append(0)
                our_index = len(node_labels)
                for index, child_index in enumerate(our_children):
                    if len(children) <= index:
                        children.append([0 for _ in node_labels])
                    if len(children[index]) <= our_index:
                        children[index].append(0)
                    else:
                        children[index][our_index] = 0
                node_labels.append(vocab)
                node_depths.append(max_child_depth + 1)
                return our_index, max_child_depth + 1
            elif isinstance(structure, dict):
                our_children = []
                if key in vocab_dict:
                    vocab = vocab_dict[key]
                else:
                    vocab = len(vocab_dict)
                    vocab_dict[key] = vocab
                max_child_depth = 0
                for key, child in structure.items():
                    child_index, depth = convert_structure(child, key, vocab_dict,
                                                    children, node_labels, node_depths)
                    our_children.append(child_index)
                    max_child_depth = max(max_child_depth, depth)
                for _ in range(len(structure), len(children)):
                    our_children.append(-1)
                our_index = len(node_labels)
                for index, child_index in enumerate(our_children):
                    if len(children) <= index:
                        children.append([0 for _ in node_labels])
                    if len(children[index]) <= our_index:
                        children[index].append(0)
                    else:
                        children[index][our_index] = 0
                node_labels.append(vocab)
                node_depths.append(max_child_depth + 1)
                return our_index, max_child_depth + 1
            else:
                key = f'{key}.{structure}'
                if key in vocab_dict:
                    vocab = vocab_dict[key]
                else:
                    vocab = len(vocab_dict)
                    vocab_dict[key] = vocab
                our_index = len(node_labels)
                node_labels.append(vocab)
                node_depths.append(0)
                for index in range(len(children)):
                    if len(children[index]) <= our_index:
                        children[index].append(0)
                    else:
                        children[index][our_index] = 0
                return our_index, 0

        vocab_dict = {}
        children = []
        node_labels = [0]
        card_indices = []
        node_depths = [-1]
        for card in cards:
            card_index, _ = convert_structure(card, "", vocab_dict, children,
                                              node_labels, node_depths)
            card_indices.append(card_index)
        children_count = len(children)
        node_count = len(node_labels)
        print(len(vocab_dict), node_count, children_count)
        children = tf.constant([[child[i] for child in children]
                                for i in range(len(node_labels))])
        node_labels = tf.constant(node_labels)
        card_indices = [0] + card_indices
        self.embedding = tf.keras.layers.Embedding(len(vocab_dict), VOCAB_SIZE, input_length=1,
                                                   name=self.assigned_name + "_vocab_embedding")
        self.activation = tf.keras.layers.Activation('relu',
                                                     name=self.assigned_name + '_activation_recursive')
        self.concat = tf.keras.layers.Concatenate(1, name=self.assigned_name + "_concat_children")
        self.add = tf.keras.layers.Add(name=self.assigned_name + '_add_recursive')
        self.W = tf.Variable(name=self.assigned_name + "_W",
                             initial_value=tf.keras.initializers.glorot_uniform()(shape=(VOCAB_SIZE, children_count * VOCAB_SIZE)))

        node_depths = tf.constant(node_depths)
        max_depth = max(node_depths)
        tensor_array = tf.TensorArray(tf.float32, size=node_count, clear_after_read=False,
                                      infer_shape=True)

        def loop_cond(_, i):
            return tf.less(i, max_depth + 1)

        def loop_body(tensor_array, i):
            idxs = tf.where(tf.math.equal(node_depths, i))

            def inner_loop_cond(_, j):
                return tf.less(j, tf.gather(tf.shape(idxs), 0))

            def inner_loop_body(tensor_array, j):
                idx = tf.squeeze(tf.gather(idxs, j))
                node_label = tf.gather(node_labels, idx)
                our_children = tf.gather(children, idx)
                child_array = tf.concat([tensor_array.read(tf.squeeze(tf.gather(our_children, k)))
                                         for k in range(children_count)], 1)
                vocab = tf.reshape(self.embedding(node_label), [1, VOCAB_SIZE])
                node_tensor = self.activation(self.add([vocab,
                                                        tf.linalg.matvec(self.W, child_array)]))
                tensor_array = tensor_array.write(idx, node_tensor)
                return tensor_array, j + 1

            tensor_array, _ = tf.while_loop(inner_loop_cond, inner_loop_body,
                                            loop_vars=[tensor_array, 0], parallel_iterations=10000)

            return tensor_array, i + 1

        tensor_array = tensor_array.write(0, tf.zeros((1, VOCAB_SIZE)))
        tensor_array, _ = tf.while_loop(loop_cond, loop_body, loop_vars=[tensor_array, 0],
                                        parallel_iterations=1)
        self.num_cards = len(cards)
        self.card_tensors = tf.concat([tensor_array.read(i) for i in card_indices], 0)
        print("finished preprocessing cards.")
        
    def call(self, x, training=None):
        print("called encoder.")
        x = tf.nn.embedding_lookup(self.card_tensors, x)
        print(x.shape)
        x = self.flatten(x)
        print(x.shape)
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
    def __init__(self, cards, max_cube_size, batch_size):
        super().__init__()
        self.N = len(cards)
        self.encoder = Encoder("encoder", cards, max_cube_size, batch_size)
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
