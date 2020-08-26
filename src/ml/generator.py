from bisect import bisect_left
from collections import Counter

import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import skipgrams

from ml.ml_utils import generate_paths, MAX_PATH_LENGTH, NUM_INPUT_PATHS

WINDOW_SIZE = 4
NUM_EXAMPLES = 2**16
NUM_START_POS = 10000
IDENTITY_EXAMPLES_DIVISOR = 10


class DataGenerator(Sequence):

    def __init__(
        self,
        adj_mtx,
        cubes,
        num_cards,
        batch_size=64,
        shuffle=True,
        to_fit=True,
        noise=0.2,
        noise_std=0.1,
    ):
        self.noise_std = noise_std
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.to_fit = to_fit
        self.noise = noise
        # initialize inputs and outputs
        self.y_reg = adj_mtx
        self.x_reg = np.zeros_like(adj_mtx)
        np.fill_diagonal(self.x_reg, 1)
        self.x_main = cubes
        self.max_cube_size = len(cubes[0])
        # initialize other needed inputs
        self.N_cubes = len(cubes)
        self.N_cards = num_cards
        self.neg_sampler = adj_mtx.sum(0)/adj_mtx.sum()
        self.cube_includes = [[x for x in cube if x > 0] for cube in cubes]
        self.cube_includes_set = [set(includes) for includes in self.cube_includes]
        self.cube_excludes = [[x - 1 for x in range(1, self.N_cards + 1) if x not in includes_set]
                              for includes_set in self.cube_includes_set]
        del self.cube_includes_set
        self.neg_samplers = [self.neg_sampler[excludes]
                             / self.neg_sampler[excludes].sum()
                             for excludes in self.cube_excludes]
        # self.pool = Pool(30)
        self.indices = np.arange(self.N_cubes)
        self.batches = []
        self.reset_indices()

    def __len__(self):
        """
        return: number of batches per epoch
        """
        self.on_epoch_end()
        return self.N_cubes // self.batch_size

    def __getitem(self, batch_number):
        """
        Generates a data mini-batch
        param batch_number: which batch to generate
        return: X and y when fitting. X only when predicting
        """
        main_indices = self.indices[
            batch_number * self.batch_size:(batch_number + 1) * self.batch_size
        ]
        reg_indices = np.random.choice(
            np.arange(self.N_cards),
            len(main_indices),
            p=self.neg_sampler,
            replace=False,
        )

        X, y = self.generate_data(
            main_indices,
            reg_indices,
        )
        if self.to_fit:
            return [X[0], X[1]], [y[0], y[1]]
        else:
            return [X[0], X[1]]

    def __getitem__(self, batch_number):
        return self.__getitem(batch_number)
        # return self.batches[batch_number].get()

    def reset_indices(self):
        self.indices = np.arange(self.N_cubes)
        if self.shuffle:
            np.random.shuffle(self.indices)
        # print('starting to calculate batches')
        # self.batches = [self.pool.apply_async(self.__getitem, (i,))
        #                 for i in range(len(self))]
        # print('started calculating batches')

    def on_epoch_end(self):
        """
        Update indices after each epoch
        """
        self.reset_indices()

    def generate_data(self, main_indices, reg_indices):
        cubes = [self.x_main[i] for i in main_indices]
        cube_includes = [self.cube_includes[i] for i in main_indices]
        cube_excludes = [self.cube_excludes[i] for i in main_indices]
        neg_samplers = [self.neg_samplers[i] for i in main_indices]
        x_regularization = np.zeros((len(reg_indices), self.max_cube_size)).astype(int)        
        for i, x in enumerate(reg_indices):
            x_regularization[i, np.random.randint(self.max_cube_size)] = x
        y_regularization = self.y_reg[reg_indices]
        y_cubes = np.zeros((len(main_indices), self.N_cards))

        processed_cubes = []
        for i, cube, includes, excludes, neg_sampler \
            in zip(range(len(cubes)), cubes, cube_includes, cube_excludes, neg_samplers):
            size = len(includes)
            noise = np.clip(
                np.random.normal(self.noise, self.noise_std),
                a_min=0.05,
                a_max=0.8,
            )
            flip_amount = int(size * noise)
            flip_include = np.random.choice(includes, flip_amount, replace=False)
            new_cube = Counter(cube)
            for to_remove in flip_include:
                new_cube[to_remove] -= 1
            flip_exclude = np.random.choice(excludes,
                                            flip_amount,
                                            p=neg_sampler,
                                            replace=False)
            for to_add in flip_exclude:
                new_cube[to_add + 1] += 1
            y_flip_include = np.random.choice(flip_include, flip_amount // 4)
            actual_cube = []
            for key, count in new_cube.items():
                actual_cube += [key for _ in range(count)]
            np.random.shuffle(actual_cube)
            for idx in cube:
                if idx > 0 and idx not in y_flip_include:
                    y_cubes[i, idx - 1] = 1
            processed_cubes.append(actual_cube)

        x_cubes = np.array(processed_cubes).astype(int)
        return [(x_cubes, x_regularization), (y_cubes, y_regularization)]


def get_cumulative_dist(dist):
    return np.cumsum(dist)


def sample_index_from_cumulative(cum_dist):
    value = np.random.rand()
    return bisect_left(cum_dist, value)


class CardDataGenerator(Sequence):
    def __init__(self, adj_mtx, walk_len, card_counts, cards, batch_size, example_count):
        super(CardDataGenerator).__init__()
        self.num_cards = adj_mtx.shape[0]
        self.batch_size = batch_size
        self.walk_len = walk_len
        self.num_walks = 1
        self.max_paths = NUM_INPUT_PATHS
        self.max_path_length = MAX_PATH_LENGTH
        self.example_count = example_count
        self.all_paths, self.card_features, self.feature_count, self.vocab_dict = generate_paths(cards)
        print(len(self.card_features[0]), self.feature_count)
        for features in self.card_features:
            if len(features) != self.feature_count:
                print("mismatched feature length")
        print('Building adjacency graph structure.')
        y_mtx = adj_mtx.copy()
        np.fill_diagonal(y_mtx, 0)
        y_mtx = y_mtx / y_mtx.sum(1)[:, None]
        self.y_probs = [[] for _ in range(self.num_cards + 1)]
        self.y_indices = [[] for _ in range(self.num_cards + 1)]
        non_zero = np.where(y_mtx > 0)
        for i, j in zip(*non_zero):
            self.y_probs[i + 1].append(y_mtx[i][j])
            self.y_indices[i + 1].append(j + 1)
        self.y_probs = [get_cumulative_dist(probs) for probs in self.y_probs]
        print('Building inverse adjacency graph structure')
        inv_y_mtx = (1 - y_mtx) * card_counts
        np.fill_diagonal(inv_y_mtx, 0)
        inv_y_mtx = inv_y_mtx / inv_y_mtx.sum(1)[:, None]
        non_zero = np.where(inv_y_mtx > 0)
        self.inv_y_probs = [[] for _ in range(self.num_cards + 1)]
        self.inv_y_indices = [[] for _ in range(self.num_cards + 1)]
        for i, j in zip(*non_zero):
            self.inv_y_probs[i + 1].append(inv_y_mtx[i][j])
            self.inv_y_indices[i + 1].append(j + 1)
        self.inv_y_probs = [get_cumulative_dist(probs) for probs in self.inv_y_probs]
        # self.inv_y_probs = self.y_probs
        # self.inv_y_indices = self.y_indices

    def generate_data(self):
        print('Calculating walks.')
        return list(self.calculate_skipgrams())

    def do_walk(self, start_node, walk_len, probs, indices, num_cards):
        walk = [start_node]
        cur_node = start_node
        for _ in range(walk_len):
            sampling_table = probs[cur_node]
            nodes = indices[cur_node]
            node_index = sample_index_from_cumulative(sampling_table)
            cur_node = nodes[node_index]
            walk.append(cur_node)
        couples, labels = skipgrams(walk, num_cards + 1, negative_samples=0.0,
                                    window_size=WINDOW_SIZE)
        for couple, label in zip(couples, labels):
            if label == 1:
                yield couple

    def calculate_skipgrams(self):
        for _ in range(NUM_START_POS):
            i = np.random.randint(1, self.num_cards + 1)
            positive = self.do_walk(i, self.walk_len, self.y_probs, self.y_indices, self.num_cards)
            for i, pos in enumerate(positive):
                j = pos[0]
                if i % IDENTITY_EXAMPLES_DIVISOR == 0:
                    pos = [j, j]
                for _ in range(self.example_count - 1):
                    sampling_table = self.inv_y_probs[j]
                    nodes = self.inv_y_indices[j]
                    node_index = sample_index_from_cumulative(sampling_table)
                    k = nodes[node_index]
                    pos.append(k)
                yield pos

    def __len__(self):
        self.data = self.generate_data()
        np.random.shuffle(self.data)
        self.data = self.data[:NUM_EXAMPLES]
        return len(self.data) // self.batch_size

    def retrieve_card_paths(self, index):
        paths = self.all_paths[index]
        np.random.shuffle(paths)
        return paths[:NUM_INPUT_PATHS]

    def __getitem__(self, item):
        paths = [[self.retrieve_card_paths(example)  # MAX_PATHS, MAX_PATH_LENGTH
                  for example in examples]  # EXAMPLE_COUNT, MAX_PATHS, MAX_PATH_LENGTH
                 for examples in self.data[self.batch_size * item: self.batch_size * (item + 1)]]  # BATCH_SIZE, EXAMPLE_COUNT, MAX_PATHS, MAX_PATH_LENGTH
        features = [[self.card_features[example]  # FEATURE_COUNT
                     for example in examples]  # EXAMPLE_COUNT, FEATURE_COUNT
                    for examples in self.data[self.batch_size * item: self.batch_size * (item + 1)]] # BATCH_SIZE, EXAMPLE_COUNT, FEATURE_COUNT
        paths = np.array(paths)
        features = np.array(features).reshape((self.batch_size, self.example_count + 1, self.feature_count, 1))
        y_s = np.zeros_like(features)
        return [paths, features], y_s