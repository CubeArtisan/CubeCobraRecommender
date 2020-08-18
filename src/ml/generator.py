from bisect import bisect_left
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Union

from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import skipgrams
import numpy as np
import random

WINDOW_SIZE = 4
MAX_PATH_LENGTH = 16
NUM_INPUT_PATHS = 16
NUM_PATHS = 512
NUM_EXAMPLES = 2**21


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
            x_regularization[i, 0] = x
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
            random.shuffle(actual_cube)
            for idx in cube:
                if idx > 0 and idx not in y_flip_include:
                    y_cubes[i, idx - 1] = 1
            processed_cubes.append(actual_cube)

        x_cubes = np.array(processed_cubes).astype(int)
        return [(x_cubes, x_regularization), (y_cubes, y_regularization)]


def get_cumulative_dist(dist):
    acc = 0
    result = []
    for prob in dist:
        acc += prob
        result.append(acc)
    return result


def sample_index_from_cumulative(cum_dist):
    value = np.random.rand()
    return bisect_left(cum_dist, value)


def convert_structure(structure: Union[list, dict, int, str, bool],
                      key: str, vocab_dict: Dict[str, int],
                      children: List[List[int]],
                      node_labels: List[int],
                      node_heights: List[int],
                      node_depths: List[int],
                      depth: int) -> Tuple[int, int]:
    our_children = []
    max_child_height = 0
    if isinstance(structure, list):
        for index, child in enumerate(structure):
            child_index, height = convert_structure(child, str(index),
                                                    vocab_dict, children,
                                                    node_labels, node_heights,
                                                    node_depths, depth + 1)
            our_children.append(child_index)
            max_child_height = max(max_child_height, height)
    elif isinstance(structure, dict):
        for key, child in structure.items():
            child_index, height = convert_structure(child, key, vocab_dict,
                                                    children, node_labels, node_heights,
                                                    node_depths, depth + 1)
            our_children.append(child_index)
            max_child_height = max(max_child_height, height)
    else:
        key = f'{key}.{structure}'
        if key in vocab_dict:
            vocab = vocab_dict[key]
        else:
            vocab = len(vocab_dict)
            vocab_dict[key] = vocab
        our_index = len(node_labels)
        node_labels.append(vocab)
        node_heights.append(0)
        for index in range(len(children)):
            children[index].append(0)
        node_depths.append(depth)
        return our_index, 0
    if key in vocab_dict:
        vocab = vocab_dict[key]
    else:
        vocab = len(vocab_dict)
        vocab_dict[key] = vocab
    for _ in range(len(structure), len(children)):
        our_children.append(-1)
    our_index = len(node_labels)
    for index, child_index in enumerate(our_children):
        if len(children) <= index:
            children.append([0 for _ in node_labels])
        children[index].append(0)
    node_labels.append(vocab)
    node_heights.append(max_child_height + 1)
    node_depths.append(depth)
    return our_index, max_child_height + 1


class CardDataGenerator(Sequence):
    def __init__(self, adj_mtx, walk_len, card_counts, cards, batch_size, data_path=None):
        super(CardDataGenerator).__init__()
        self.num_cards = adj_mtx.shape[0]
        self.batch_size = batch_size
        self.walk_len = walk_len
        self.num_walks = 1
        self.max_paths = NUM_INPUT_PATHS
        self.max_path_length = MAX_PATH_LENGTH
        card_counts = np.concatenate(([0], card_counts))
        card_counts = card_counts / card_counts.sum()
        self.card_counts = get_cumulative_dist(card_counts)
        y_mtx = adj_mtx.copy()
        np.fill_diagonal(y_mtx, 0)
        y_mtx = (y_mtx / y_mtx.sum(1)[:, None])
        self.y_probs = [[] for _ in range(self.num_cards + 1)]
        self.y_indices = [[] for _ in range(self.num_cards + 1)]
        print('Building adjacency graph structure.')
        non_zero = np.where(y_mtx > 0)
        for i, j in zip(*non_zero):
            self.y_probs[i + 1].append(y_mtx[i][j])
            self.y_indices[i + 1].append(j + 1)
        self.y_probs = [get_cumulative_dist(probs) for probs in self.y_probs]
        print('Converting cards to trees.')
        vocab_dict = {"<unkown>": 0}
        children = []
        node_labels = [0]
        self.card_indices = [0]
        node_heights = [-1]
        node_depths = [-1]
        for card in cards:
            card_index, _ = convert_structure(card, "", vocab_dict, children,
                                              node_labels, node_heights, node_depths, 0)
            self.card_indices.append(card_index)
        self.node_labels = node_labels
        self.node_heights = node_heights
        self.node_depths = node_depths
        children_count = len(children)
        self.node_count = len(node_labels)
        children = [[child[i] for child in children]
                    for i in range(self.node_count)]
        self.node_parents = [0 for _ in range(self.node_count)]
        for i, our_children in enumerate(children):
            for child in our_children:
                if child != 0:
                    self.node_parents[child] = i
        self.vocab_count = len(vocab_dict)
        print(len(vocab_dict), self.node_count, children_count)
        self.all_paths = self.generate_paths()

    def generate_data(self):
        print('Calculating walks.')
        positive_examples = [Counter() for _ in range(self.num_cards + 1)]
        negative_examples = [Counter() for _ in range(self.num_cards + 1)]
        for _ in range(self.num_walks):
            examples = self.calculate_skipgrams()
            for positive, negative in examples:
                for i, j in positive:
                    positive_examples[i][j] += 1
                for i, j in negative:
                    negative_examples[i][j] += 1
        for positive, negative in zip(positive_examples, negative_examples):
            for key in list(positive.keys()):
                if key in negative:
                    if positive[key] > negative[key]:
                        del negative[key]
                    else:
                        del positive[key]
        for i, positive in enumerate(positive_examples):
            for j in list(positive.keys()):
                del positive_examples[j][i]
                if i in negative_examples[j]:
                    if positive[j] > negative_examples[j][i]:
                        del negative_examples[j][i]
                    else:
                        del positive_examples[i][j]
        for i, negative in enumerate(negative_examples):
            for j in list(negative.keys()):
                del negative_examples[j][i]
        return [(i, j, 1)
                for i, positive in enumerate(positive_examples)
                for j in positive.keys()] + [(i, j, 0)
                                             for i, negative in enumerate(negative_examples)
                                             for j in negative.keys()]

    def calculate_skipgrams(self):
        for i in range(1, self.num_cards + 1):
            walk = [i]
            cur_node = i
            for _ in range(self.walk_len):
                sampling_table = self.y_probs[cur_node]
                nodes = self.y_indices[cur_node]
                node_index = sample_index_from_cumulative(sampling_table)
                cur_node = nodes[node_index]
                walk.append(cur_node)
            couples, labels = skipgrams(walk, self.num_cards + 1,
                                        negative_samples=0.0,
                                        window_size=WINDOW_SIZE)
            positive = []
            negative = []
            for couple, label in zip(couples, labels):
                i, j = couple
                i = int(i)
                j = int(j)
                if label == 1:
                    positive.append((i, j))
                    positive.append((j, i))
                    neg_i = sample_index_from_cumulative(self.card_counts)
                    negative.append((i, neg_i))
                    negative.append((neg_i, i))
            yield positive, negative

    def generate_paths(self):
        print('Calculating AST paths')
        all_paths = [[]]
        for i, max_index in enumerate(self.card_indices):
            if i == 0:
                continue
            min_index = self.card_indices[i - 1] + 1
            paths = []
            index_range = [x for x in range(min_index, max_index) if self.node_heights[x] == 0]
            computed_values = defaultdict(lambda: defaultdict(lambda: False))
            used_indices = set()
            iterations = 0
            if len(index_range) == 0:
                all_paths.append([])
                continue
            while len(paths) < NUM_PATHS and iterations < NUM_PATHS * 2:
                iterations += 1
                new_index_range = [x for x in index_range if x not in used_indices]
                if len(new_index_range) > 0:
                    start = random.choice(new_index_range)
                else:
                    start = random.choice(index_range)
                remaining = [x for x in index_range if x != start and not computed_values[start][x]]

                if len(remaining) == 0:
                    continue
                new_remaining = [x for x in remaining if x not in used_indices]
                if len(new_remaining) > 0:
                    end = random.choice(new_remaining)
                else:
                    end = random.choice(remaining)
                used_indices.add(start)
                used_indices.add(end)
                computed_values[start][end] = True
                computed_values[end][start] = True
                start_depth = self.node_depths[start]
                end_depth = self.node_depths[end]
                if end_depth > start_depth:
                    start, end = end, start
                    start_depth, end_depth = end_depth, start_depth
                path = [self.node_labels[start]]
                end_path = [self.node_labels[end]]
                while start_depth > end_depth:
                    start = self.node_parents[start]
                    start_depth -= 1
                    path.append(start)
                while self.node_parents[start] != self.node_parents[end]:
                    start = self.node_parents[start]
                    end = self.node_parents[end]
                    path.append(start)
                    end_path.append(end)
                path.append(self.node_parents[start])
                path += end_path[::-1]
                if len(path) > MAX_PATH_LENGTH:
                    continue
                paths.append(path)
            all_paths.append(paths)
        for node_paths in all_paths:
            for path in node_paths:
                for _ in range(len(path), MAX_PATH_LENGTH):
                    path.append(0)
        return all_paths
            
    def __len__(self):
        self.data = self.generate_data()
        np.random.shuffle(self.data)
        self.data = self.data[:NUM_EXAMPLES]
        return len(self.data) // self.batch_size

    def __getitem__(self, item):
        a_s = []
        b_s = []
        y_s = []
        for i in range(self.batch_size):
            a, b, y = self.data[self.batch_size * item + i]
            a = self.all_paths[a]
            np.random.shuffle(a)
            a = a[:NUM_INPUT_PATHS]
            a += [[0 for _ in range(MAX_PATH_LENGTH)] for _ in range(len(a), NUM_INPUT_PATHS)]
            b = self.all_paths[b]
            np.random.shuffle(b)
            b = b[:NUM_INPUT_PATHS]
            b += [[0 for _ in range(MAX_PATH_LENGTH)] for _ in range(len(b), NUM_INPUT_PATHS)]
            a_s.append(a)
            b_s.append(b)
            y_s.append(y)
            
        a_s = np.array(a_s).reshape((self.batch_size, 1, NUM_INPUT_PATHS, MAX_PATH_LENGTH))
        b_s = np.array(b_s).reshape((self.batch_size, 1, NUM_INPUT_PATHS, MAX_PATH_LENGTH))
        x_s = np.concatenate((a_s, b_s), 1)
        y_s = np.array(y_s).reshape((self.batch_size, 1))
        return x_s, y_s
