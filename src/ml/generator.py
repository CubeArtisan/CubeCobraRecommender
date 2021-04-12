import multiprocessing
import multiprocessing.dummy
import time
from pathlib import Path

import numpy as np


def load_adj_mtx():
    print('Loading Adjacency Matrix . . .\n')
    adj_mtx = np.load('data/adj_mtx.npy')
    np.fill_diagonal(adj_mtx, 1)
    y_mtx = adj_mtx / adj_mtx.sum(1)[:, None]
    return y_mtx

def gen_worker(task_queue, done_queue, cube_includes, cube_excludes, neg_sampler):
    for args in iter(task_queue.get, 'STOP'):
        done_queue.put(generate_data(*args, cube_includes, cube_excludes, neg_sampler))

def generate_data(top_level_indices, to_fit, noise, noise_std, cube_includes, cube_excludes, neg_sampler):
    cut_mask = []
    add_mask = []
    y_cut_mask = []
    cube_indices = []
    for i, cube_index in enumerate(top_level_indices):
        includes = cube_includes[cube_index]
        excludes = cube_excludes[cube_index]
        size = len(includes)
        if size == 0:
            continue
        noise = np.clip(
            np.random.normal(noise, noise_std),
            a_min=0.05,
            a_max=0.95,
        )
        flip_amount = min(max(int(size * noise), 1), size - 1)
        to_exclude = np.random.choice(includes, flip_amount, replace=False)
        if len(excludes) < flip_amount:
            to_include = excludes
        else:
            neg_sampler_exclude = neg_sampler[excludes] / neg_sampler[excludes].sum()
            to_include = np.random.choice(excludes, flip_amount, p=neg_sampler_exclude, replace=False)
            to_include = list(set(to_include))
        neg_sampler_y_include = neg_sampler[to_exclude]
        if neg_sampler_y_include.sum() == 0:
            print('neg_sampler_y_include sum:', neg_sampler_y_include.sum(), 'len:', len(neg_sampler_y_include), 'to_exclude len:', len(to_exclude), 'includes len:', len(includes))
            neg_sampler_y_include = np.ones_like(neg_sampler_y_include) / flip_amount
        neg_sampler_y_include = neg_sampler_y_include / neg_sampler_y_include.sum()
        y_to_exclude = np.random.choice(to_exclude, flip_amount // 4, p=neg_sampler_y_include, replace=False)

        cube_indices.append(includes)
        cut_mask.append(to_exclude)
        y_cut_mask.append(y_to_exclude)
        add_mask.append(to_include)
    if to_fit:
        # reg_indices = np.random.choice(
        #     len(neg_sampler),
        #     len(top_level_indices),
        #     p=neg_sampler,
        #     replace=False,
        # )
        reg_indices = np.random.choice(
            len(neg_sampler),
            len(top_level_indices),
            replace=False,
        )
        return (cube_indices, cut_mask, y_cut_mask, add_mask), reg_indices
    else:
        return cube_indices, cut_mask, y_cut_mask, add_mask

def create_sequence(*args, **kwargs):
    from tensorflow.keras.utils import Sequence
    class DataGenerator(Sequence):
        def __init__(
            self,
            num_cubes,
            num_cards,
            task_queue,
            batch_queue,
            processes,
            batch_size=64,
            shuffle=True,
            to_fit=True,
            noise=0.2,
            noise_std=0.1,
        ):
            self.task_queue = task_queue
            self.batch_queue = batch_queue
            self.noise_std = noise_std
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.to_fit = to_fit
            self.noise = noise
            self.N_cubes = num_cubes
            self.N_cards = num_cards
            self.y_reg = load_adj_mtx()
            # Initialize other needed inputs
            self.indices = np.arange(self.N_cubes)
            self.processes = processes
            self.reset_indices()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            print('Exiting the generator')
            for _ in self.processes:
                self.task_queue.put('STOP')
            time.sleep(1)
            for process in self.processes:
                process.terminate()
            return False

        def __len__(self):
            """
            return: number of batches per epoch
            """
            return self.N_cubes // self.batch_size

        def __getitem__(self, batch_number):
            """
            Generates a data mini-batch
            param batch_number: which batch to generate
            return: X and y when fitting. X only when predicting
            """
            if self.task_queue.empty() and self.batch_queue.qsize() < len(self):
                 self.reset_indices()
            batched_data, reg_indices = self.batch_queue.get(True, 5)
            cube_x = np.zeros((self.batch_size, self.N_cards))
            cube_y = np.zeros((self.batch_size, self.N_cards))
            for i, data in enumerate(zip(*batched_data)):
                includes, cut_mask, y_cut_mask, add_mask = data
                cube_x[i, includes] = 1
                cube_x[i, cut_mask] = 0
                cube_x[i, add_mask] = 1
                cube_y[i, includes] = 1
                cube_y[i, y_cut_mask] = 0
            bad_indices_x = np.where((cube_x != 0) & (cube_x != 1))
            bad_indices_y = np.where((cube_y != 0) & (cube_y != 1))
            if len(bad_indices_x[0]) > 0:
                print('Bad x', cube_x[bad_indices_x])
            if len(bad_indices_y[0]) > 0:
                print('Bad y', cube_y[bad_indices_y])
            cards_x = np.zeros((self.batch_size, self.N_cards))
            cards_x[list(range(len(reg_indices))), reg_indices] = 1
            cards_y = self.y_reg[reg_indices]
            return (cube_x, cards_x), (cube_y, cards_x, cards_y)


        def reset_indices(self):
            if self.shuffle:
                np.random.shuffle(self.indices)
            for i in range(self.N_cubes // self.batch_size):
                indices = self.indices[i * self.batch_size:(i + 1) * self.batch_size]
                self.task_queue.put((
                    indices,
                    self.to_fit,
                    self.noise,
                    self.noise_std
                ))

        def on_epoch_end(self):
            """
            Update indices after each epoch
            """
            # self.reset_indices()
    return DataGenerator(*args, **kwargs)


def create_dataset(adj_mtx, cubes, batch_size, noise=0.2, noise_std=0.1):
    import tensorflow as tf
    logits = tf.math.log(neg_sampler)
    y_mtx = tf.constant(adj_mtx)
    num_cards = len(adj_mtx)

    def apply_noise(cube):
        includes = tf.where(cube == 1)
        excludes = tf.where(cube == 0)
        size = len(includes.shape)
        cube_noise = tf.clip_by_value(
            tf.random.normal((), noise, noise_std),
            0.05,
            0.95,
        )
        flip_amount = tf.cast(size * cube_noise, dtype=tf.int32)
        excluded_probs = tf.gather(neg_sampler, excludes)
        excluded_neg_sampler = excluded_probs / tf.reduce_sum(excluded_probs, keepdims=True)
        included_probs = tf.ones_like(includes)
        included_neg_sampler = included_probs / tf.reduce_sum(included_probs, keepdims=True)
        # This will have to be close enough since there's not a great way to sample without replacement in tf
        flip_exclude = tf.reshape(tf.random.categorical(tf.math.log(excluded_neg_sampler), flip_amount), (-1,))
        flip_include = tf.reshape(tf.random.categorical(tf.math.log(included_neg_sampler), flip_amount), (-1,))
        return tf.tensor_scatter_nd_add(cube, tf.concat([flip_exclude, flip_include], 0),
                                        tf.concat([tf.ones_like(flip_exclude, dtype=tf.float64),
                                                   -tf.ones_like(flip_include, dtype=tf.float64)], 0)), cube

    card_indices_dataset = tf.data.Dataset.range(batch_size).map(lambda _: tf.reshape(tf.random.categorical([logits], 1), (1,)))
    cubes_dataset = tf.data.Dataset.from_tensor_slices(cubes).map(apply_noise, num_parallel_calls=32)
    return tf.data.Dataset.zip((card_indices_dataset, cubes_dataset)).batch(batch_size)\
             .map(lambda cards, cube_data: ((cube_data[0], tf.one_hot(cards, num_cards), (cube_data[1], tf.gather(y_mtx, cards)))))\
             .prefetch(tf.data.AUTOTUNE)
