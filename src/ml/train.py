# enable sibling imports
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    from os.path import dirname as dir
    sys.path.append(dir(sys.path[0]))

from generator import DataGenerator
from model import CC_Recommender
from non_ml import utils
<<<<<<< HEAD
import json
=======

>>>>>>> 514d3b46b0b628911b2c7574bab516fa0b835287

def reset_random_seeds(seed):
    # currently not used
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


args = sys.argv[1:]

epochs = int(args[0])
batch_size = int(args[1])
name = args[2]
reg = float(args[3])
noise = float(args[4])

if len(args) == 6:
    seed = int(args[5])
    reset_random_seeds(seed)

map_file = '././data/maps/nameToId.json'
folder = "././data/cubes/"

print('Loading Cube Data . . .\n')

int_to_card = json.load(open('././output/int_to_card.json','rb'))
int_to_card = {int(k):v for k,v in int_to_card.items()}
card_to_int = {v:k for k,v in int_to_card.items()}

num_cubes = utils.get_num_cubes(folder)
num_cards = len(int_to_card)

with open('cards.json', 'r', encoding="utf-8") as cardsjson:
    cards = json.load(cardsjson)
    cards = [cards.get(int_to_card[i], "") for i in range(num_cards)]
    for card in cards:
        if "otherParses" in card:
            del card["otherParses"]

cubes = utils.build_cubes(folder, num_cubes, num_cards,
                          card_to_int)

print('Loading Adjacency Matrix . . .\n')

adj_mtx = np.load('././output/full_adj_mtx.npy')

# print('Converting Graph Weights to Probabilities . . . \n')
print('Creating Graph for Regularization . . . \n')

# make easier to learn by dropping super low conditional probabilities
# too_small = np.where(adj_mtx < thresh)
# y_mtx = adj_mtx.copy()
# y_mtx[too_small] = 0
# np.fill_diagonal(y_mtx,1)
# y_mtx = (adj_mtx/adj_mtx.sum(1)[:,None])
# y_mtx = np.nan_to_num(y_mtx,0)
# y_mtx[np.where(y_mtx.sum(1) == 0),np.where(y_mtx.sum(1) == 0)] = 1

# y_mtx = adj_mtx.copy()
# np.fill_diagonal(y_mtx, 1)

print('Setting Up Data for Training . . .\n')


# x_items = np.zeros(adj_mtx.shape)
# np.fill_diagonal(x_items,1)

print('Setting Up Model . . . \n')

autoencoder = CC_Recommender(cards)
autoencoder.compile(
    optimizer='adam',
    loss=['binary_crossentropy', 'kullback_leibler_divergence'],
    loss_weights=[1.0, reg],
    metrics=['accuracy'],
)

generator = DataGenerator(
    y_mtx,
    cubes,
    batch_size=batch_size,
    noise=noise,
)

# pdb.set_trace()

autoencoder.fit(
    generator,
    epochs=epochs,
)

output_dir = f'././ml_files/{name}'
Path(output_dir).mkdir(parents=True, exist_ok=True)
autoencoder.save(output_dir, save_format='tf')
