import json
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import CosineSimilarity
import sys
import numpy as np

import non_ml.utils as utils

args = sys.argv[1:]
name = args[0].replace('_',' ')
N = int(args[1])
if len(args) > 2:
    model_name = args[2]
else:
    model_name = 'recommender'
folder = "././data/cube/"

int_to_card = json.load(open('ml_files/recommender_id_map.json','rb'))
int_to_card = {int(k): v for k, v in int_to_card.items()}
card_to_int = {v: k for k, v in int_to_card.items()}
num_cubes = utils.get_num_cubes(folder)
num_cards = len(int_to_card)

_, max_cube_size = utils.build_cubes(folder, num_cubes,
                                     num_cards, card_to_int)

num_cards = len(int_to_card)

model = load_model(f'ml_files/{model_name}')

cards = np.zeros((num_cards,max_cube_size))
for i in range(num_cards):
    cards[i, 0] = i + 1

dist_f = CosineSimilarity()

embs = model.encoder(cards)
idx = card_to_int[name]

dists = np.array([
    dist_f(embs[idx],x).numpy() for x in embs
])

ranked = dists.argsort()

for i in range(N):
    card_idx = ranked[i]
    print(str(i + 1) + ":", int_to_card[card_idx], dists[card_idx])