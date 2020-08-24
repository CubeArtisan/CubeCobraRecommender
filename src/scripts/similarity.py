import json
import tensorflow as tf
from tensorflow.keras.losses import CosineSimilarity
import sys
import numpy as np

if __name__ == "__main__":
    from os.path import dirname as dir
    sys.path.append(dir(sys.path[0]))

from ml.model import CardEncoderWrapper
import non_ml.utils as utils
import ml.ml_utils as ml_utils

args = sys.argv[1:]
name = args[0].replace('_',' ')
N = int(args[1])
if len(args) > 2:
    model_name = args[2]
else:
    model_name = 'card_encoder'
folder = "././data/cube/"

int_to_card = json.load(open('././output/int_to_card.json', 'rb'))
int_to_card = {int(k): v for k, v in int_to_card.items()}
card_to_int = {v: k for k, v in int_to_card.items()}
num_cubes = utils.get_num_cubes(folder)
num_cards = len(int_to_card)

with open('cards.json', 'r', encoding="utf-8") as cardsjson:
    cards = json.load(cardsjson)
cards = [cards.get(int_to_card[i], "") for i in range(num_cards)]
for card in cards:
    if "otherParses" in card:
        del card["otherParses"]
all_paths, vocab_dict = ml_utils.generate_paths(cards, return_vocab_count=True)
our_paths = []
for a in all_paths:
    a += [[0 for _ in range(ml_utils.MAX_PATH_LENGTH)]
          for _ in range(len(a), ml_utils.NUM_INPUT_PATHS)]
    np.random.shuffle(a)
    a = a[:ml_utils.NUM_INPUT_PATHS]
    our_paths.append(a)
print('loading model')
card_model = CardEncoderWrapper(vocab_dict, ml_utils.NUM_INPUT_PATHS, ml_utils.MAX_PATH_LENGTH)
latest = tf.train.latest_checkpoint('ml_files/')
card_model.load_weights(latest)
print('Looking up embeddings')
STRIDE = 512
embeddings = []
for i in range(1, len(all_paths), STRIDE):
    cur_paths = tf.constant(our_paths[i:i + STRIDE])
    card_embeddings = card_model.card_encoder(cur_paths)
    embeddings.append(card_embeddings)
card_embeddings = tf.concat(embeddings, 0)
# _, max_cube_size = utils.build_cubes(folder, num_cubes,
#                                      num_cards, card_to_int)
#
# num_cards = len(int_to_card)
#
# model = load_model(f'ml_files/{model_name}')
#
# cards = np.zeros((num_cards, max_cube_size))
# for i in range(num_cards):
#     cards[i, 0] = i + 1
# embs = model.encoder(cards)

dist_f = CosineSimilarity()

idx = card_to_int[name]
print('calculating similiarities')
dists = np.array([
    np.absolute(dist_f(card_embeddings[idx], x).numpy()) for x in card_embeddings
])
ranked = dists.argsort()

for i in range(N):
    card_idx = ranked[i]
    print(str(i + 1) + ":", int_to_card[card_idx], dists[card_idx])