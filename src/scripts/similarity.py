import json
import tensorflow as tf
from tensorflow.keras.losses import CosineSimilarity
import sys
import numpy as np

if __name__ == "__main__":
    from os.path import dirname as dir
    sys.path.append(dir(sys.path[0]))

from ml.model import CardEncoder
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
    if 'typeLine' in card:
        if isinstance(card['typeLine'], str):
            card['type'] = card['typeLine'].replace('— ', '').split(' ')
        else:
            for key, value in card['typeLine'][0].items():
                if isinstance(value, dict):
                    value = value['and']
                else:
                    value = [value]
                card[key] = value
        del card['typeLine']
    if "otherParses" in card:
        del card["otherParses"]
all_paths, features, feature_count, vocab_dict = ml_utils.generate_paths(cards)
print('shortening paths')
our_paths = []
for a in all_paths:
    np.random.shuffle(a)
    a = a[:ml_utils.NUM_INPUT_PATHS]
    our_paths.append(a)
print('loading model')
card_model = CardEncoder("card_encoder", vocab_dict, ml_utils.NUM_INPUT_PATHS, ml_utils.MAX_PATH_LENGTH, 1, feature_count)
latest = tf.train.latest_checkpoint('ml_files/card_encoder_code2seq_contrastive')
# latest = tf.train.latest_checkpoint('ml_files/card_encoder-code2seq-features')
card_model.load_weights(latest)
print('Looking up embeddings')
STRIDE = 512
embeddings = []
for i in range(1, len(all_paths), STRIDE):
    cur_paths = tf.constant(our_paths[i:i + STRIDE])
    cur_features = tf.constant(features[i:i + STRIDE])
    card_embeddings = card_model([cur_paths, cur_features])
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


idx = card_to_int[name]
print('calculating similiarities')
# dist_f = CosineSimilarity()
dists = np.array([
    tf.reshape(tf.keras.layers.dot([card_embeddings[idx], x], 1, normalize=True), ()).numpy() for x in card_embeddings
])
ranked = dists.argsort()[::-1]
# card = tf.tile(tf.reshape(card_embeddings[idx], (1, 256)), [len(all_paths) - 1, 1])
# dists = tf.norm(tf.subtract(card, card_embeddings), axis=1).numpy()
# dists = np.array([tf.norm(tf.subtract(card_embeddings[idx], x)) for x in card_embeddings])
# ranked = dists.argsort()

for i in range(N):
    card_idx = ranked[i]
    print(str(i + 1) + ":", int_to_card[card_idx], dists[card_idx])