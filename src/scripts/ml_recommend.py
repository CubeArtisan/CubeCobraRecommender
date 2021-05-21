import argparse
import json
import sys
import unidecode
import urllib.request
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
# from tensorflow.keras.models import load_model
from tqdm.auto import tqdm

from src.ml.metapath import MetapathRecommender

DEFAULT_COUNT = 25
DEFAULT_MODEL = '20210409'
DEFAULT_JSON = False
DEFAULT_ROOT = 'https://cubecobra.com'
data = Path('data')

def recommend_changes(cube, count=DEFAULT_COUNT, model=DEFAULT_MODEL, use_json=DEFAULT_JSON, root=DEFAULT_ROOT,
                      embed_dims=64, metapath_dims=32, num_heads=8):
    print('Getting Cube List . . . \n')

    url = root + "/cube/api/cubelist/" + cube

    with urllib.request.urlopen(url) as request:
        mybytes = request.read()
    mystr = mybytes.decode("utf8")

    card_names = mystr.split("\n")
    model_dir = Path('ml_files') / model

    print ('Loading Card Name Lookup . . . \n')

    with open(model_dir / 'int_to_card.json', 'rb') as map_file:
        int_to_card = json.load(map_file)
    card_to_int = {v:k for k,v in enumerate(int_to_card)}

    num_cards = len(int_to_card)

    print ('Creating Cube Vector . . . \n')

    cube_indices = [card_to_int[unidecode.unidecode(name.lower())]
                    for name in card_names if unidecode.unidecode(name.lower()) in card_to_int]
    one_hot_cube = np.zeros(num_cards)
    one_hot_cube[cube_indices] = 1

    print(f'Loading Model {model_dir}. . . \n')

    # loaded_model = load_model(model_dir)
    recommender = MetapathRecommender(
        card_metapaths=load_metapaths(), embed_dims=embed_dims,
        metapath_dims=metapath_dims, num_heads=num_heads, name='MetapathRecommender'
    )
    latest = tf.train.latest_checkpoint(str(model_dir))
    if latest is not None:
        print('Loading Checkpoint. Saved values are:')
        recommender.load_weights(latest).expect_partial()

    print ('Generating Recommendations . . . \n')

    one_hot_cube = one_hot_cube.reshape(1, num_cards)
    results = recommender((one_hot_cube, one_hot_cube), training=False)[0][0].numpy()
    # results = loaded_model.decoder(loaded_model.encoder(one_hot_cube))[0].numpy()
    # recommender.save(model_dir, save_format='tf')

    ranked = results.argsort()[::-1]

    output = {
        'additions':dict(),
        'cuts':dict(),
    }
    output_str = ''

    cuts = []
    adds = []
    for i, rec in enumerate(ranked):
        card = int_to_card[rec]
        if one_hot_cube[0][rec] == 0 and len(adds) < count:
            adds.append((card, results[rec]))
        elif one_hot_cube[0][rec] == 1:
            cuts.append((card, results[rec]))
    cuts = cuts[-count:]
    if use_json:
        return json.dumps({
            'additions': {name: value.item() for name, value in adds},
            'cuts': {name: value.item() for name, value in cuts}
        })
    else:
        adds_str = '\n'.join(f'{name}: {value}' for name, value in adds)
        cuts_str = '\n'.join(f'{name}: {value}' for name, value in cuts)
        return f'{adds_str}\n...\n{cuts_str}'


def load_metapaths():
    print(f'Loading metapath adjacency matrices')
    return tuple(sp.load_npz(filename).tocoo() / 1024 for filename in tqdm(list((data / 'adjs').iterdir())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cube', '-c', help='The id or short name of the cube to recommend for.')
    parser.add_argument('--count', '--number', '-n', default=DEFAULT_COUNT, type=int, help='The number of recommended cuts and additions to recommend.')
    parser.add_argument('--model', '-m', default=DEFAULT_MODEL, help='The path under ml_files to the model to use for recommendations.')
    parser.add_argument('--json', dest='use_json', action='store_true', help='Output the results as json instead of plain text.')
    parser.add_argument('--root', default=DEFAULT_ROOT, help='The base url for the CubeCobra instance to retrieve the cube from.')
    parser.add_argument('--embed-dims', type=int, default=128, choices=[2**i for i in range(0, 12)], help='The number of dimensions for the card ebeddings.')
    parser.add_argument('--metapath-dims', type=int, default=64, choices=[2**i for i in range(0, 10)], help='The number of dimensions for the metapath specific views of the pool embeddings.')
    parser.add_argument('--num-heads', type=int, default=16, choices=[2**i for i in range(0, 9)], help='The number of attention heads to use for combining the metapaths.')
    args = parser.parse_args()

    print(recommend_changes(**args.__dict__))
