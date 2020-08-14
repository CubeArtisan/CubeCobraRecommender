import json
import numpy as np
import unidecode
from tensorflow.keras.models import load_model
import sys
import urllib.request

if __name__ == "__main__":
    from os.path import dirname as dir
    sys.path.append(dir(sys.path[0]))

import non_ml.utils as utils

args = sys.argv[1:]
cube_name = args[0]

non_json = True
root = "https://cubecobra.com"
if len(args) > 1:
    amount = int(args[1])
    if len(args) > 2:
        root = args[2]
        non_json = False
else:
    amount = 100
folder = "././data/cube/"

print('Getting Cube List . . . \n')

url = root + "/cube/api/cubelist/" + cube_name

fp = urllib.request.urlopen(url)
mybytes = fp.read()

mystr = mybytes.decode("utf8")
fp.close()

card_names = mystr.split("\n")

print('Loading Card Name Lookup . . . \n')

int_to_card = json.load(open('ml_files/recommender_id_map.json','rb'))
int_to_card = {int(k):v for k,v in int_to_card.items()}
card_to_int = {v:k for k,v in int_to_card.items()}
num_cubes = utils.get_num_cubes(folder)
num_cards = len(int_to_card)

_, max_cube_size = utils.build_cubes(folder, num_cubes,
                                     num_cards, card_to_int)
num_cards = len(int_to_card)

print('Creating Cube Vector . . . \n')

cube_indices = []
for name in card_names:
    idx = card_to_int.get(unidecode.unidecode(name.lower()))
    #skip unknown cards (e.g. custom cards)
    if idx is not None:
        cube_indices.append(idx + 1)
cube_indices += [0 for _ in range(len(cube_indices), max_cube_size)]
cube_indices = cube_indices[:max_cube_size]
cube = np.array(cube_indices)

print('Loading Model . . . \n')

model = load_model('ml_files/recommender-code2seq')

# def encode(model,data):
#     return model.encoder.bottleneck(
#         model.encoder.encoded_3(
#             model.encoder.encoded_2(
#                 model.encoder.encoded_1(
#                     data
#                 )
#             )
#         )
#     )

# def decode(model,data):
#     return model.decoder.reconstruct(
#         model.decoder.decoded_3(
#             model.decoder.decoded_2(
#                 model.decoder.decoded_1(
#                     data
#                 )
#             )
#         )
#     )

def recommend(model, data):
    encoded = model.encoder(data, training=False)
    return model.decoder(encoded, training=False)

print ('Generating Recommendations . . . \n')

cube = np.array(cube, dtype=int).reshape(1, max_cube_size)
results = recommend(model, cube)[0].numpy()

ranked = results.argsort()[::-1]

output = {
    'additions':dict(),
    'cuts':dict(),
}

recommended = 0
for rec in ranked:
    if rec + 1 not in cube_indices:
        card = int_to_card[rec]
        if non_json:
            print(card)
        else:
            output['additions'][card] = results[rec].item()
        recommended += 1
        if recommended >= amount:
            break

for idx in cube_indices:
    if idx > 0:
        card = int_to_card[idx - 1]
        output['cuts'][card] = results[idx - 1].item()

if non_json:
    cards = list(output['cuts'].keys())
    vals = list(output['cuts'].values())
    rank_cuts = np.array(vals).argsort()
    out = [cards[idx] for idx in rank_cuts[:amount]]
    print('\n')
    for i,item in enumerate(out): print(item,vals[rank_cuts[i]])
