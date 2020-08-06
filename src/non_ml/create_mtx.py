import utils
import numpy as np
import json
import sys
from pathlib import Path
import os

map_file = '././data/maps/nameToId.json'
folder = "././data/deck/"
require_side = True
print('getting data')
num_cards = 0
num_decks = 0
card_to_int = dict()
for f in os.listdir(folder):
    full_path = os.path.join(folder, f)
    with open(full_path, 'rb') as fp:
        contents = json.load(fp)
    for cube in contents:
        card_ids = []
        if require_side and len(cube['side']) == 0:
            continue
        num_decks += 1
        for card_name in cube['main']:
            if card_name is not None:
                if card_name not in card_to_int:
                    card_to_int[card_name] = num_cards
                    num_cards += 1
print(f'num decks: {num_decks}')

int_to_card = {v:k for k,v in card_to_int.items()}

print(f'num cubes: {num_cubes}')
cubes = utils.build_decks(folder, num_decks, num_cards,
                          card_to_int, require_side=require_side)

print('creating matrix')
adj_mtx = utils.create_adjacency_matrix(cubes)

Path(f'././output').mkdir(parents=True, exist_ok=True)
with open('././output/full_adj_mtx.npy', 'wb') as out_mtx:
    np.save(out_mtx, adj_mtx)

with open('././output/int_to_card.json', 'w') as out_lookup:
    json.dump(int_to_card,  out_lookup)
