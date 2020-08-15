import json
import numpy as np
import os
import utils
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

map_file = '././data/maps/nameToId.json'
folder = "././data/deck/"
require_side = False
print('getting data')
num_cards = 0
num_decks = 0
card_to_int = dict()
card_counts_dict: Dict[int, float] = defaultdict(lambda: 0)
total_count = 0.0
for f in os.listdir(folder):
    full_path = os.path.join(folder, f)
    with open(full_path, 'rb') as fp:
        contents = json.load(fp)
    for deck in contents:
        card_ids: List[int] = []
        weight = 1.0
        if len(deck['side']) == 0:
            weight = 1 / 3
            if require_side:
                continue
        total_count += weight
        num_decks += 1
        for card_name in deck['main']:
            if card_name is not None:
                if card_name not in card_to_int:
                    card_to_int[card_name] = num_cards
                    num_cards += 1
                card_counts_dict[card_to_int[card_name]] += 1
print(f'num decks: {num_decks}')
card_counts = np.array([card_counts_dict[i] for i in range(num_cards)])
card_counts = card_counts / total_count

int_to_card = {v: k for k, v in card_to_int.items()}

cubes = utils.build_decks(folder, num_decks, num_cards,
                          card_to_int, require_side=require_side)

print('creating matrix')
adj_mtx = utils.create_adjacency_matrix(cubes)

Path(f'././output').mkdir(parents=True, exist_ok=True)
with open('././output/int_to_card.json', 'w') as out_lookup:
    json.dump(int_to_card,  out_lookup)
with open('././output/card_counts.npy', 'wb') as out_counts:
    np.save(out_counts, card_counts)
with open('././output/full_adj_mtx.npy', 'wb') as out_mtx:
    np.save(out_mtx, adj_mtx)
