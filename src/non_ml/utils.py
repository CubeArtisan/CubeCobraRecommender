import json
import os

import numpy as np
from tqdm.auto import tqdm

BAD_NAMES = [
    'plains',
    'island',
    'swamp',
    'mountain',
    'forest',
    'snow-covered plains',
    'snow-covered island',
    'snow-covered swamp',
    'snow-covered mountain',
    'snow-covered forest',
    'invalid card',
]
BAD_FUNCTIONS = [
    lambda x: x.get('isToken'),
]


def get_exclusions(card_to_int, card_file=None):
    exclusions = list(BAD_NAMES)
    if card_file is None:
        return frozenset(exclusions)
    with open(card_file, 'rb') as cf:
        card_dict = json.load(cf)
    for cd in card_dict.values():
        for bf in BAD_FUNCTIONS:
            if cd.get('name_lower') not in exclusions and bf(cd):
                card_id = card_to_int.get(cd.get('name_lower', '-1'), None)
                if card_id is not None:
                    exclusions.append(card_id)
    return frozenset(exclusions)


def get_card_maps(map_file, exclude_file=None):
    exclusions = exclude(exclude_file)
    with open(map_file, 'rb') as mf:
        names = json.load(mf)
    name_lookup = dict()
    card_to_int = dict()
    num_cards = 0
    for name, ids in names.items():
        if name in exclusions:
            continue
        card_to_int[name] = num_cards
        for idx in ids:
            name_lookup[idx] = name
        num_cards += 1
    int_to_card = {v: k for k, v in card_to_int.items()}
    return (
        num_cards,
        name_lookup,
        card_to_int,
        int_to_card
    )

def get_num_objs(cube_folder, validation_func=lambda _: True):
    num_objs = 0
    for filename in cube_folder.iterdir():
        with open(filename, 'rb') as obj_file:
            contents = json.load(obj_file)
        num_objs += len([obj for obj in contents if validation_func(obj)])
    return num_objs


def build_cubes(cube_folder, num_cubes, num_cards,
                validation_func=lambda _: True, exclusions=frozenset()):
    cubes = np.zeros((num_cubes, num_cards))
    counter = 0
    with tqdm(total=num_cubes, unit='cube', dynamic_ncols=True, unit_scale=True) as tqdm_bar:
        for filename in cube_folder.iterdir():
            with open(filename, 'rb') as cube_file:
                contents = json.load(cube_file)
            for cube in contents:
                if validation_func(cube):
                    card_ids = []
                    for card_id in cube['cards']:
                        if card_id is not None and card_id not in exclusions:
                            card_ids.append(card_id)
                    cubes[counter, card_ids] = 1
                    counter += 1
                    tqdm_bar.update(1)
    return cubes


def build_sparse_cubes(cube_folder, validation_func=lambda _: True, exclusions=frozenset()):
    cubes = []
    cube_ids = []
    num_cubes = get_num_objs(cube_folder, validation_func)
    with tqdm(total=num_cubes, unit='cube', dynamic_ncols=True, unit_scale=True) as tqdm_bar:
        for filename in tqdm(list(cube_folder.iterdir()), dynamic_ncols=True, unit='file'):
            with open(filename, 'rb') as cube_file:
                contents = json.load(cube_file)
            for cube in contents:
                if validation_func(cube):
                    card_ids = []
                    for card_id in cube['cards']:
                        if card_id is not None and card_id not in exclusions:
                            card_ids.append(card_id)
                    if len(card_ids) > 0:
                        cubes.append(card_ids)
                        cube_ids.append(cube['id'])
                    tqdm_bar.update(1)
    return cubes, cube_ids


def build_decks(deck_folder, num_decks, num_cards,
                validation_func=lambda _: True,
                soft_validation=0, exclusions=frozenset()):
    decks = np.zeros((num_decks, num_cards), dtype=np.uint8)
    counter = 0
    with tqdm(total=num_decks, unit='deck', dynamic_ncols=True, unit_scale=True) as tqdm_bar:
        for filename in tqdm(list(deck_folder.iterdir()), dynamic_ncols=True, unit='file'):
            with open(filename, 'rb') as deck_file:
                contents = json.load(deck_file)
            for deck in contents:
                if soft_validation > 0 or validation_func(deck):
                    card_ids = []
                    for card_id in deck['main']:
                        if card_id is not None and card_id not in exclusions:
                            card_ids.append(card_id)
                    weight = 1
                    if not validation_func(deck):
                        weight = soft_validation
                    decks[counter, card_ids] = weight
                    counter += 1
                    tqdm_bar.update(1)
    return decks


def build_deck_with_sides(deck_folder, cube_id_to_index, validation_func=lambda _: True,
                          exclusions=frozenset()):
    decks = []
    counter = 0
    num_decks = get_num_objs(deck_folder, validation_func)
    with tqdm(total=num_decks, unit='deck', dynamic_ncols=True, unit_scale=True) as tqdm_bar:
        for filename in tqdm(list(deck_folder.iterdir()), dynamic_ncols=True, unit='file'):
            with open(filename, 'rb') as deck_file:
                contents = json.load(deck_file)
            for deck in contents:
                if validation_func(deck):
                    main = []
                    side = []
                    for card_id in deck['main']:
                        if card_id is not None and card_id not in exclusions:
                            main.append(card_id)
                    for card_id in deck['side']:
                        if card_id is not None and card_id not in exclusions:
                            side.append(card_id)
                    decks.append({'main': main, 'side': side, 'cube': cube_id_to_index.get(deck['cubeid'], None)})
                    tqdm_bar.update(1)
    return decks


def build_mtx(deck_folder, num_cards,
              validation_func=lambda _: True,
              soft_validation=0):
    adj_mtx = np.zeros((num_cards, num_cards), dtype=np.uint32)
    counter = 0
    for filename in tqdm(list(deck_folder.iterdir()), unit='file', dynamic_ncols=True):
        with open(filename, 'rb') as deck_file:
            contents = json.load(deck_file)
        for deck in tqdm(contents, unit='deck', unit_scale=True, dynamic_ncols=True, initial=counter, leave=None):
            if soft_validation > 0 or validation_func(deck):
                card_ids = []
                for card_id in deck['main']:
                    if card_id is not None:
                        card_ids.append(card_id)
                weight = 1 if validation_func(deck) else 0
                if not validation_func(deck):
                    weight = soft_validation
                for i, id1 in enumerate(card_ids):
                    adj_mtx[id1, id1] += weight
                    for id2 in card_ids[:i]:
                        adj_mtx[id1, id2] += weight
                        adj_mtx[id2, id1] += weight
                counter += 1
    return adj_mtx


def create_adjacency_matrix(decks, verbose=True, force_diag=None):
    num_cards = decks.shape[1]
    adj_mtx = np.empty((num_cards, num_cards))
    for i in range(num_cards):
        if verbose:
            if i % 100 == 0:
                print(i+1, "/", num_cards)
        idxs = np.where(decks[:, i] > 0)
        decks_w_cards = np.float64(decks[idxs])
        step1 = decks_w_cards.sum(0)  # (num_cards,)
        if step1[i] != 0:
            step1 = step1/step1[i]
        adj_mtx[i] = step1
    if force_diag is not None:
        np.fill_diagonal(adj_mtx, force_diag)
    return adj_mtx
