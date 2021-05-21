import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from tqdm.auto import tqdm

import src.non_ml.utils as utils


def is_valid_cube(cube):
    return cube['numDecks'] > 0 and len(set(cube['cards'])) >= 120


def is_valid_deck(deck):
    return len(deck['side']) > 0


def build_sparse_adj(objs, num_rows, num_cols):
    lookup = {}
    for i, source in enumerate(tqdm(list(objs), unit='obj', dynamic_ncols=True, unit_scale=True)):
        for target in source:
            key = (i, target)
            lookup[key] = lookup.get(key, 0) + 1
    lookup = list(zip(*lookup.items()))
    values = np.int32(list(lookup[1]))
    rows = np.int32([key[0] for key in lookup[0]])
    cols = np.int32([key[1] for key in lookup[0]])
    mat = sp.csr_matrix((values, (rows, cols)), shape=(num_rows, num_cols)).astype(np.float32)
    mat.sort_indices()
    mat.eliminate_zeros()
    mat.prune()
    return mat


def toggle_trans(edge):
    TRANS_SUFFIX = '_trans'
    if edge.endswith(TRANS_SUFFIX):
        return edge[:-len(TRANS_SUFFIX)]
    else:
        return f'{edge}{TRANS_SUFFIX}'


def print_matrix(matrix, name):
    print(f'{name}: shape({matrix.shape}), nonzero({matrix.nnz:n}), type({type(matrix)})')


def compute_pathname_adj(path_name, remaining_path, adjs):
    if path_name not in adjs and len(path_name) == 1 and (toggle_trans(path_name[0]),) in adjs:
        adjs[path_name] = adjs[(toggle_trans(path_name[0]),)].transpose()
    if path_name in adjs:
        prefix_adj = adjs[path_name]
        if len(remaining_path) <= 0:
            adj = prefix_adj
        else:
            remaining_adj = compute_metapath_adj(remaining_path, adjs)
            adj = prefix_adj @ remaining_adj
            adj.sort_indices()
            adj.prune()
        return adj
    else:
        return None

def compute_metapath_adj(path, adjs):
    trans_path = tuple(toggle_trans(edge) for edge in path[::-1])
    for i in range(len(path)):
        regular = compute_pathname_adj(tuple(path[:len(path)-i]), tuple(path[len(path)-i:]), adjs)
        if regular is not None:
            adjs[path] = regular
            return regular
        else:
            transposed = compute_pathname_adj(tuple(trans_path[:len(path)-i]), tuple(trans_path[len(path)-i:]), adjs)
            if transposed is not None:
                adj = transposed.transpose()
                adjs[path] = adj
                return adj
    print('no prefix or suffix found')


SEED_PATHS = (
    # ('in_main_trans', 'in_main'),
    # ('in_main_trans', 'in_side'),
    # ('in_main_trans', 'in_pool'),
    # ('in_main_trans', 'from_cube'),
    # ('in_side_trans', 'in_main'),
    # ('in_side_trans', 'in_side'),
    # ('in_side_trans', 'in_pool'),
    # ('in_side_trans', 'from_cube'),
    # ('in_pool_trans', 'in_main'),
    # ('in_pool_trans', 'in_side'),
    # ('in_pool_trans', 'in_pool'),
    # ('in_pool_trans', 'from_cube'),
)
# CARD_TO_CUBE_PATHS = (
#     ('in_cube_trans',),

#     ('in_main_trans', 'in_main', 'in_cube_trans'),
#     ('in_main_trans', 'in_side', 'in_cube_trans'),
#     ('in_main_trans', 'from_cube'),

#     ('in_side_trans', 'in_main', 'in_cube_trans'),
#     ('in_side_trans', 'in_side', 'in_cube_trans'),
#     ('in_side_trans', 'from_cube'),
# )
# CARD_TO_DECK_PATHS = (
#     ('in_main_trans',),
#     ('in_side_trans',),
#     ('in_cube_trans', 'from_cube_trans'),
# )
CARD_TO_CARD_PATHS = (
    ('in_cube_trans', 'in_cube'),
    ('in_cube_trans', 'from_cube_trans', 'in_main'),
    # ('in_cube_trans', 'from_cube_trans', 'in_side'),

    ('in_main_trans', 'in_pool'),
    ('in_main_trans', 'in_main'),
    # ('in_main_trans', 'in_side'),
    # ('in_main_trans', 'from_cube', 'in_cube'),
    # ('in_main_trans', 'from_cube', 'from_cube_trans', 'in_main'),
    # ('in_main_trans', 'from_cube', 'from_cube_trans', 'in_side'),

    ('in_side_trans', 'in_pool'),
    ('in_side_trans', 'in_main'),
    # ('in_side_trans', 'in_side'),
    # ('in_side_trans', 'from_cube', 'in_cube'),
    ('in_side_trans', 'from_cube', 'from_cube_trans', 'in_main'),
    # ('in_side_trans', 'from_cube', 'from_cube_trans', 'in_side'),

    # ('in_pool_trans', 'in_main'),
    # ('in_pool_trans', 'in_side'),
    # ('in_pool_trans', 'from_cube', 'in_cube'),
    # ('in_pool_trans', 'from_cube', 'from_cube_trans', 'in_main'),
    # ('in_pool_trans', 'from_cube', 'from_cube_trans', 'in_side'),
    ('card_id',),
)


if __name__ == "__main__":
    import locale

    locale.setlocale(locale.LC_ALL, '')

    data = Path('data')
    maps = data / 'maps'
    int_to_card_filepath = maps / 'int_to_card.json'
    cards_dict_filepath = maps / 'carddict.json'
    cube_folder = data / "cubes"
    decks_folder = data / "decks"
    adjs_dir = data / 'adjs'

    print('Loading card data, cubes, and decks.')
    with open(int_to_card_filepath, 'rb') as int_to_card_file:
        int_to_card = json.load(int_to_card_file)
    card_to_int = {v: i for i, v in enumerate(int_to_card)}
    exclusions = utils.get_exclusions(card_to_int, cards_dict_filepath)
    num_cards = len(int_to_card)
    cubes, cube_ids = utils.build_sparse_cubes(cube_folder, is_valid_cube, exclusions)
    num_cubes = len(cubes)
    cube_id_to_index = {v: i for i, v in enumerate(cube_ids)}
    decks = utils.build_deck_with_sides(decks_folder, cube_id_to_index,
                                        is_valid_deck, exclusions)
    num_decks = len(decks)
    print(f'There are {num_decks} decks, {num_cubes} cubes, made of {num_cards} cards.')

    in_cube_adj = build_sparse_adj(cubes, num_cubes, num_cards)
    in_main_adj = build_sparse_adj((d['main'] for d in decks), num_decks, num_cards)
    in_side_adj = build_sparse_adj((d['side'] for d in decks), num_decks, num_cards)
    in_pool_adj = build_sparse_adj((d['main'] + d['side'] for d in decks), num_decks, num_cards)
    from_cube_adj = build_sparse_adj(((d['cube'],) for d in decks if d['cube'] is not None), num_decks, num_cubes)
    adjs = {
        ('in_cube',): in_cube_adj, # cube -> card (must be initial of cube <-> cube paths)
        ('in_main',): in_main_adj, # deck -> card
        ('in_side',): in_side_adj, # deck -> card
        ('in_pool',): in_pool_adj, # deck -> card
        ('from_cube',): from_cube_adj, # deck -> cube
        ('card_id',): sp.identity(num_cards),
    }

    print('Getting metapath adjs')
    paths = SEED_PATHS + CARD_TO_CARD_PATHS
    for path in tqdm(paths, unit='path', dynamic_ncols=True):
        compute_metapath_adj(path, adjs)
    save_paths = CARD_TO_CARD_PATHS
    print('Saving metapath adjs')
    adjs_dir.mkdir(exist_ok=True, parents=True)
    total_nnz = 0
    for path in tqdm(save_paths, dynamic_ncols=True, unit='matrix'):
        filename = adjs_dir / '-'.join(path)
        adj = adjs[path] * (1024 / adjs[path].max())
        adj.eliminate_zeros()
        print(f'With {adj.nnz:09,} elements we have a density of {adj.nnz/adj.shape[0]/adj.shape[1]:06.2%} for path {path}.')
        total_nnz += adj.nnz
        sp.save_npz(filename, adj)
        # np.save(filename, adj.toarray())
    print(f'We have a total of {total_nnz:010,} elements for a density of {total_nnz/num_cards/num_cards/len(save_paths):06.2%}.')
