import glob
import heapq
import json
import logging
import sys
from collections import Counter
from pathlib import Path

logger = logging.getLogger('parse_picks')

import numpy as np

MAX_IN_PACK = 20
MAX_SEEN = 400
MAX_PICKED = 48
NUM_LAND_COMBS = 8
FEATURES = (
    (np.int32, (MAX_IN_PACK,), 'in_packs'), # int32 since you can't index with smaller or unsigned
    (np.int32, (MAX_SEEN,), 'seens'), # int32 same as for in_packs
    (np.float16, (), 'seen_counts'), # float16
    (np.int32, (MAX_PICKED,), 'pickeds'), # int32 same as for in_packs
    (np.float16, (), 'picked_counts'), # float16
    (np.int32, (4, 2), 'coords'),
    (np.float16, (4,), 'coord_weights'),
    (np.float16, (NUM_LAND_COMBS, MAX_SEEN), 'prob_seens'), # float16
    (np.float16, (NUM_LAND_COMBS, MAX_PICKED), 'prob_pickeds'), # float16
    (np.float16, (NUM_LAND_COMBS, MAX_IN_PACK), 'prob_in_packs'), # float16
)
RAGGED_FEATURES = (
    (np.int32, (None,), 'in_packs'), # int32 since you can't index with smaller or unsigned
    (np.int32, (None,), 'seens'), # int32 same as for in_packs
    (np.float16, (), 'seen_counts'), # float16
    (np.int32, (None,), 'pickeds'), # int32 same as for in_packs
    (np.float16, (), 'picked_counts'), # float16
    (np.int32, (4, 2), 'coords'),
    (np.float16, (4,), 'coord_weights'),
    (np.float16, (NUM_LAND_COMBS, None), 'prob_seens'), # float16
    (np.float16, (NUM_LAND_COMBS, None), 'prob_pickeds'), # float16
    (np.float16, (NUM_LAND_COMBS, None), 'prob_in_packs'), # float16
)
NUM_TRAIN_SHARDS = 1024
NUM_TEST_SHARDS = 256
pick_cache_dir = Path('data/parsed_picks/')
FETCH_LANDS = {
    "arid mesa": ['r', 'w'],
    "bad river": ['u', 'b'],
    "bloodstained mire": ['b', 'r'],
    "flood plain": ['w', 'u'],
    "flooded strand": ['w', 'u'],
    "grasslands": ['g', 'w'],
    "marsh flats": ['w', 'b'],
    "misty rainforest": ['g', 'u'],
    "mountain valley": ['r', 'g'],
    "polluted delta": ['u', 'b'],
    "rocky tar pit": ['b', 'r'],
    "scalding tarn": ['u', 'r'],
    "windswept heath": ['g', 'w'],
    "verdant catacombs": ['b', 'g'],
    "wooded foothills": ['r', 'g'],
    "prismatic vista": ['w', 'u', 'b', 'r', 'g'],
    "fabled passage": ['w', 'u', 'b', 'r', 'g'],
    "terramorphic expanse": ['w', 'u', 'b', 'r', 'g'],
    "evolving wilds": ['w', 'u', 'b', 'r', 'g'],
}
COLOR_COMBINATIONS = [frozenset(list(c)) for c in
                      ['', 'w', 'u', 'b', 'r', 'g', 'wu', 'ub', 'br', 'rg', 'gw', 'wb', 'ur',
                       'bg', 'rw', 'gu', 'gwu', 'wub', 'ubr', 'brg', 'rgw', 'rwb', 'gur',
                       'wbg', 'urw', 'bgu', 'ubrg', 'wbrg', 'wurg', 'wubg', 'wubr', 'wubrg']]
COLOR_COMB_INDEX = { s: i for i, s in enumerate(COLOR_COMBINATIONS) }
INTERSECTS_LIST = [frozenset([i for i, b in enumerate(COLOR_COMBINATIONS) if len(a & b) > 0]) for a in COLOR_COMBINATIONS]
INTERSECTS_LOOKUP = { s: l for s, l in zip(COLOR_COMBINATIONS, INTERSECTS_LIST) }
MAX_CMC = 8
MAX_REQUIRED_A = 6
MAX_REQUIRED_B = 3


def static_vars(**kwargs):
    def decorate(func):
        for k, v in kwargs.items():
            setattr(func, k, kwargs[k])
        return func
    return decorate


def load_card_data():
    cards_json = []
    with open('data/maps/int_to_card.json', 'r') as cards_file:
        cards_json = json.load(cards_file)
    card_colors = [None,]
    card_devotions = [-1]
    costs = {}
    costs_list = []

    for card in cards_json:
        type_line = card["type"].lower()
        colors = set([c.lower() for c in card.get("color_identity", [])])
        if "Land" in type_line:
            for t, c in (('plains', 'w'), ('island', 'u'), ('swamp', 'b'), ('mountain', 'r'), ('forest', 'g')):
                if t in type_line:
                    colors.add(c)
            fetch = FETCH_LANDS.get(card['name'], None)
            if fetch is not None:
                colors = fetch
            card_colors.append(COLOR_COMB_INDEX[frozenset(colors)])
        else:
            card_colors.append(None)
        parsed_cost = card.get('parsed_cost', [])
        devotions = {}
        total_devotion = 0
        for symbol in parsed_cost:
            symbol = symbol.lower()
            if 'p' in symbol or '2' in symbol:
                continue
            symbol_colors = ''.join([c for c in 'wubrg' if c in symbol])
            if len(symbol_colors) > 0:
                devotions[symbol_colors] = devotions.get(symbol_colors, 0) + 1
                total_devotion += 1
        cmc = min(max(card["cmc"], total_devotion), MAX_CMC)
        devotions = tuple((INTERSECTS_LOOKUP[frozenset(c)], v) for c, v in devotions.items())
        dev_len = len(devotions)
        if dev_len == 0:
            cost_index = -1
        else:
            if dev_len == 1:
                devotion_costs = (cmc, ((tuple(sorted(devotions[0][0])), min(devotions[0][1], MAX_REQUIRED_A)),))
            if dev_len == 2:
                colors_a, count_a = devotions[0]
                colors_b, count_b = devotions[1]
                if count_b > count_a:
                    colors_a, colors_b = colors_b, colors_a
                    count_a, count_b = count_b, count_a
                colors_ab = colors_a & colors_b
                colors_a, colors_b = colors_a - colors_b, colors_b - colors_a
                count_a, count_b = min(count_a, MAX_REQUIRED_A), min(count_b, MAX_REQUIRED_B)
                devotions_costs = (cmc, ((tuple(sorted(colors_a)), count_a),
                                         (tuple(sorted(colors_b)), count_b),
                                         tuple(sorted(colors_ab))))
            if dev_len > 2:
                mono_colors = devotions
                full_colors = set()
                full_count = 0
                for colors, count in devotions:
                    full_colors = full_colors | colors
                    full_count += count
                devotions_list = [*mono_colors, (full_colors, full_count)]
                devotions = []
                for colors, count in devotions_list:
                    colors = tuple(sorted(colors))
                    count = min(count, MAX_REQUIRED_A)
                    sub_cost = (cmc, ((colors, count),))
                    if sub_cost not in costs:
                        costs[sub_cost] = len(costs_list)
                        costs_list.append(sub_cost)
                    devotions.append((costs[sub_cost], *sub_cost))
                devotion_costs = (0, tuple(sorted(devotions)))

            if devotion_costs not in costs:
                costs[devotion_costs] = len(costs_list)
                costs_list.append(devotion_costs)
            cost_index = costs[devotion_costs]
        card_devotions.append(cost_index)
    logger.info(f"Populated all card data. There were {len(costs_list)} unique costs.")
    return card_devotions, card_colors, costs_list


def load_prob_table():
    prob_table = np.full((MAX_CMC + 1, MAX_REQUIRED_A + 1, MAX_REQUIRED_B + 1, 18, 18, 18), 0, dtype=np.float32)
    prob_table_json = []
    with open('data/maps/probTable.json', 'r') as prob_file:
        prob_table_json = json.load(prob_file)
    for str_cmc, nested1 in prob_table_json.items():
        for str_required_a, nested2 in nested1.items():
            for str_required_b, nested3 in nested2.items():
                for str_land_count_a, nested4 in nested3.items():
                    for str_land_count_b, nested5 in nested4.items():
                        for str_land_count_ab, prob in nested5.items():
                            prob_table[int(str_cmc)][int(str_required_a)][int(str_required_b)][int(str_land_count_a)][int(str_land_count_b)][int(str_land_count_ab)] = prob
    prob_table_json = None
    logger.info("Populated prob_table")
    return prob_table


def prob_to_cast_1(devotions, cmc, lands, lands_set, prob_table, prob_cache, lands_index, lands_list):
    colors = devotions[0][0]
    # colors = colors & lands_set
    usable_count = sum(lands[c] for c in colors)
    return prob_table[cmc, min(MAX_REQUIRED_A, devotions[0][1]), 0, usable_count, 0, 0]


def prob_to_cast_2(devotions, cmc, lands, lands_set, prob_table, prob_cache, lands_index, lands_list):
    colors_a, count_a = devotions[0]
    colors_b, count_b = devotions[1]
    colors_ab = devotions[2]
    # colors_a, colors_b, colors_ab = colors_a & lands_set, colors_b & lands_set, colors_ab & lands_set
    usable_count_a = sum(lands[c] for c in colors_a)
    usable_count_b = sum(lands[c] for c in colors_b)
    usable_count_ab = sum(lands[c] for c in colors_ab)
    return prob_table[cmc, min(MAX_REQUIRED_A, count_a), count_b, usable_count_a, usable_count_b, usable_count_ab]


def prob_to_cast_many(devotions, cmc, lands, lands_set, prob_table, prob_cache, lands_index, lands_list):
    return np.prod([prob_to_cast(ci, prob_cache, prob_to_cast_1, devotion, cmc, lands_index, prob_table, lands_list)
                    for ci, cmc, devotion in devotions])

PROB_TO_CAST_BY_COUNT = (None, prob_to_cast_1, None, prob_to_cast_2, prob_to_cast_many, prob_to_cast_many, prob_to_cast_many)


def prob_to_cast(cost_index, prob_cache, prob_func, devotions, cmc, lands_index, prob_table, lands_list):
    if cost_index not in prob_cache:
        lands, lands_set = lands_list[lands_index]
        prob_cache[cost_index] = prob_func(devotions, cmc, lands, lands_set, prob_table, prob_cache, lands_index, lands_list)
    return prob_cache[cost_index]


def lands_base(picked, card_colors):
    lands = np.full((32,), 0, dtype=np.uint8)
    remaining = 17
    for c in picked:
        colors = card_colors[c]
        if colors is not None and (colors > 5 or colors == 0):
            lands[colors] += 1
            remaining -= 1
            if remaining == 0:
                break
    return lands, remaining


def random_basics(n):
    indices = sorted(np.random.choice(n + 4, 4, replace=False))
    basics = [0 for _ in range(5)]
    prev = 0
    for i, x in enumerate(indices):
        basics[i] = x - prev
        prev = x + 1
    basics[4] = n + 4 - prev
    return basics


@static_vars(prob_cache=None, possible_lands={}, lands_list=[], prob_table=None)
def generate_probs(picked, seen, in_pack, costs_list, card_devotions, card_colors, total_climbs=NUM_LAND_COMBS):
    if generate_probs.prob_cache is None:
        generate_probs.prob_cache = {}
    if generate_probs.prob_table is None:
        generate_probs.prob_table = load_prob_table()
    base_lands, remaining = lands_base(picked, card_colors)
    heap = [(0, 0, 0, 0, ([0 for _ in seen], [0 for _ in picked], [0 for _ in in_pack]), -1) for _ in range(NUM_LAND_COMBS)]
    picked_costs, seen_costs, in_pack_costs = [[card_devotions[c] for c in cl] for cl in (picked, seen, in_pack)]
    costs = (picked_costs, seen_costs, in_pack_costs)
    picked_counter, seen_counter = [Counter(c for c in cs if c >= 0) for cs in (picked_costs, seen_costs)]
    in_pack_set = tuple(set(c for c in in_pack_costs if c >= 0))
    if len(in_pack_set) == 0:
        in_pack_set = (-1,)
    costs_to_calc = tuple((c, PROB_TO_CAST_BY_COUNT[len(costs_list[c][1])], *costs_list[c])
                          for c in frozenset(picked_costs + seen_costs + in_pack_costs) if c != -1)
    probs = { -1: 0 }
    if remaining <= 0:
        new_lands = tuple(base_lands)
        if new_lands not in generate_probs.possible_lands:
            generate_probs.possible_lands[new_lands] = len(generate_probs.lands_list)
            generate_probs.lands_list.append((new_lands, frozenset([i for i, x in enumerate(new_lands) if x > 0])))
        lands_index = generate_probs.possible_lands[new_lands]
        if lands_index not in generate_probs.prob_cache:
            generate_probs.prob_cache[lands_index] = {}
        prob_by_cost = generate_probs.prob_cache[lands_index]
        probs.update({ c: prob_to_cast(c, prob_by_cost, prob_func, devotions, cmc, lands_index, generate_probs.prob_table, generate_probs.lands_list)
                       for c, prob_func, cmc, devotions in costs_to_calc })
        picked_probs, seen_probs, in_pack_probs = [[probs[c] for c in cs] for cs in costs]
        heap[0] = (0, 0, 0, 0, (seen_probs, picked_probs, in_pack_probs), 0)
        total_climbs = 0
    visited = set()
    for loop in range(total_climbs):
        base_lands[1:6] = random_basics(remaining)
        prev_best, prev_lands = -1, tuple(base_lands)
        next_best, next_lands = 0, prev_lands
        while next_best > prev_best:
            prev_best, prev_lands = next_best, next_lands
            for add in range(1, 6):
                for cut in range(1, 6):
                    if add == cut or prev_lands[cut] == 0:
                        continue
                    new_lands = list(prev_lands)
                    new_lands[add] += 1
                    new_lands[cut] -= 1
                    new_lands = tuple(new_lands)
                    if new_lands not in generate_probs.possible_lands:
                        generate_probs.possible_lands[new_lands] = len(generate_probs.lands_list)
                        generate_probs.prob_cache[len(generate_probs.lands_list)] = {}
                        generate_probs.lands_list.append((new_lands, frozenset(i for i, x in enumerate(new_lands) if x > 0)))
                    lands_index = generate_probs.possible_lands[new_lands]
                    if lands_index in visited:
                        continue
                    visited.add(lands_index)
                    prob_by_cost = generate_probs.prob_cache[lands_index]
                    probs.update({ c: prob_to_cast(c, prob_by_cost, prob_func, devotions, cmc, lands_index, generate_probs.prob_table, generate_probs.lands_list)
                                   for c, prob_func, cmc, devotions in costs_to_calc })
                    picked_probs, seen_probs, in_pack_probs = [[probs[c] for c in cs] for cs in costs]
                    total_picked_prob, total_seen_prob = [sum(v * probs[c] for c, v in cs.items()) for cs in (picked_counter, seen_counter)]
                    max_in_pack_prob = max(probs[c] for c in in_pack_set)
                    total_score = total_picked_prob + max_in_pack_prob * 4 + total_seen_prob / 32
                    entry = (total_score, total_picked_prob, max_in_pack_prob, total_seen_prob,
                             (seen_probs, picked_probs, in_pack_probs), lands_index)
                    heapq.heappushpop(heap, entry)
                    if total_score > next_best:
                        next_add, next_cut, next_best, next_lands = add, cut, total_score, new_lands
                        break
                if next_best > prev_best:
                    break
    return [tuple(arr) for arr in zip(*[e[4] for e in heap])]


def to_one_hot(item, num_items):
    if item < 0:
        item = num_items
    result = [0 for _ in range(num_items)]
    result[item] = 1
    return result


def load_pick(pick, costs_list, card_devotions, card_colors, num_climbs=None):
    in_pack = np.int32([pick[3][0], *(c for c in set(pick[2]) if c != pick[3][0])])
    if len(in_pack) > MAX_IN_PACK:
        # logger.debug(f'Too many cards in pack({len(in_pack)}).')
        return None
    if len(in_pack) <= 1:
        return None
    seen, picked = [pick[k] for k in (0, 1)]
    seen_count = len(seen)
    picked_count = len(picked)
    pack_num, packs_count, pick_num, pack_size = pick[3][1], pick[3][2], pick[3][3], pick[3][4]
    pack_float, pick_float = 3 * pack_num / packs_count, 15 * pick_num / pack_size
    pack_0, pack_1, pack_frac = int(pack_float), min(2, int(pack_float) + 1), pack_float - int(pack_float)
    pick_0, pick_1, pick_frac = int(pick_float), min(14, int(pick_float) + 1), pick_float - int(pick_float)
    coords = ((pack_0, pick_0), (pack_0, pick_1), (pack_1, pick_0), (pack_1, pick_1))
    coord_weights = ((1 - pack_frac) * (1 - pick_frac), (1 - pack_frac) * pick_frac, pack_frac * (1 - pick_frac), pack_frac * pick_frac)
    prob_seen, prob_picked, prob_in_pack = generate_probs(picked, seen, in_pack, costs_list, card_devotions,
                                                          card_colors,num_climbs)
    return tuple(feature[0](value) for feature, value in zip(FEATURES, (in_pack, seen, seen_count,
                                                                        picked, picked_count, coords,
                                                                        coord_weights, prob_seen,
                                                                        prob_picked, prob_in_pack)))


def trim_pick(pick):
    if 'seen' not in pick or len(pick['seen']) > MAX_SEEN or None in pick['seen']:
        # logger.debug(f'Seen was invalid ({len(pick[0])}).')
        return None
    if 'picked' not in pick or len(pick['picked']) > MAX_PICKED or None in pick['picked']:
        # logger.debug(f'Picked was invalid ({len(pick[1])}).')
        return None
    if 'cardsInPack' not in pick or 'chosenCard' not in pick or None in pick['cardsInPack'] or pick['chosenCard'] not in pick['cardsInPack']:
        # logger.debug(f'Picked was invalid {pick[2]} with chosen card {pick[3]}.')
        return None
    return (
        np.int32(pick['seen']) + 1,
        np.int32(pick['picked']) + 1,
        np.int32(pick['cardsInPack']) + 1,
        np.int32([pick['chosenCard'] + 1, pick['pack'], pick['packs'], pick['pick'], pick['packSize']]),
    )


def load_pick_file(pick_file_name):
    with open(pick_file_name, 'r') as pf:
        pick_file_json = [tuple(x for x in (trim_pick(pick) for pick in draft['picks']) if x is not None) for draft in json.load(pf)]
    return pick_file_json


@static_vars(parsed=0, total_drafts=0, total_usable=0)
def parse_picks(pool, costs_list, card_devotions, card_colors, num_climbs=None, drafts_batch_size=16):
    def inner(pick_file_name):
        parsed_data = []
        draftCount = 0
        pick_file_json = []
        usable = 0
        pick_file_json = pool.apply_async(load_pick_file, (pick_file_name.decode('utf-8'),))
        pick_file_json.wait()
        pick_file_json = pick_file_json.get()
        print('parsed the json.')
        loading_data = []
        while len(pick_file_json) > 0 or len(loading_data) > 0:
            if len(loading_data) >= drafts_batch_size:
                loading_data[0].wait()
            elif len(pick_file_json) > 0:
                picks_list = pick_file_json.pop()
                loading_data.append(pool.starmap_async(load_pick, [(p, costs_list, card_devotions, card_colors, num_climbs) for p in picks_list], len(picks_list) + 1))
            loaded_data = []
            for i, loading in enumerate(loading_data):
                if loading.ready():
                    loaded_data = [pick for pick in loading.get() if pick is not None]
                    len_loaded = len(loaded_data)
                    if len_loaded > 0:
                        usable += len_loaded
                        draftCount += 1
                    yield from loaded_data
                    loading_data.pop(i)
                    break
        parse_picks.parsed += 1
        parse_picks.total_drafts += draftCount
        parse_picks.total_usable += usable
        parsed, total_drafts, total_usable = parse_picks.parsed, parse_picks.total_drafts, parse_picks.total_usable
        logger.info(f'Parsed #{parsed:05d} with {usable:07n} usable picks from {draftCount:06n} usable drafts for a total of {total_usable:010n} picks from {total_drafts:09n} drafts.')
    return inner


def picks_dataset(num_climbs=NUM_LAND_COMBS, proc_pool=None, num_workers=None, tf=None, cycle_length=8):
    card_devotions, card_colors, costs_list = load_card_data()
    return tf.data.Dataset.list_files('data/drafts/*.json', shuffle=True).interleave(
        lambda name: tf.data.Dataset.from_generator(parse_picks(proc_pool, costs_list, card_devotions, card_colors, num_climbs, num_workers), args=(name,),
                                                    output_signature=tuple(tf.TensorSpec(shape=shape, dtype=dtype) for dtype, shape, _ in RAGGED_FEATURES)
        ),
        cycle_length=cycle_length, num_parallel_calls=cycle_length, block_length=45, deterministic=False,
    ).shuffle(2**20).enumerate().prefetch(tf.data.AUTOTUNE)


def load_picks(cache_type, batch_size, num_workers=128):
    import tensorflow as tf
    default_target = np.zeros((batch_size, MAX_IN_PACK))
    default_target[:,0] = 1
    default_target = tf.constant(default_target, dtype=tf.float64)
    # default_target = np.float64([1] + [0 for _ in range(MAX_IN_PACK - 1)])
    directory = pick_cache_dir / cache_type
    return tf.data.experimental.load(
        # str(directory),
        'data/parsed_picks',
        (tf.TensorSpec(shape=(), dtype=tf.int64), tuple(tf.TensorSpec(shape=shape, dtype=dtype) for dtype, shape, _ in FEATURES)),
        compression='GZIP',
        reader_func=lambda ds:
            ds.shuffle(NUM_TRAIN_SHARDS)
              .interleave(lambda x: x,
                          cycle_length=128,
                          num_parallel_calls=128,
                          deterministic=False)
    ).shuffle(2**16).padded_batch(batch_size, ((), (
        (MAX_IN_PACK,),
        (MAX_SEEN,),
        (),
        (MAX_PICKED,),
        (),
        (4, 2),
        (4,),
        (NUM_LAND_COMBS, MAX_SEEN),
        (NUM_LAND_COMBS, MAX_PICKED),
        (NUM_LAND_COMBS, MAX_IN_PACK),
    )), drop_remainder=True).prefetch(2**21 // batch_size // num_workers)

def features_to_ragged(in_pack_card_indices, seen_indices, seen_counts,
                       picked_card_indices, picked_counts, coords, coord_weights,
                       prob_seen_matrices, prob_picked_matrices, prob_in_pack_matrices):
    in_pack_mask = in_pack_card_indices > 0
    in_pack_card_indices = tf.boolean_mask(in_pack_card_indices, in_pack_mask)
    prob_in_pack_matrices = tf.boolean_mask(prob_in_pack_matrices, tf.expand_dims(in_pack_mask, 0))
    target = tf.boolean_mask(np.float64([1] + [0 for _ in range(MAX_IN_PACK - 1)]), in_pack_mask)
    seen_mask = seen_indices > 0
    seen_indices = tf.boolean_mask(seen_indices, seen_mask)
    prob_seen_matrices = tf.boolean_mask(prob_seen_matrices, tf.expand_dims(seen_mask, 0))
    picked_mask = picked_card_indices > 0
    picked_card_indices = tf.boolean_mask(picked_card_indices, picked_mask)
    prob_picked_matrices = tf.boolean_mask(prob_picked_matrices, tf.expand_dims(picked_mask, 0))
    return (
        (
            in_pack_card_indices, seen_indices, seen_counts, picked_card_indices, picked_counts, coords, coord_weights,
            prob_seen_matrices, prob_picked_matrices, prob_in_pack_matrices
        ),
        target,
    )

if __name__ == '__main__':
    import multiprocessing
    logger = multiprocessing.get_logger()
    full_cache_dir = pick_cache_dir / 'full_compressed'
    formatter = logging.Formatter('{asctime} [{levelname}] {message}', style='{')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    num_workers = int(sys.argv[1])
    cycle_length = int(sys.argv[2])

    full_dataset_save = 'gs://cubecobratesting/full_drafts_dataset'

    with multiprocessing.Pool(num_workers) as proc_pool:
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(64)
        tf.config.threading.set_inter_op_parallelism_threads(64)
        full_dataset = picks_dataset(proc_pool=proc_pool, num_workers=num_workers, tf=tf)
        tf.data.experimental.save(full_dataset, full_dataset_save,
                                  compression='GZIP', shard_func=lambda x, _: x % (NUM_TRAIN_SHARDS + NUM_TEST_SHARDS))

    full_dataset = tf.data.experimental.load(
        str(pick_cache_dir / 'full'),
        compression='GZIP',
        reader_func=lambda ds: ds.interleave(lambda x: x, deterministic=True, num_parallel_calls=num_workers, cycle_length=num_workers)
    )
    num_picks = full_dataset.cardinality().numpy()
    test_dataset = full_dataset.skip(int(num_picks * 0.8)).prefetch(tf.data.AUTOTUNE)
    train_dataset = full_dataset.take(int(num_pikcs * 0.8)).prefetch(tf.data.AUTOTUNE)

    tf.data.experimental.save(train_dataset, str(pick_cache_dir / 'train'),
                              compression='GZIP', shard_func=lambda x, _: x % NUM_TRAIN_SHARDS)
    tf.data.experimental.save(test_dataset, str(pick_cache_dir / 'test'),
                              compression='GZIP', shard_func=lambda x, _: x % NUM_TEST_SHARDS)

#     full_ragged_dataset = full_dataset.map(lambda i, y: (i, features_to_ragged(*y)))
#     full_ragged_element_type = full_ragged_dataset.element_spec
#     test_ragged_dataset = full_ragged_dataset.skip(NUM_TRAIN_PICKS).prefetch(tf.data.AUTOTUNE)
#     train_ragged_dataset = full_ragged_dataset.take(NUM_TRAIN_PICKS).prefetch(tf.data.AUTOTUNE)
#     tf.data.experimental.save(full_ragged_dataset.prefetch(tf.data.AUTOTUNE), str(pick_cache_dir / 'full_ragged'),
#                               compression='GZIP', shard_func=lambda x, _: x % ((NUM_TRAIN_SHARDS + NUM_TEST_SHARDS) // 2))
#     tf.data.experimental.save(train_ragged_dataset, str(pick_cache_dir / 'train_ragged'),
#                               compression='GZIP', shard_func=lambda x, _: x % (NUM_TRAIN_SHARDS // 2))
#     tf.data.experimental.save(test_ragged_dataset, str(pick_cache_dir / 'test_ragged'),
#                               compression='GZIP', shard_func=lambda x, _: x % (NUM_TEST_SHARDS // 2))
