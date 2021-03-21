import glob
import heapq
import json
import logging
import multiprocessing
import sys
from collections import Counter
from pathlib import Path

import numpy as np

MAX_IN_PACK = 16
MAX_SEEN = 360
MAX_PICKED = 48
NUM_LAND_COMBS = 32
FEATURES = (
    (np.int32, (MAX_IN_PACK,), 'in_packs'),
    (np.int32, (MAX_SEEN,), 'seens'),
    (np.float32, (), 'seen_counts'),
    (np.int32, (MAX_PICKED,), 'pickeds'),
    (np.float32, (), 'picked_counts'),
    (np.int32, (4, 2), 'coords'),
    (np.float32, (4,), 'coord_weights'),
    (np.uint8, (NUM_LAND_COMBS, MAX_SEEN), 'prob_seens'),
    (np.uint8, (NUM_LAND_COMBS, MAX_PICKED), 'prob_pickeds'),
    (np.uint8, (NUM_LAND_COMBS, MAX_IN_PACK), 'prob_in_packs'),
)
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
MAX_REQUIRED_A = 6

logger = multiprocessing.get_logger()


def static_vars(**kwargs):
    def decorate(func):
        for k, v in kwargs.items():
            setattr(func, k, kwargs[k])
        return func
    return decorate


def load_card_data():
    cards_json = []
    with open('data/intToCard.json', 'r') as cards_file:
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
        cmc = min(max(card["cmc"], total_devotion), 8)
        devotion_costs = (cmc, frozenset([(INTERSECTS_LOOKUP[frozenset(c)], v) for c, v in devotions.items()]))
        if devotion_costs not in costs:
            costs[devotion_costs] = len(costs)
            costs_list.append(devotion_costs)
        cost_index = costs[devotion_costs]
        card_devotions.append(cost_index)
    logger.info("Populated all card data")
    return card_devotions, card_colors, costs_list


def load_prob_table():
    prob_table = np.full((9, 7, 4, 18, 18, 18), 0, dtype=np.float32)
    prob_table_json = []
    with open('data/probTable.json', 'r') as prob_file:
        prob_table_json = json.load(prob_file)
    for str_cmc, nested1 in prob_table_json.items():
        for str_required_a, nested2 in nested1.items():
            for str_required_b, nested3 in nested2.items():
                for str_land_count_a, nested4 in nested3.items():
                    for str_land_count_b, nested5 in nested4.items():
                        for str_land_count_ab, prob in nested5.items():
                            prob_table[int(str_cmc)][int(str_required_a)][int(str_required_b)][int(str_land_count_a)][int(str_land_count_b)][int(str_land_count_ab)] = prob * 255
    prob_table_json = None
    logger.info("Populated prob_table")
    return prob_table


def prob_to_cast_0(*args):
    return 1


def prob_to_cast_1(devotions, cmc, lands, lands_set, prob_table):
    colors = devotions[0][0] & lands_set
    usable_count = sum(lands[c] for c in colors)
    return prob_table[cmc, min(MAX_REQUIRED_A, devotions[0][1]), 0, usable_count, 0, 0]


def prob_to_cast_2(devotions, cmc, lands, lands_set, prob_table):
    colors_a, count_a = devotions[0]
    colors_b, count_b = devotions[1]
    colors_a = colors_a & lands_set
    colors_b = colors_b & lands_set
    if count_b > count_a:
        colors_a, colors_b = colors_b, colors_a
        count_a, count_b = count_b, count_a
    idx_a = colors_a - colors_b
    idx_b = colors_b - colors_a
    idx_ab = colors_a & colors_b
    usable_count_a = sum(lands[c] for c in idx_a)
    usable_count_b = sum(lands[c] for c in idx_b)
    usable_count_ab = sum(lands[c] for c in idx_ab)
    return prob_table[cmc, min(MAX_REQUIRED_A, count_a), count_b, usable_count_a, usable_count_b, usable_count_ab]


def prob_to_cast_3(devotions, cmc, lands, lands_set, prob_table):
    total_devotion = 0
    prob = 1
    usable_land_colors = set()
    for colors, count in devotions:
        colors = colors & lands_set
        total_devotion += count
        usable_count = sum([lands[c] for c in colors])
        usable_land_colors = usable_land_colors | colors
        prob *= prob_table[cmc, min(MAX_REQUIRED_A, count), 0, usable_count, 0, 0] / 255
    usable_count = sum([lands[c] for c in usable_land_colors])
    return prob * prob_table[cmc, min(MAX_REQUIRED_A, total_devotion), 0, usable_count, 0, 0]


PROB_TO_CAST_BY_COUNT = (prob_to_cast_0, prob_to_cast_1, prob_to_cast_2, prob_to_cast_3, prob_to_cast_3, prob_to_cast_3)


def prob_to_cast(cost_index, probs_by_land, prob_func, devotions, cmc, lands_index, prob_table, lands_list):
    if lands_index not in probs_by_land:
        devotions = tuple(devotions)
        lands, lands_set = lands_list[lands_index]
        probs_by_land[lands_index] = prob_func(devotions, cmc, lands, lands_set, prob_table)
    return probs_by_land[lands_index]


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


@static_vars(prob_cache=None, possible_lands={}, lands_list=[])
def generate_probs(picked, seen, in_pack, costs_list, card_devotions, card_colors, prob_table, total=120, total_tries=256):
    if generate_probs.prob_cache is None:
        generate_probs.prob_cache = [{} for _ in costs_list]
    base_lands, remaining = lands_base(picked, card_colors)
    heap = []
    picked_costs, seen_costs, in_pack_costs = [[card_devotions[c] for c in cl] for cl in (picked, seen, in_pack)]
    costs = (picked_costs, seen_costs, in_pack_costs)
    picked_counter, seen_counter = [Counter(c for c in cs if c >= 0) for cs in (picked_costs, seen_costs)]
    in_pack_set = frozenset(c for c in in_pack_costs if c >= 0)
    costs_to_calc = [(c, generate_probs.prob_cache[c], PROB_TO_CAST_BY_COUNT[len(costs_list[c][1])], *costs_list[c])
                     for c in frozenset(picked_costs + seen_costs + in_pack_costs) if c != -1]
    probs = { -1: 0 }
    if remaining <= 0:
        new_lands = tuple(base_lands)
        if new_lands not in generate_probs.possible_lands:
            generate_probs.possible_lands[new_lands] = len(generate_probs.lands_list)
            generate_probs.lands_list.append((new_lands, frozenset([i for i, x in enumerate(new_lands) if x > 0])))
        lands_index = generate_probs.possible_lands[new_lands]
        probs.update({ c: prob_to_cast(c, cache, prob_func, devotions, cmc, lands_index, prob_table, generate_probs.lands_list)
                       for c, cache, prob_func, cmc, devotions in costs_to_calc })
        picked_probs, seen_probs, in_pack_probs = [[probs[c] for c in cs] for cs in costs]
        heap = [(0, 0, 0, 0, 0, (seen_probs, picked_probs, in_pack_probs))]
        total_tries = 0
    while total_tries > 0:
        base_lands[1:6] = random_basics(remaining)
        prev_best, prev_lands, prev_add, prev_cut = -1, tuple(base_lands), -1, -1
        next_best, next_lands, next_add, next_cut = 0, prev_lands, -1, -1
        while next_best > prev_best and total_tries > 0:
            prev_best, prev_lands, prev_add, prev_cut = next_best, next_lands, next_add, next_cut
            for add in range(1, 6):
                for cut in range(1, 6):
                    if add == cut or add == prev_cut or prev_lands[cut] == 0 or cut == prev_add:
                        continue
                    new_lands = list(prev_lands)
                    new_lands[add] += 1
                    new_lands[cut] -= 1
                    new_lands = tuple(new_lands)
                    if new_lands not in generate_probs.possible_lands:
                        generate_probs.possible_lands[new_lands] = len(generate_probs.lands_list)
                        generate_probs.lands_list.append((new_lands, frozenset([i for i, x in enumerate(new_lands) if x > 0])))
                    lands_index = generate_probs.possible_lands[new_lands]
                    probs.update({ c: prob_to_cast(c, cache, prob_func, devotions, cmc, lands_index, prob_table, generate_probs.lands_list)
                                   for c, cache, prob_func, cmc, devotions in costs_to_calc })
                    picked_probs, seen_probs, in_pack_probs = [[probs[c] for c in cs] for cs in costs]
                    total_picked_prob, total_seen_prob = [sum(v * probs[c] for c, v in cs.items()) for cs in (picked_counter, seen_counter)]
                    max_in_pack_prob = max(probs[c] for c in in_pack_set)
                    total_score = total_picked_prob + max_in_pack_prob * 2 + total_seen_prob / 16
                    entry = (total_score, total_picked_prob, max_in_pack_prob, total_seen_prob, total_tries,
                             (seen_probs, picked_probs, in_pack_probs))
                    if len(heap) < total:
                        heapq.heappush(heap, entry)
                    else:
                        heapq.heappushpop(heap, entry)
                    total_tries -= 1
                    if total_tries <= 0 or total_score > next_best:
                        next_add, next_cut, next_best, next_lands = add, cut, total_score, new_lands
                        break
                if next_best > prev_best or total_tries <= 0:
                    break
    result = [e[5] for e in heap] + [([0 for _ in seen], [0 for _ in picked], [0 for _ in in_pack]) for _ in range(total - len(heap))]
    return [np.uint8(arr) for arr in zip(*result)]


def to_one_hot(item, num_items):
    if item < 0:
        item = num_items
    result = [0 for _ in range(num_items)]
    result[item] = 1
    return result


def load_pick(pick, costs_list, card_devotions, card_colors, prob_table, num_land_combs=16, num_land_tries=50):
    if (not (2 <= len(pick['cardsInPack']) <= 16 and 1 <= len(pick['seen']) <= 360 and 2 <= len(pick['picked']) <= 48))\
            or None in pick['seen'] or None in pick['picked'] or None in pick['cardsInPack']:
        return None
    seen, picked, in_pack = [[c + 1 for c in pick[k]] for k in ('seen', 'picked', 'cardsInPack')]
    in_pack = in_pack + [0 for _ in range(16 - len(in_pack))]
    seen, seen_count = seen + [0 for _ in range(360 - len(seen))], len(seen)
    picked, picked_count = picked + [0 for _ in range(48 - len(picked))], len(picked)
    pack_num, packs_count, pick_num, pack_size = pick['pack'], pick['packs'], pick['pick'], pick['packSize']
    pack_float, pick_float = 3 * pack_num / packs_count, 15 * pick_num / pack_size
    pack_0, pack_1, pack_frac = int(pack_float), min(2, int(pack_float) + 1), pack_float - int(pack_float)
    pick_0, pick_1, pick_frac = int(pick_float), min(14, int(pick_float) + 1), pick_float - int(pick_float)
    coords = ((pack_0, pick_0), (pack_0, pick_1), (pack_1, pick_0), (pack_1, pick_1))
    coord_weights = ((1 - pack_frac) * (1 - pick_frac), (1 - pack_frac) * pick_frac, pack_frac * (1 - pick_frac), pack_frac * pick_frac)
    prob_seen, prob_picked, prob_in_pack = generate_probs(picked, seen, in_pack, costs_list, card_devotions,
                                                          card_colors, prob_table, num_land_combs, num_land_tries)
    return (in_pack, seen, seen_count, picked, picked_count, coords, coord_weights, prob_seen,
            prob_picked, prob_in_pack)


def parse_picks(pick_file_name, costs_list, card_devotions, card_colors, prob_table, num_land_combs=16, num_land_tries=50):
    logger.info(f'Started parsing {pick_file_name}')
    parsed_data = []
    draftNum = 0
    pick_file_json = []
    with open(pick_file_name, 'r') as pick_file:
        pick_file_json = json.load(pick_file)
    for draft_entry in pick_file_json:
        for pick in draft_entry["picks"]:
            loaded_data = load_pick(pick, costs_list, card_devotions, card_colors, prob_table, num_land_combs, num_land_tries)
            if loaded_data is not None:
                parsed_data.append(loaded_data)
        draftNum += 1
        logger.debug(f'Completed draft {draftNum} and pick {len(parsed_data)}')
    logger.info(f'Parsed {pick_file_name} which had {len(parsed_data)} usable picks.')

    return [feature[0](parsed) for feature, parsed in zip(FEATURES, zip(*parsed_data))]


def parse_all_picks(pick_cache_dir, num_land_combs=16, num_land_tries=50, num_workers=None):
    card_devotions, card_colors, costs_list = load_card_data()
    prob_table = load_prob_table()
    parsed_data = []
    shared_params = (costs_list, card_devotions, card_colors, prob_table, num_land_combs, num_land_tries)
    with multiprocessing.Pool(num_workers) as proc_pool:
        parsed_data = proc_pool.starmap(parse_picks, [(f'data/drafts/{i}.json', *shared_params)
                                                      for i in range(len(glob.glob('data/drafts/*.json')))][1308:],
                                        1)
    parsed_data = [np.concatenate(feature) for feature in zip(*parsed_data)]
    pick_cache_dir.mkdir(exist_ok=True, parents=True)
    for feature, parsed in zip(FEATURES, parsed_data):
        path = pick_cache_dir / f'{feature[2]}.bin'
        path.touch()
        num_rows = path.stat().st_size // np.dtype(feature[0]).itemsize // int(np.prod(feature[1]))
        saved = np.memmap(path, mode='r+', dtype=feature[0], shape=(num_rows + parsed.shape[0], *feature[1]))
        saved[num_rows:] = parsed
        saved.flush()


if __name__ == '__main__':
    num_land_combs = NUM_LAND_COMBS
    num_land_tries = int(sys.argv[1])
    num_workers = int(sys.argv[2])
    formatter = logging.Formatter('[%(levelname)s/%(processName)s] %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    parse_all_picks(Path('data/parsed_picks/'), num_land_combs, num_land_tries, num_workers)
