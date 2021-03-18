# enable sibling imports
import glob
import json
import math
import pickle
import sys
from pathlib import Path

from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass

def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        if isinstance(o, np.ndarray):
            return o.nbytes
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

if __name__ == "__main__":
    from os.path import dirname as dir
    sys.path.append(dir(sys.path[0]))

from generator import DraftBotGenerator
from draftbots import DraftBot
from metrics import FilteredBinaryAccuracy, TripletFilteredAccuracy, ContrastiveFilteredAccuracy
from losses import TripletLoss, ContrastiveLoss

if __name__ == "__main__":
    cards_json = []
    with open('data/intToCard.json', 'r') as cards_file:
        cards_json = json.load(cards_file)
    pickle_cache_file = Path('data/parsed_picks.pickle')
    parsed_data = None
    if pickle_cache_file.exists():
        with open(pickle_cache_file, 'rb') as pickle_file:
            parsed_data = pickle.load(pickle_file)
    if not parsed_data:
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
        COLOR_COMBINATIONS = [frozenset(list(c)) for c in ['', 'w', 'u', 'b', 'r', 'g', 'wu', 'ub',
                                                           'br', 'rg', 'gw', 'wb', 'ur',
                              'bg', 'rw', 'gu', 'gwu', 'wub', 'ubr', 'brg', 'rgw', 'rwb', 'gur',
                              'wbg', 'urw', 'bgu', 'ubrg', 'wbrg', 'wurg', 'wubg', 'wubr', 'wubrg']]
        COLOR_COMB_INDEX = { s: i for i, s in enumerate(COLOR_COMBINATIONS) }
        INTERSECTS_LIST = [frozenset([i for i, b in enumerate(COLOR_COMBINATIONS) if len(a & b) > 0]) for a in COLOR_COMBINATIONS]
        INTERSECTS_LOOKUP = { s: l for s, l in zip(COLOR_COMBINATIONS, INTERSECTS_LIST) }
        card_colors = [None,]
        cmcs = [0,]
        card_devotions = [[],]
        embeddings = [[0 for _ in range(64)],]
        costs = set()

        for card in cards_json:
            cmc = card["cmc"]
            cmcs.append(cmc)
            type_line = card["type"]
            colors = set([c.lower() for c in card.get("color_identity", [])])
            if "Land" in type_line:
                if "Plains" in type_line:
                    colors.add('w')
                if "Island" in type_line:
                    colors.add('u')
                if "Swamp" in type_line:
                    colors.add('b')
                if "Mountain" in type_line:
                    colors.add('r')
                if "Forest" in type_line:
                    colors.add('g')
                name = card["name"]
                if name in FETCH_LANDS:
                    colors = FETCH_LANDS[name]
                card_colors.append(COLOR_COMB_INDEX[frozenset(colors)])
            else:
                card_colors.append(None)
            parsed_cost = card.get('parsed_cost', [])
            devotions = {}
            for symbol in parsed_cost:
                symbol = symbol.lower()
                if 'p' in symbol or '2' in symbol:
                    continue
                symbol_colors = ''.join([c for c in 'wubrg' if c in symbol])
                if len(symbol_colors) > 0:
                    devotions[symbol_colors] = devotions.get(symbol_colors, 0) + 1
            devotion_costs = [(INTERSECTS_LOOKUP[frozenset(c)], v) for c, v in devotions.items()]
            card_devotions.append(devotion_costs)
            costs.add((cmc, frozenset(devotion_costs)))
            embedding = card.get('embedding', [0 for _ in range(64)])
            if len(embedding) == 0:
                embedding = [0 for _ in range(64)]
            if len(embedding) != 64:
                print(len(embedding), embedding)
                sys.exit(1)
            embeddings.append(embedding)
        synergies = np.full((len(embeddings), len(embeddings)), -1, dtype=np.float32)
        norms = [np.linalg.norm(embedding) for embedding in embeddings]
        print("Populated all card data")
        print(f"There are {len(costs)} unique costs.")
        # synergies_size = total_size(synergies)
        # print(f'size of synergies is: {synergies_size}/{synergies_size/1024}K/{synergies_size/1024/1024}M')

        def calculate_synergy(i, j):
            if i == j:
                return 10.0
            if synergies[i][j] < 0:
                if norms[i] * norms[j] > 0:
                    similarity = np.dot(embeddings[i], embeddings[j]) / np.linalg.norm(embeddings[i]) / np.linalg.norm(embeddings[j])
                    scaled = max(0, similarity - 0.7) / 0.3
                    transformed = scaled / (1 - scaled)
                    if math.isnan(transformed):
                        transformed = 10.0
                    synergies[i][j] = synergies[j][i] = min(transformed, 10.0)
                else:
                    synergies[i][j] = synergies[j][i] = 0
            return synergies[i][j]

        prob_table = np.full((9, 7, 4, 18, 18, 18), 0, dtype=np.float32)
        max_prob_indices = np.uint8([10, 8, 4, 17, 17, 17])
        prob_table_json = []
        with open('data/probTable.json', 'r') as prob_file:
            prob_table_json = json.load(prob_file)
        max_cmc = 0
        max_required_a = 0
        max_required_b = 0
        max_land_count_a = 0
        max_land_count_b = 0
        max_land_count_ab = 0
        for str_cmc, nested1 in prob_table_json.items():
            max_cmc = max(int(str_cmc), max_cmc)
            for str_required_a, nested2 in nested1.items():
                max_required_a = max(int(str_required_a), max_required_a)
                for str_required_b, nested3 in nested2.items():
                    max_required_b = max(int(str_required_b), max_required_b)
                    for str_land_count_a, nested4 in nested3.items():
                        max_land_count_a = max(int(str_land_count_a), max_land_count_a)
                        for str_land_count_b, nested5 in nested4.items():
                            max_land_count_b = max(int(str_land_count_b), max_land_count_b)
                            for str_land_count_ab, prob in nested5.items():
                                max_land_count_ab = max(int(str_land_count_ab), max_land_count_ab)
                                prob_table[int(str_cmc)][int(str_required_a)][int(str_required_b)][int(str_land_count_a)][int(str_land_count_b)][int(str_land_count_ab)] = prob * 255
        print(max_cmc, max_required_a, max_required_b, max_land_count_a, max_land_count_b, max_land_count_ab)
        prob_table_json = None
        print("Populated prob_table")

        possible_lands = set()

        def prob_to_cast_0(devotions, cmc, lands, lands_set):
            return 1

        def prob_to_cast_1(devotions, cmc, lands, lands_set):
            possible_lands.add(tuple(lands[1:]))
            colors, count = list(devotions)[0]
            colors = colors & lands_set
            usable_count = sum([lands[c] for c in colors])
            return prob_table[min(cmc, max_cmc), min(max_required_a, count), 0, usable_count, 0, 0]

        def prob_to_cast_2(devotions, cmc, lands, lands_set):
            possible_lands.add(tuple(lands[1:]))
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
            usable_count_a = sum([lands[c] for c in idx_a])
            usable_count_b = sum([lands[c] for c in idx_b])
            usable_count_ab = sum([lands[c] for c in idx_ab])
            return prob_table[min(cmc, max_cmc), min(max_required_a, count_a), min(max_required_b, count_b),
                              usable_count_a, usable_count_b, usable_count_ab]

        def prob_to_cast_3(devotions, cmc, lands, lands_set):
            possible_lands.add(tuple(lands[1:]))
            total_devotion = 0
            prob = 1
            usable_land_colors = set()
            for colors, count in devotions:
                colors = colors & lands_set
                total_devotion += count
                usable_count = sum([lands[c] for c in colors])
                usable_land_colors = usable_land_colors | colors
                prob *= prob_table[min(cmc, max_cmc), min(max_required_a, count), 0, usable_count, 0, 0] / 255
            usable_count = sum([lands[c] for c in usable_land_colors])
            return prob * prob_table[min(cmc, max_cmc), min(max_required_a, total_devotion), 0, usable_count, 0, 0] / 255

        prob_to_cast = (prob_to_cast_0, prob_to_cast_1, prob_to_cast_2, prob_to_cast_3, prob_to_cast_3, prob_to_cast_3)

        def sum_iter(n, k):
            if n == 0:
                yield []
            if n == 1:
                yield [k]
            elif k == 0:
                yield [0 for _ in range(n)]
            else:
                for minus_one in sum_iter(n - 1, k):
                    yield [0] + minus_one
                for plus_one in sum_iter(n, k - 1):
                    res = list(plus_one)
                    res[0] += 1
                    yield res

        def lands_iter(picked):
            lands = np.full((32,), 0, dtype=np.uint8)
            remaining = 17
            lands_set = set()
            for c in picked:
                colors = card_colors[c]
                if colors is not None:
                    if colors > 5 or colors == 0:
                        lands[colors] += 1
                        lands_set.add(colors)
                        remaining -= 1
            total_count = 0
            lands_set = frozenset(lands_set)
            for mono_counts in sum_iter(5, remaining):
                lands_set_copy = set(lands_set)
                for i, x in enumerate(mono_counts):
                    if x > 0:
                        lands_set_copy.add(i + 1)
                lands_copy = list(lands)
                lands_copy[1:6] = mono_counts
                total_count += 1
                yield (lands_copy, lands_set_copy)

        in_pack_card_indices = []
        in_pack_seen_indices = []
        in_pack_counts = []
        seen_indices = []
        seen_counts = []
        picked_card_indices = []
        picked_seen_indices = []
        picked_counts = []
        pack_0s = []
        pack_1s = []
        pick_0s = []
        pick_1s = []
        frac_packs = []
        frac_picks = []
        internal_synergy_matrices = []
        picked_synergy_matrices = []
        prob_seen_matrices = []
        prob_picked_matrices = []
        prob_in_pack_matrices = []
        chosen_cards = []
        parsed_data = [in_pack_card_indices, in_pack_counts, seen_indices,
                       seen_counts,
                       picked_card_indices, picked_counts, pack_0s, pack_1s, pick_0s, pick_1s,
                       frac_packs, frac_picks, internal_synergy_matrices, picked_synergy_matrices,
                       prob_seen_matrices, prob_picked_matrices, prob_in_pack_matrices, chosen_cards]
        previous_size = total_size(parsed_data)
        draftNum = 0
        pickNum = 0

        def load_pick(pick_entry):
            global pickNum
            seen = [c + 1 for c in pick_entry['seen']]
            seen_count = len(seen)
            if seen_count > 360:
                return
            seen_set = list(set(seen))
            seen_ref = [seen_set.index(c) for c in seen]
            seen_devotions = [(card_devotions[c], prob_to_cast[len(card_devotions[c])], cmcs[c]) for c in seen_set]
            picked_cards = [c + 1 for c in pick_entry['picked']]
            prob_seen_set = [[f(d, c, lands, lands_set) for d, f, c in seen_devotions]
                                  for lands, lands_set in lands_iter(picked_cards)]
            prob_seen = np.uint8([[x[i] for i in seen_ref] for x in prob_seen_set])
            if len(prob_seen) < 5985:
                prob_seen = np.concatenate([prob_seen, np.full((5985 - len(prob_seen), seen_count), 0, dtype=np.uint8)])
            if seen_count < 360:
                prob_seen = np.concatenate([prob_seen, np.full((5985, 360 - seen_count), 0, dtype=np.uint8)], 1)
                seen = seen + [0 for _ in range(360 - len(seen))]
            seen = seen + [0]
            prob_seen = np.concatenate([prob_seen, np.full((5985, 1), 0, dtype=np.uint8)], 1)
            in_pack_cards = [c + 1 for c in pick_entry['cardsInPack']]
            in_pack_count = len(in_pack_cards)
            if in_pack_count > 16:
                return
            in_pack_cards = in_pack_cards + [0 for _ in range(16 - in_pack_count)]
            in_pack = [seen.index(c) for c in in_pack_cards]
            picked_count = len(picked_cards)
            if picked_count > 48:
                return
            picked_cards = picked_cards + [0 for _ in range(48 - picked_count)]
            picked = [seen.index(c) for c in picked_cards]
            pack_num = pick_entry['pack']
            packs_count = pick_entry['packs']
            pack_float = 3 * pack_num / packs_count
            pack_0 = int(pack_float)
            pack_1 = min(2, int(pack_float) + 1)
            pack_frac = pack_float - int(pack_float)
            pick_num = pick_entry['pick']
            pack_size = pick_entry['packSize']
            pick_float = 15 * pick_num / pack_size
            pick_0 = int(pick_float)
            pick_1 = min(14, int(pick_float) + 1)
            pick_frac = pick_float - int(pick_float)
            chosen_card = pick_entry['cardsInPack'].index(pick_entry['chosenCard'])
            internal_synergy = [[calculate_synergy(c1, c2) for c2 in picked_cards] for c1 in picked_cards]
            picked_synergy = [[calculate_synergy(c1, c2) for c2 in picked_cards] for c1 in in_pack_cards]
            prob_picked = prob_seen[:,picked]
            prob_in_pack = prob_seen[:,in_pack]
            in_pack_card_indices.append(in_pack_cards)
            in_pack_counts.append(in_pack_count)
            seen_indices.append(seen[:-1])
            seen_counts.append(seen_count)
            picked_card_indices.append(picked_cards)
            picked_counts.append(picked_count)
            pack_0s.append(pack_0)
            pack_1s.append(pack_1)
            pick_0s.append(pick_0)
            pick_1s.append(pick_1)
            frac_packs.append(pack_frac)
            frac_picks.append(pick_frac)
            internal_synergy_matrices.append(internal_synergy)
            picked_synergy_matrices.append(picked_synergy)
            prob_seen_matrices.append(prob_seen[:,:-1])
            prob_picked_matrices.append(prob_picked)
            prob_in_pack_matrices.append(prob_in_pack)
            chosen_cards.append(chosen_card)
            pickNum += 1
            print(f'Completed pick {pickNum}')

        for pick_file_name in glob.glob('data/drafts/*.json'):
            pick_file_json = []
            with open(pick_file_name, 'r') as pick_file:
                pick_file_json = json.load(pick_file)
            for draft_entry in pick_file_json:
                for pick_entry in draft_entry["picks"]:
                    load_pick(pick_entry)
                draftNum += 1
                print(f'\nCompleted draft {draftNum}')
                break
            print(pick_file_name, len(chosen_cards))
            break
            if pickNum > 2**16:
                break
        print(f'There are {len(costs)} unique costs and {len(possible_lands)} possible land configurations')
        print(f'To store a full table would take {len(costs)*len(possible_lands)/1024/1024}MB')
        with open(pickle_cache_file, 'wb') as parsed_picks_file:
            pickle.dump(parsed_data, parsed_picks_file)

    card_ratings = [0] + [10 ** ((c.get('elo', 1200) / 800) - 2) for c in cards_json]

    args = sys.argv[1:]

    epochs = int(args[0])
    batch_size = int(args[1])
    name = args[2]
    temperature = float(args[3])

    output_dir = f'././ml_files/{name}/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print('Setting up Generator . . .\n')
    generator = DraftBotGenerator(batch_size, *parsed_data)
    print('Setting Up Model . . . \n')
    autoencoder = DraftBot(card_ratings, temperature)
    autoencoder.compile(
        # optimizer='adam',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-06, epsilon=1e+00, global_clipnorm=1),
        # optimizer=tfa.optimizers.AdamW(weight_decay=0.9, learning_rate=1e-06, epsilon=1e+00, global_clipnorm=1),
        loss=['categorical_crossentropy'])
    latest = tf.train.latest_checkpoint(output_dir)
    if latest is not None:
        print('loading checkpoint')
        autoencoder.load_weights(latest)
        # print(autoencoder.get_weights())
    # cross_entropy = tf.keras.losses.CategoricalCrossentropy()
    # for i in range(len(generator)):
    #     inputs, y_true = generator[i]
    #     output = autoencoder.call(inputs)
    #     print(output)
        # print(cross_entropy(output, y_true).numpy())
    # sys.exit(0)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=output_dir + 'model',
        monitor='loss',
        verbose=True,
        save_best_only=True,
        mode='min',
        save_freq='epoch')
    autoencoder.fit(
        generator,
        epochs=epochs,
        callbacks=[cp_callback]
    )
    Path(f'{output_dir}/final').mkdir(parents=True, exist_ok=True)
    autoencoder.save(f'{output_dir}/final', save_format='tf')
