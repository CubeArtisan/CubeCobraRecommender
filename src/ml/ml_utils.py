import re
from copy import deepcopy
from itertools import chain
from typing import Dict, List, Tuple, Union

import numpy as np

MAX_PATH_LENGTH = 32
NUM_INPUT_PATHS = 64


def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


def full_split(identifier):
    identifier = identifier.replace('.', ' ')
    identifier = identifier.replace(',', ' ')
    identifier = identifier.replace("'", '')
    return [y for x in identifier.split(' ') for y in camel_case_split(x) if len(y) > 0]


def convert_structure(structure: Union[list, dict, int, str, bool],
                      key: str, vocab_dict: Dict[str, Tuple[int, np.array]],
                      children: List[List[int]],
                      node_labels: List[int],
                      node_heights: List[int],
                      node_depths: List[int],
                      depth: int, nlp) -> Tuple[int, int]:
    our_children = []
    max_child_height = 0
    if isinstance(structure, list):
        for index, child in enumerate(structure):
            child_index, height = convert_structure(child, str(index),
                                                    vocab_dict, children,
                                                    node_labels, node_heights,
                                                    node_depths, depth + 1, nlp)
            our_children.append(child_index)
            max_child_height = max(max_child_height, height)
    elif isinstance(structure, dict):
        for key, child in structure.items():
            child_index, height = convert_structure(child, key, vocab_dict,
                                                    children, node_labels, node_heights,
                                                    node_depths, depth + 1, nlp)
            our_children.append(child_index)
            max_child_height = max(max_child_height, height)
    else:
        if key in vocab_dict:
            vocab, _ = vocab_dict[key]
        else:
            vocab = len(vocab_dict)
            components = full_split(key)
            components = nlp(' '.join(components))
            vector = np.zeros((300,))
            for component in components:
                vector += component.vector
            if len(components) > 0:
                vector =  vector / len(components)
            vocab_dict[key] = vocab, vector
        key_vocab = vocab
        key = str(structure)
        if key in vocab_dict:
            vocab, _ = vocab_dict[key]
        else:
            vocab = len(vocab_dict)
            components = full_split(key)
            components = nlp(' '.join(components))
            vector = np.zeros((300,))
            for component in components:
                vector += component.vector
            if len(components) > 0:
                vector = vector / len(components)
            vocab_dict[key] = vocab, vector
        value_index = len(node_labels)
        node_labels.append(vocab)
        node_heights.append(0)
        for index in range(len(children)):
            children[index].append(0)
        node_depths.append(depth + 1)
        key_index = len(node_labels)
        node_heights.append(1)
        if len(children) < 1:
            children.append([0 for _ in node_labels])
        children[0].append(value_index)
        for child in children[1:]:
            child.append(0)
        node_labels.append(key_vocab)
        node_depths.append(depth)
        return key_index, 0
    if key in vocab_dict:
        vocab, _ = vocab_dict[key]
    else:
        vocab = len(vocab_dict)
        components = full_split(key)
        components = nlp(' '.join(components))
        vector = np.zeros((300,))
        for component in components:
            vector += component.vector
        if len(components) > 0:
            vector = vector / len(components)
        vocab_dict[key] = vocab, vector
    for _ in range(len(structure), len(children)):
        our_children.append(-1)
    our_index = len(node_labels)
    for index, child_index in enumerate(our_children):
        if len(children) <= index:
            children.append([0 for _ in node_labels])
        children[index].append(child_index)
    node_labels.append(vocab)
    node_heights.append(max_child_height + 1)
    node_depths.append(depth)
    return our_index, max_child_height + 1


def generate_card_structures(cards):
    import spacy
    print('Converting cards to trees.')
    nlp = spacy.load('en_core_web_lg')
    vocab_dict = {"<unkown>": (0, np.zeros((300,)))}
    children = []
    node_labels = [0]
    card_indices = [0]
    node_heights = [-1]
    node_depths = [-1]
    lengths = {'superType': 2, 'type': 5, 'subType': 4}
    continuous_features_template = {
        'cost': {'generic': 0, 'w': 0, 'u': 0, 'b': 0, 'r': 0, 'g': 0, 'x': 0, 'y': 0,
                'z': 0, 'w-u': 0, 'u-b': 0, 'b-r': 0, 'r-g': 0, 'g-w': 0, 'w-b': 0,
                'u-r': 0, 'b-g': 0, 'r-w': 0, 'g-u': 0, 'w-p': 0, 'u-p': 0, 'b-p': 0,
                'r-p': 0, 'g-p': 0, 'split': 0, 'c': 0, '': 0, 's': 0, '2-w': 0,
                '2-u': 0, '2-b': 0, '2-r': 0, '2-g': 0},
        'power': {'value': 0, 'existant': 0, 'star': 0, 'plus': 0, 'squared': 0, 'infinite': 0,
                  'question': 0},
        'toughness': {'value': 0, 'existant': 0, 'star': 0, 'plus': 0, 'squared': 0, 'infinite': 0,
                      'question': 0},
        'loyalty': {'value': 0, 'existant': 0, 'x': 0, 'star': 0, 'dice_roll': 0},
    }
    categorical_features_template = {
        'oracleText': [0],
        'superType': [0 for _ in range(lengths['superType'])],
        'type': [0 for _ in range(lengths['type'])],
        'subType': [0 for _ in range(lengths['subType'])],
    }
    continuous_ordering = {'cost': ['generic', 'w', 'u', 'b', 'r', 'g', 'x', 'y', 'z', 'w-u', 'u-b',
                                    'b-r', 'r-g', 'g-w', 'w-b', 'u-r', 'b-g', 'r-w', 'g-u', 'w-p',
                                    'u-p', 'b-p', 'r-p', 'g-p', 'split', 'c', '', 's', '2-w', '2-u',
                                    '2-b', '2-r', '2-g'],
                           'power': ['value', 'existant', 'star', 'plus', 'squared', 'infinite',
                                     'question'],
                           'toughness': ['value', 'existant', 'star', 'plus', 'squared', 'infinite',
                                         'question'],
                           'loyalty': ['value', 'existant', 'x', 'star', 'dice_roll']}
    categorical_ordering = ['oracleText', 'superType', 'type', 'subType']
    card_categorical_features = [deepcopy(categorical_features_template)]
    card_continuous_features = [deepcopy(continuous_features_template)]

    def get_vocab_value(value):
        if value not in vocab_dict:
            vocab = len(vocab_dict)
            components = full_split(value)
            components = nlp(' '.join(components))
            vector = np.zeros((300,))
            count = 0
            for component in components:
                vector += component.vector
                count += 1
            length = np.linalg.norm(vector)
            if length > 0:
                vector = vector / length
            else:
                RANDOM_DIMS = 15
                weight = np.sqrt(1 / RANDOM_DIMS)
                for i in range(RANDOM_DIMS):
                    vector[np.random.randint(300)] = weight
            vocab_dict[value] = vocab, vector
        vocab, _ = vocab_dict[value]
        return vocab
    for card in cards:
        categorical_features = deepcopy(categorical_features_template)
        continuous_features = deepcopy(continuous_features_template)
        if isinstance(card, dict):
            for key, value in card.items():
                if key == "parsedCost":
                    for pip in value:
                        if pip in continuous_features['cost']:
                            continuous_features['cost'][pip] += 1
                        elif pip == 'hw':
                            continuous_features['cost']['w'] += 0.5
                        else:
                            try:
                                continuous_features['cost']['generic'] += float(pip)
                            except ValueError:
                                pass
                elif key == 'power' or key == 'toughness':
                    try:
                        continuous_features[key]['value'] = float(value)
                    except ValueError:
                        pass
                    continuous_features[key]['existant'] = 1
                    if value == '*':
                        continuous_features[key]['star'] = 1
                    elif value == '1+*' or value == '*+1':
                        continuous_features[key]['star'] = 1
                        continuous_features[key]['plus'] = 1
                    elif value == '2+*':
                        continuous_features[key]['star'] = 1
                        continuous_features[key]['plus'] = 2
                    elif value == '7-*':
                        continuous_features[key]['star'] = -1
                        continuous_features[key]['plus'] = 7
                    elif value == '*²':
                        continuous_features[key]['star'] = 1
                        continuous_features[key]['squared'] = 1
                    elif value == '∞':
                        continuous_features[key]['infinite'] = 1
                    elif value == '?':
                        continuous_features[key]['question'] = 1
                elif key == 'loyalty':
                    try:
                        continuous_features['loyalty']['value'] = int(value)
                    except ValueError:
                        pass
                    continuous_features['loyalty']['existant'] = 1
                    if value == 'X':
                        continuous_features['loyalty']['x'] = 1
                    elif value == '*':
                        continuous_features['loyalty']['star'] = 1
                    elif value == '1d4+1':
                        continuous_features['loyalty']['dice_roll'] = 1
                elif key == 'oracleText':
                    categorical_features['oracleText'] = [get_vocab_value(value)]
                elif key != 'parsed':
                    categorical_features[key] = [get_vocab_value(v) for v in value]
                    categorical_features[key] += [0 for _ in range(len(categorical_features[key]),
                                                                   lengths[key])]
            card = card.get("parsed", "")
        card_index, _ = convert_structure(card, "", vocab_dict, children,
                                          node_labels, node_heights, node_depths, 0, nlp)
        card_indices.append(card_index)
        card_categorical_features.append(categorical_features)
        card_continuous_features.append(continuous_features)
    card_continuous_features = [[features[key][k]
                                 for key, value in continuous_ordering.items()
                                 for k in value]
                                for features in card_continuous_features]
    card_categorical_features = [list(chain(*(features[key] for key in categorical_ordering)))
                                 for features in card_categorical_features]
    children_count = len(children)
    children = [[child[i] for child in children]
                for i in range(len(node_labels))]
    node_parents = [0 for _ in node_labels]
    for i, our_children in enumerate(children):
        for child in our_children:
            if child != 0:
                node_parents[child] = i
    continuous_features_count = len(card_continuous_features[0])
    categorical_features_count = len(card_categorical_features[0])
    print(len(vocab_dict), len(node_labels), children_count, max(node_depths),
          continuous_features_count, categorical_features_count)
    return card_indices, node_depths, node_heights, node_labels, vocab_dict, node_parents, \
        card_continuous_features, continuous_features_count, card_categorical_features, \
        categorical_features_count


def generate_paths(cards):
    card_indices, node_depths, node_heights, node_labels, vocab_dict, node_parents,\
        card_continous_features, continuous_features_count, card_categorical_features,\
        categorical_features_count = generate_card_structures(cards)
    print('Calculating AST paths')
    all_paths = [[]]
    max_path_length = 0
    max_pairs = 0
    for i, max_index in enumerate(card_indices):
        if i == 0:
            continue
        min_index = card_indices[i - 1] + 1
        paths = []
        index_range = [x for x in range(min_index, max_index) if node_heights[x] == 0]
        pairs = [(index_range[x], index_range[y])
                 for x in range(len(index_range))
                 for y in range(x)]
        max_pairs = max(max_pairs, len(pairs))
        if len(index_range) == 0:
            all_paths.append([])
            continue
        for start, end in pairs:
            start_depth = node_depths[start]
            end_depth = node_depths[end]
            if end_depth > start_depth:
                start, end = end, start
                start_depth, end_depth = end_depth, start_depth
            path = [node_labels[start]]
            end_path = [node_labels[end]]
            while start_depth > end_depth:
                start = node_parents[start]
                start_depth -= 1
                path.append(node_labels[start])
            iterator = 0
            while node_parents[start] != node_parents[end]:
                start = node_parents[start]
                end = node_parents[end]
                path.append(node_labels[start])
                end_path.append(node_labels[end])
                iterator += 1
            path.append(node_labels[node_parents[start]])
            path += end_path[::-1]
            max_path_length = max(max_path_length, len(path))
            if len(path) > MAX_PATH_LENGTH:
                continue
            paths.append(path)
        all_paths.append(paths)
    print('max_path_length', max_path_length)
    print('max_pairs', max_pairs)
    for node_paths in all_paths:
        for path in node_paths:
            for _ in range(len(path), MAX_PATH_LENGTH):
                path.append(0)
        for _ in range(len(node_paths), NUM_INPUT_PATHS):
            node_paths.append([0 for _ in range(MAX_PATH_LENGTH)])

    return all_paths, card_continous_features, continuous_features_count,\
        card_categorical_features, categorical_features_count, vocab_dict
