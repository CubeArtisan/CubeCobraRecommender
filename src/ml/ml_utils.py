import random
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import numpy as np
import spacy

MAX_PATH_LENGTH = 16
NUM_INPUT_PATHS = 32
NUM_PATHS = 32  # 512

nlp = spacy.load('en_core_web_lg')


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
                      depth: int) -> Tuple[int, int]:
    our_children = []
    max_child_height = 0
    if isinstance(structure, list):
        for index, child in enumerate(structure):
            child_index, height = convert_structure(child, str(index),
                                                    vocab_dict, children,
                                                    node_labels, node_heights,
                                                    node_depths, depth + 1)
            our_children.append(child_index)
            max_child_height = max(max_child_height, height)
    elif isinstance(structure, dict):
        for key, child in structure.items():
            child_index, height = convert_structure(child, key, vocab_dict,
                                                    children, node_labels, node_heights,
                                                    node_depths, depth + 1)
            our_children.append(child_index)
            max_child_height = max(max_child_height, height)
    else:
        key = f'{key}.{structure}'
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
        our_index = len(node_labels)
        node_labels.append(vocab)
        node_heights.append(0)
        for index in range(len(children)):
            children[index].append(0)
        node_depths.append(depth)
        return our_index, 0
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
        children[index].append(0)
    node_labels.append(vocab)
    node_heights.append(max_child_height + 1)
    node_depths.append(depth)
    return our_index, max_child_height + 1


def generate_card_structures(cards):
    print('Converting cards to trees.')
    vocab_dict = {"<unkown>": (0, np.zeros((300,)))}
    children = []
    node_labels = [0]
    card_indices = [0]
    node_heights = [-1]
    node_depths = [-1]
    for card in cards:
        card_index, _ = convert_structure(card, "", vocab_dict, children,
                                          node_labels, node_heights, node_depths, 0)
        card_indices.append(card_index)
    node_depths = node_depths
    children_count = len(children)
    children = [[child[i] for child in children]
                for i in range(len(node_labels))]
    node_parents = [0 for _ in node_labels]
    for i, our_children in enumerate(children):
        for child in our_children:
            if child != 0:
                node_parents[child] = i
    print(len(vocab_dict), len(node_labels), children_count)
    return card_indices, node_depths, node_heights, node_labels, vocab_dict, node_parents


def generate_paths(cards, return_vocab_count=False):
    card_indices, node_depths, node_heights, node_labels, vocab_count, node_parents = \
        generate_card_structures(cards)
    print('Calculating AST paths')
    all_paths = [[]]
    for i, max_index in enumerate(card_indices):
        if i == 0:
            continue
        min_index = card_indices[i - 1] + 1
        paths = []
        index_range = [x for x in range(min_index, max_index) if node_heights[x] == 0]
        computed_values = defaultdict(lambda: defaultdict(lambda: False))
        used_indices = set()
        iterations = 0
        if len(index_range) == 0:
            all_paths.append([])
            continue
        while len(paths) < NUM_PATHS and iterations < NUM_PATHS * 2:
            iterations += 1
            new_index_range = [x for x in index_range if x not in used_indices]
            if len(new_index_range) > 0:
                start = random.choice(new_index_range)
            else:
                start = random.choice(index_range)
            remaining = [x for x in index_range if x != start and not computed_values[start][x]]

            if len(remaining) == 0:
                continue
            new_remaining = [x for x in remaining if x not in used_indices]
            if len(new_remaining) > 0:
                end = random.choice(new_remaining)
            else:
                end = random.choice(remaining)
            used_indices.add(start)
            used_indices.add(end)
            computed_values[start][end] = True
            computed_values[end][start] = True
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
                path.append(start)
            while node_parents[start] != node_parents[end]:
                start = node_parents[start]
                end = node_parents[end]
                path.append(node_labels[start])
                end_path.append(node_labels[end])
            path.append(node_labels[node_parents[start]])
            path += end_path[::-1]
            if len(path) > MAX_PATH_LENGTH:
                continue
            paths.append(path)
        all_paths.append(paths)
    for node_paths in all_paths:
        for path in node_paths:
            for _ in range(len(path), MAX_PATH_LENGTH):
                path.append(0)
    if return_vocab_count:
        return all_paths, vocab_count
    else:
        return all_paths
