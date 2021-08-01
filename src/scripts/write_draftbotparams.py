#!/usr/bin/env python
import argparse
import json
import logging
import struct
from pathlib import Path

import tensorflow as tf

from src.ml.draftbots import DraftBot

EMBED_DIMS_CHOICES = (2, 4, 8, 16, 32, 64, 128, 256, 512)

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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-o', '-n', type=str, required=True, help="The name to save this model under.")
    parser.add_argument('--embed-dims', '-d', type=int, default=16, choices=EMBED_DIMS_CHOICES,
                        help="The number of dimensions to use for card embeddings.")
    parser.add_argument('--num-heads', type=int, default=16,
                        help='The number of heads to use for the self attention layer.')
    args = parser.parse_args()

    with open('data/maps/int_to_card.json', 'r') as cards_file:
        cards_json = json.load(cards_file)

    output_dir = Path(f'ml_files/{args.name}')
    draftbots = DraftBot(len(cards_json) + 1, 1, embed_dims=args.embed_dims, num_heads=args.num_heads, name='DraftBots')
    latest = tf.train.latest_checkpoint(output_dir)
    draftbots.load_weights(latest).expect_partial()

    card_ratings = tf.nn.sigmoid(tf.constant(args.embed_dims, dtype=tf.float32) * draftbots.card_rating_logits).numpy()
    card_embeddings = draftbots.card_embeddings.numpy()
    oracle_weights = tf.math.softplus(draftbots.oracle_weights).numpy()
    basic_ids = ['b2c6aa39-2d2a-459c-a555-fb48ba993373', "56719f6a-1a6c-4c0a-8d21-18f7d7350b68",
                 "b34bb2dc-c1af-4d77-b0b3-a0fb342a5fc6", "a3fb7228-e76b-4e96-a40e-20b5fed75685",
                 "bc71ebf6-2056-41f7-be35-b2e5c34afa99"]
    with open(output_dir / 'draftbotparams.bin', 'wb') as output_file:
        written = 0
        written += output_file.write(struct.pack(f'{args.embed_dims}fB', *card_embeddings[0], 6))

        for i, name in enumerate(('Rating', 'Pick Synergy', 'Colors', 'Internal Synergy', 'Openness', 'External Synergy')):
            weights = oracle_weights[:, :, i].flatten()
            encoded = name.encode()
            written += output_file.write(struct.pack(f'{len(weights)}f{len(encoded) + 1}s', *weights, encoded))
        written += output_file.write(struct.pack('I', len(cards_json)))
        print('cards_start', written)
        for i, card in enumerate(cards_json):
            rating = card_ratings[i + 1]
            embedding = card_embeddings[i + 1]
            color_identity = 32
            if 'land' in card.get('type', '').lower():
                color_identity = COLOR_COMB_INDEX[frozenset(c.lower() for c in card.get('color_identity', []))]
                if card.get('name_lower', '') in FETCH_LANDS:
                    color_identity = COLOR_COMB_INDEX[frozenset(FETCH_LANDS[card.get('name_lower')])]
            if card.get('oracle_id', '') in basic_ids:
                print(card.get('oracle_id'), color_identity, card.get('color_identity'), i)
            cmc = min(card.get('cmc', 0), 20)
            symbols = card.get('parsed_cost', [])
            written += output_file.write(struct.pack('f', rating))
            if i == 0: print('Embedding', written)
            written += output_file.write(struct.pack(f'{args.embed_dims}f', *embedding))
            if i == 0: print('Color Identity', written)
            written += output_file.write(struct.pack('B', color_identity))
            if i == 0: print('cmc', written)
            written += output_file.write(struct.pack('B', int(cmc)))
            if i == 0: print('len symbols', written)
            written += output_file.write(struct.pack('B', len(symbols)))
            for symbol in symbols:
                encoded = symbol.encode()
                if len(encoded) > 3:
                    encoded = encoded[:3]
                elif len(encoded) < 3:
                    encoded = (symbol + ' ' * (3 - len(encoded))).encode()[:3]
                written += output_file.write(struct.pack(f'3s', encoded))
            oracle_id = card.get('oracle_id', '')
            encoded = oracle_id.encode()
            if len(encoded) > 36:
                print(i, oracle_id)
                encoded = encoded[:36]
            elif len(encoded) < 36:
                print(i, oracle_id)
                encoded = str(i).encode()
            written += output_file.write(struct.pack(f'36s', encoded))
            if i == 0:
                print(written)