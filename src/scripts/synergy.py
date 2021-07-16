import argparse
import json

import tensorflow as tf
import numpy as np

from src.ml.draftbots import DraftBot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help="The name to save this model under.")
    parser.add_argument('--card', '-c', type=str, required=True, help="The card name to calculate synergy for.")
    parser.add_argument('--embed-dims', '-d', type=int, default=16, help="The number of dimensions to use for card embeddings.")
    parser.add_argument('--num-heads', type=int, default=16, help='The number of heads to use for the self attention layer.')
    parser.add_argument('--top-n', '-n', type=int, default=10, help='The number of most synergistic cards to show.')
    args = parser.parse_args()

    with open('data/maps/int_to_card.json', 'r') as cards_file:
        cards_json = json.load(cards_file)
    card_to_int = {card['name_lower']: i + 1 for i, card in enumerate(cards_json)}

    print('Loading DraftBot model.')
    output_dir = f'././ml_files/{args.name}/'
    draftbots = DraftBot(len(cards_json) + 1, 1, embed_dims=args.embed_dims, num_heads=args.num_heads, name='DraftBots')
    latest = tf.train.latest_checkpoint(output_dir)
    draftbots.load_weights(latest)

    card_index = card_to_int[args.card]
    picked_cards = np.zeros((1,48), dtype=np.int32)
    picked_cards[0] = card_index
    attention_mask = tf.logical_and(tf.expand_dims(picked_cards > 0, 1), tf.expand_dims(picked_cards > 0, 2))
    picked_card_embeds = tf.gather(draftbots.card_embeddings, picked_cards)
    pool_attentions = draftbots.self_attention(picked_card_embeds, picked_card_embeds, attention_mask=attention_mask)
    pool_embed = tf.math.l2_normalize(pool_attentions[0, 0])
    card_embeddings = tf.math.l2_normalize(draftbots.card_embeddings, axis=1)
    card_synergies = tf.einsum('ce,e->c', card_embeddings, pool_embed).numpy()
    ranked = card_synergies.argsort()

    for idx in ranked[:args.top_n]:
        print(f'{cards_json[idx - 1]["name"]}: {card_synergies[idx]}')
    print('...')
    for idx in ranked[-args.top_n:]:
        print(f'{cards_json[idx - 1]["name"]}: {card_synergies[idx]}')