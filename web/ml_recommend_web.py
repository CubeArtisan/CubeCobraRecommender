import json
import numpy as np
import unidecode
from tensorflow import keras
import urllib.request

ROOT = "https://cubecobra.com"

def get_ml_recommend(model, int_to_card, card_to_int, cube_name, amount, root=ROOT, non_json=False):
    url = root + "/cube/api/cubelist/" + cube_name

    fp = urllib.request.urlopen(url)
    mybytes = fp.read()

    mystr = mybytes.decode("utf8")
    fp.close()

    card_names = mystr.split("\n")

    num_cards = len(int_to_card)

    cube_indices = []
    for name in card_names:
        idx = card_to_int.get(unidecode.unidecode(name.lower()), None)
        # skip unknown cards (e.g. custom cards)
        if idx is not None:
            cube_indices.append(idx)

    cube = np.zeros(num_cards)
    cube[cube_indices] = 1

    def recommend(model, data):
        encoded = model.encoder(data)
        return model.decoder(encoded)

    print('Preparing to fetch recommendations')
    cube = np.array(cube, dtype=float).reshape(1, num_cards)
    results = recommend(model, cube)[0].numpy()
    print('Got results,', results)

    ranked = results.argsort()[::-1]

    output = {"additions": dict(), "cuts": dict()}

    recommended = 0
    for rec in ranked:
        if cube[0][rec] != 1:
            card = int_to_card[rec]
            if non_json:
                print(card)
            else:
                output["additions"][card] = results[rec].item()
            recommended += 1
            if recommended >= amount:
                break

    for idx in cube_indices:
        card = int_to_card[idx]
        output["cuts"][card] = results[idx].item()
    print('Output:', output)
    if not non_json:
        return output
