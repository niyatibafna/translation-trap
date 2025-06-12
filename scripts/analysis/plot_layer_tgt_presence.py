import os, sys
import pandas as pd
import json
from  collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.cm as cm
import random


sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.lang_codes import lang_order, map_flores_to_ours, map_ours_to_flores, label_map

project_home = os.path.join(os.path.dirname(__file__), '..', '..')

def format_perc(value):
    """
    Format the value as a percentage with 2 decimal places
    """
    return round(value * 100, 1)


filename = os.path.join(project_home, "analysis", "overview", "tgt_lang_presence_main.pdf")
# filename = os.path.join(project_home, "analysis", "overview", "tgt_lang_presence_main.png")
tgt_langs = ["fra_Latn", "hin_Deva", "uzn_Latn", "swa_Latn", "nep_Deva"]
colours = ["blue", "orange", "green", '#D55E00', '#CC79A7']
# colours = ['#0072B2',  # Blue (replaces "blue")
#            '#E69F00',  # Orange (replaces "orange")
#            '#009E73',  # Green (replaces "green")
#            '#D55E00',  # Vermilion (safer than bright red)
#            '#CC79A7']  # Reddish Purple (replaces "purple")
markers = ["o", "D", "x"]

plt.figure(figsize=(3, 2))

for exp_idx, exp_key in enumerate(["aya-23-8b_wt_temp-0", "llama-3.1-8b-it_wt"]):
    output_dir = os.path.join(project_home, "analysis", "overview", exp_key)

    with open(os.path.join(output_dir, f"{exp_key}_tgt_lang_presence_per_src_tgt.json"), "r") as f:
        tgt_lang_presence_per_src_tgt = json.load(f)

    tgt_lang_presence = defaultdict(lambda: defaultdict(lambda: 0))
    for source in tgt_lang_presence_per_src_tgt:
        for target in tgt_lang_presence_per_src_tgt[source]:
            for layer in tgt_lang_presence_per_src_tgt[source][target]:
                tgt_lang_presence[target][layer] += tgt_lang_presence_per_src_tgt[source][target][layer]
    # Now we normalize the tgt_lang_presence by the number of source languages
    for target in tgt_lang_presence:
        for layer in tgt_lang_presence[target]:
            tgt_lang_presence[target][layer] = format_perc(tgt_lang_presence[target][layer] / len(tgt_lang_presence_per_src_tgt))

    tgt_langs = [map_flores_to_ours(tgt) for tgt in tgt_langs]
    tgt_langs = [tgt_lang for tgt_lang in tgt_langs if tgt_lang in tgt_lang_presence]

    for idx, target in enumerate(tgt_langs):
        layers = []
        values = []
        for layer in tgt_lang_presence[target]:
            layers.append(int(layer))
            values.append(tgt_lang_presence[target][layer])
        # Sort the layers and values
        layers, values = zip(*sorted(zip(layers, values), key=lambda x: int(x[0])))
        # Convert to int
        plt.plot(layers, values, label=label_map(target, exp_key=exp_key), marker=markers[exp_idx], color=colours[idx])


plt.xlabel("Layer")
plt.ylabel("Target presence (%)")
plt.xticks([i for i in range(-10, 0)])
# plt.title(model_name)

# Legend showing colour and language
legend_elements = []
for idx, tgt_lang in enumerate(tgt_langs):
    legend_elements.append(plt.Line2D([0], [0], linestyle='-', label=label_map(tgt_lang, exp_key=exp_key),
                                        color=colours[idx]))



# Legend showing marker and model

for idx, exp_key in enumerate(["aya-23-8b_wt_temp-0", "llama-3.1-8b-it_wt"]):
    model_name = {"llama-3.1-8b-it_wt": "llama-3.1", \
            "llama-3.1-70b-it_wt": "llama-3.1-70b", \
            "aya-23-8b_wt_temp-0": "aya-23"}[exp_key]

    legend_elements.append(plt.Line2D([0], [0], marker=markers[idx], color='w', label=model_name,
                                       markeredgecolor='k', markersize=7.5))
plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)

plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)

plt.savefig(filename, bbox_inches='tight')
plt.tight_layout()
plt.close()