import os, sys
import pandas as pd
import json
from  collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.lang_codes import lang_order, map_flores_to_ours, map_ours_to_flores, label_map



def plot_aggregated_info(aggregated_info, src_lang, tgt_lang, exp_key, output_file):
    """
    Plots the aggregated information.
    """
    # Initialize counts
    layer_counts = {}

    for word, info in aggregated_info.items():
        for layer_str, layer_info in info["per_layer_info"].items():
            layer = int(layer_str)
            if layer not in layer_counts:
                layer_counts[layer] = {
                    'on_target_correct': 0,
                    'on_target_incorrect': 0,
                    'off_target_correct': 0,
                    'off_target_incorrect': 0,
                    'incoherent': 0,
                    'coherent': 0,
                }

            if not layer_info["coherent"]:
                layer_counts[layer]['incoherent'] += 1
                continue

            layer_counts[layer]['coherent'] += 1

            if layer_info["on_target"]:
                if layer_info["correct"]:
                    layer_counts[layer]['on_target_correct'] += 1
                else:
                    layer_counts[layer]['on_target_incorrect'] += 1
            else:
                if layer_info["correct_in_some_lang"]: # Note that if "correct" is True, then we would have entered the above if statement 
                    # since we set the language to tgt_lang in that case; we would not reach here
                    layer_counts[layer]['off_target_correct'] += 1
                else:
                    layer_counts[layer]['off_target_incorrect'] += 1


    # Prepare data for plotting
    layers = sorted(layer_counts.keys())

    totals = [layer_counts[l]['coherent'] + layer_counts[l]['incoherent'] for l in layers]

    on_target_correct_pct = [layer_counts[l]['on_target_correct'] / total * 100 if total > 0 else 0 for l, total in zip(layers, totals)]
    on_target_incorrect_pct = [layer_counts[l]['on_target_incorrect'] / total * 100 if total > 0 else 0 for l, total in zip(layers, totals)]
    off_target_correct_pct = [-layer_counts[l]['off_target_correct'] / total * 100 if total > 0 else 0 for l, total in zip(layers, totals)]
    off_target_incorrect_pct = [-layer_counts[l]['off_target_incorrect'] / total * 100 if total > 0 else 0 for l, total in zip(layers, totals)]

    # Plotting
    fig, ax = plt.subplots(figsize=(4, 2))
    x = range(len(layers))
    layer_labels = [str(l) for l in layers]

    # Positive bars (on-target), with hatched bars for correct
    ax.bar(x, on_target_correct_pct, label='✔, on-target', color='green', hatch='/////')
    ax.bar(x, on_target_incorrect_pct, bottom=on_target_correct_pct, label='✖, on-target', color='red')

    # Negative bars (off-target in percentage)
    ax.bar(x, off_target_correct_pct, label='✔, off-target', color='blue', hatch='\\\\\\\\\\')
    ax.bar(x, off_target_incorrect_pct, bottom=off_target_correct_pct, label='✖, off-target', color='gold')

    # for i, (on_pct, off_pct) in enumerate(zip(on_target_correct_pct, off_target_correct_pct)):
    #     if on_pct > 0:
    #         ax.text(i, on_pct / 2, f'{on_pct:.1f}%', ha='center', va='center', color='white', fontsize=9)
    #     if off_pct < 0:
    #         ax.text(i, off_pct / 2, f'{-off_pct:.1f}%', ha='center', va='center', color='white', fontsize=9)

    model = {"llama-3.1-8b-it_wt": "Llama-3.1-8B-Instruct", "aya-23-8b_wt_temp-0": "Aya-23-8B"}[exp_key]
    title = f"{model}, {label_map(src_lang, exp_key = exp_key)}-{label_map(tgt_lang, exp_key= exp_key)}"

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    y_ticks = np.arange(-100, 101, 50)  # or any interval you prefer
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{abs(int(tick))}" for tick in y_ticks])
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('% of total inputs')
    ax.set_title(title)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    




project_home = os.path.join(os.path.dirname(__file__), '..', '..')

exp_key = "aya-23-8b_wt_temp-0"
# exp_key = "llama-3.1-8b-it_wt"
model = exp_key.split("-")[0]

analysis_dir = f"{project_home}/analysis/{exp_key}/"
output_dir = f"{project_home}/analysis_figures/evolution_figures"

os.makedirs(analysis_dir, exist_ok=True)


for dir in os.listdir(analysis_dir):
    # if dir not in {"spa_Latn-mar_Deva"}:
    #     continue
    if dir not in {"spa_Latn-eng_Latn", "spa_Latn-ell_Latn", "hin_Deva-uzn_Latn", "hin_Deva-kor_Hang", "tel_Telu-bos_Latn",\
                   "hin_Deva-tur_Latn", "hin_Deva-fra_Latn", "spa_Latn-mar_Deva", "spa_Latn-tam_Taml"}:
        continue
    print(f"Processing {dir}")
    if not os.path.exists(os.path.join(analysis_dir, dir, "layerwise_analysis.json")):
        print(f"Skipping {dir} - no layerwise_analysis.json")
        continue
    with open(os.path.join(analysis_dir, dir, "layerwise_analysis.json"), "r") as f:
        layerwise_analysis = json.load(f)
        if not layerwise_analysis:
            print(f"Skipping {dir} - empty layerwise_analysis.json")
            continue

    src_lang, tgt_lang = dir.split("-")
    # os.makedirs(os.path.join(output_dir, dir), exist_ok=True)
    # plot_aggregated_info(layerwise_analysis, src_lang, tgt_lang, exp_key, os.path.join(output_dir, dir, f"{src_lang}-{tgt_lang}_{model}_layerwise_analysis.pdf"))
    # plot_aggregated_info(layerwise_analysis, src_lang, tgt_lang, exp_key, os.path.join(output_dir, dir, f"{src_lang}-{tgt_lang}_{model}_layerwise_analysis.png"))
    os.makedirs(output_dir, exist_ok=True)
    plot_aggregated_info(layerwise_analysis, src_lang, tgt_lang, exp_key, os.path.join(output_dir, f"{src_lang}-{tgt_lang}_{model}_layerwise_analysis.pdf"))
    plot_aggregated_info(layerwise_analysis, src_lang, tgt_lang, exp_key, os.path.join(output_dir, f"{src_lang}-{tgt_lang}_{model}_layerwise_analysis.png"))
    
