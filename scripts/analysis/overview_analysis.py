"""
This script is for taking the layerwise analysis and creating a summary of the results by source and target.
"""

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

exp_key = "aya-23-8b_wt_temp-0"
# exp_key = "llama-3.1-8b-it_wt"
# exp_key = "llama-3.1-70b-it_wt"
analysis_dir = f"{project_home}/analysis/{exp_key}/"
# analysis_dir = f"/weka/scratch/dkhasha1/nbafna1/projects/diagnosing_genspace/analysis/{exp_key}/"
os.makedirs(analysis_dir, exist_ok=True)


def get_translation_loss_from_layer(layerwise_analysis):
    """
    Find the following:
    max(on-target correct + off-target correct) - final layer on-target correct
    This is saying that if we stopped the model at some internal layer, we would have better performance (in some other language)
    """
    layer_correct = defaultdict(lambda: 0)
    final_layer_correct = 0
    for src_word in layerwise_analysis:
        for layer, layer_info in layerwise_analysis[src_word]["per_layer_info"].items():
            if layer_info["correct"] or layer_info["correct_in_some_lang"]:
                layer_correct[layer] += 1
        final_layer_correct += 1 if layerwise_analysis[src_word]["per_layer_info"]["-1"]["correct"] else 0

    max_layer_correct = max(layer_correct.values())
    translation_loss = max_layer_correct - final_layer_correct
    translation_loss = translation_loss / len(layerwise_analysis)
    return translation_loss


def get_final_layer_correct(layerwise_analysis):
    """
    Find the final layer correct
    """
    final_layer_correct = 0
    for src_word in layerwise_analysis:
        final_layer_correct += 1 if layerwise_analysis[src_word]["per_layer_info"]["-1"]["correct"] else 0
    return final_layer_correct / len(layerwise_analysis)


def get_final_layer_ontarget(layerwise_analysis):
    """
    Find the final layer on-target
    """
    final_layer_ontarget = 0
    for src_word in layerwise_analysis:
        final_layer_ontarget += 1 if layerwise_analysis[src_word]["per_layer_info"]["-1"]["on_target"] else 0
    return final_layer_ontarget / len(layerwise_analysis)


def get_correct_at_any_layer(layerwise_analysis):
    """
    Find the number of words that were correct at any layer in any language
    """
    correct_at_any_layer = {}
    for src_word in layerwise_analysis:
        for layer, layer_info in layerwise_analysis[src_word]["per_layer_info"].items():
            if layer_info["correct"] or layer_info["correct_in_some_lang"]:
                correct_at_any_layer[src_word] = True
                
    return len(correct_at_any_layer) / len(layerwise_analysis)


def get_translation_loss_word(layerwise_analysis):
    """
    Find the following:
    #words that were correct at any non-final layer - #words that were correct on-target in the final layer
    """
    # correct_int_but_not_final = {} # src_word: bool
    # correct_final_but_never_int = {} # src_word: bool
    # correct_final = {} # src_word: bool
    correct_int_but_not_final = 0
    correct_final_but_never_int = 0
    correct_final = 0

    for src_word in layerwise_analysis:
        correct_final_word = layerwise_analysis[src_word]["per_layer_info"]["-1"]["correct"]
        # correct_final[src_word] = correct_final_word
        correct_int = False
        for layer, layer_info in layerwise_analysis[src_word]["per_layer_info"].items():
            if (layer_info["correct_in_some_lang"] and not layer_info["correct"]) and layer != "-1":
            # if (layer_info["correct_in_some_lang"] or layer_info["correct"]) and layer != "-1":
                correct_int = True
        if correct_int and (not correct_final_word):
            # correct_int_but_not_final[src_word] = True
            correct_int_but_not_final += 1
        if correct_final_word and (not correct_int):
            # correct_final_but_never_int[src_word] = True
            correct_final_but_never_int += 1
        

    # Calculate the translation loss
    # translation_loss = len(correct_int_but_not_final) / len(layerwise_analysis)
    # correct_final_but_never_int = len(correct_final_but_never_int) / len(layerwise_analysis)

    translation_loss = correct_int_but_not_final / len(layerwise_analysis)
    correct_final_but_never_int_total = correct_final_but_never_int / len(layerwise_analysis)
    # correct_final_but_never_int_total = correct_final_but_never_int / correct_final

    return translation_loss, correct_final_but_never_int_total

    # translation_loss : of all words that were finally incorrect, how many were correct at some intermediate layer
    # incorrect_final = 0
    # incorrect_final_correct_intermediate = 0
    # for src_word in layerwise_analysis:
    #     final_layer_info = layerwise_analysis[src_word]["per_layer_info"]["-1"]
    #     if not final_layer_info["correct"]:
    #         incorrect_final += 1
    #         for layer, layer_info in layerwise_analysis[src_word]["per_layer_info"].items():
    #             if layer != "-1" and (layer_info["correct_in_some_lang"] or layer_info["correct"]):
    #                 incorrect_final_correct_intermediate += 1
    #                 break
    # # Calculate the translation loss
    # translation_loss = incorrect_final_correct_intermediate / len(layerwise_analysis)




def get_layer_of_translation(layerwise_analysis):
    """
    Find the layer such that the difference between #off-target correct in this layer and on-target in the next layer is maximized
    """
    # layer_off_target_correct = defaultdict(lambda: 0)
    # layer_on_target = defaultdict(lambda: 0)
    # for src_word in layerwise_analysis:
    #     for layer, layer_info in layerwise_analysis[src_word]["per_layer_info"].items():
    #         if not layer_info["on_target"] and layer_info["correct_in_some_lang"]: # if it's off-target and correct
    #             layer_off_target_correct[layer] += 1
    #         if layer_info["on_target"]:
    #             layer_on_target[layer] += 1
    
    # layer_difference = {}
    # for layer in layer_off_target_correct:
    #     if layer == "-1":
    #         continue
    #     next_layer = str(int(layer) + 1)
    #     layer_difference[layer] = layer_on_target[next_layer] - layer_off_target_correct[layer]

    # # Find the layer with the maximum difference
    # # If there are multiple layers with the same difference, return the first one
    # max_layer = max(layer_difference, key=layer_difference.get)
    # max_difference = layer_difference[max_layer] / len(layerwise_analysis)
    # return max_layer, max_difference

    # We do this by finding the layer with the most off-target correct words
    layer_off_target_correct = defaultdict(lambda: 0)
    layer_on_target = defaultdict(lambda: 0)
    for src_word in layerwise_analysis:
        for layer, layer_info in layerwise_analysis[src_word]["per_layer_info"].items():
            if not layer_info["on_target"] and layer_info["correct_in_some_lang"]:
                layer_off_target_correct[layer] += 1
    # Find the layer with the most off-target correct words
    max_layer = max(layer_off_target_correct, key=layer_off_target_correct.get)

    return max_layer, -1

def get_intermediate_languages(layerwise_analysis, tgt_lang):
    """
    Find the intermediate languages for each source word. We ignore the tgt_lang.
    """
    intermediate_languages = defaultdict(lambda: 0)
    for src_word in layerwise_analysis:
        int_lang_set = set()
        for layer, layer_info in layerwise_analysis[src_word]["per_layer_info"].items():
            if layer in {"-1", "-2", "-3", "-4"}:
                continue
            for lang in layer_info["intermediate_langs"]:
                int_lang_set.add(lang)
            if layer_info["correct"]:
                int_lang_set.add(tgt_lang)
        for lang in int_lang_set:
        #     if lang == tgt_lang:
        #         continue
            intermediate_languages[lang] += 1
                
    # Normalize the intermediate languages by the number of source words
    for lang in intermediate_languages:
        intermediate_languages[lang] = intermediate_languages[lang] / len(layerwise_analysis)
    
    return intermediate_languages


def get_target_language_presence_by_layer(layerwise_analysis, tgt_lang):
    """
    Find the intermediate languages for each source word. 
    """
    src_word = list(layerwise_analysis.keys())[0]
    # Missing layers
    for layer in range(-10, 0):
        if str(layer) not in layerwise_analysis[src_word]["per_layer_info"]:
            print(f"WARNING: Layer {layer} not found in layerwise_analysis for {tgt_lang}.")

    tgt_lang_presence_by_layer = defaultdict(lambda: 0)
    for layer in range(-10, 0):
        intermediate_langs = defaultdict(lambda: 0)
        for src_word in layerwise_analysis:
            if str(layer) not in layerwise_analysis[src_word]["per_layer_info"]:
                continue
            layer_info = layerwise_analysis[src_word]["per_layer_info"][str(layer)]
            if layer_info["correct"]:
                intermediate_langs[tgt_lang] += 1
            else:
                if layer_info["intermediate_langs"]:
                    lang = random.choice(layer_info["intermediate_langs"])
                    intermediate_langs[lang] += 1
            
        
        total_intermediate_langs = sum(intermediate_langs.values())
        tgt_lang_presence_by_layer[layer] = intermediate_langs[tgt_lang] / total_intermediate_langs if total_intermediate_langs > 0 else 0
    
    return tgt_lang_presence_by_layer



def format_perc(value):
    """
    Format the value as a percentage with 2 decimal places
    """
    return round(value * 100, 1)

translation_loss = defaultdict(lambda: dict()) # source: {target: loss}
translation_loss_word = defaultdict(lambda: dict()) # source: {target: loss}
translation_final_but_never_int = defaultdict(lambda: dict()) # source: {target: %}
layer_of_translation = defaultdict(lambda: dict()) # source: {target: (layer, difference)}
final_layer_correct = defaultdict(lambda: dict()) # source: {target: correct}
final_layer_ontarget = defaultdict(lambda: dict()) # source: {target: correct}
correct_at_any_layer = defaultdict(lambda: dict()) # source: {target: correct}
intermediate_langs_per_src_tgt = defaultdict(lambda: dict()) # source: {target: {intermediate_lang: %}}
tgt_lang_presence_per_src_tgt = defaultdict(lambda: dict()) # source: {target: {layer: %}}
output_dir = os.path.join(project_home, "analysis", "overview", exp_key)
os.makedirs(output_dir, exist_ok=True)

for dir in os.listdir(analysis_dir):
    print(f"Processing {dir}")
    if not os.path.isdir(os.path.join(analysis_dir, dir)):
        continue
    if not os.path.exists(os.path.join(analysis_dir, dir, "layerwise_analysis.json")):
        print(f"Skipping {dir} - no layerwise_analysis.json")
        continue
    with open(os.path.join(analysis_dir, dir, "layerwise_analysis.json"), "r") as f:
        layerwise_analysis = json.load(f)
        if not layerwise_analysis:
            print(f"Skipping {dir} - empty layerwise_analysis.json")
            continue
        # Get the source and target from the directory name
        source, target = dir.split("-")

        # Get the translation loss and layer of translation
        loss_at_some_layer = format_perc(get_translation_loss_from_layer(layerwise_analysis))
        layer, difference = get_layer_of_translation(layerwise_analysis)
        translation_loss_value, final_but_never_int = get_translation_loss_word(layerwise_analysis)
        loss_by_word = format_perc(translation_loss_value)
        final_but_never_int = 100 - format_perc(final_but_never_int)
        final_layer_correct_value = format_perc(get_final_layer_correct(layerwise_analysis))
        final_layer_ontarget_value = format_perc(get_final_layer_ontarget(layerwise_analysis))
        correct_at_any_layer_value = format_perc(get_correct_at_any_layer(layerwise_analysis))
        intermediate_langs = get_intermediate_languages(layerwise_analysis, target)
        tgt_lang_presence = get_target_language_presence_by_layer(layerwise_analysis, target)
        # Store the results
        translation_loss[source][target] = loss_at_some_layer
        layer_of_translation[source][target] = (layer, format_perc(difference))
        translation_loss_word[source][target] = loss_by_word
        translation_final_but_never_int[source][target] = final_but_never_int
        final_layer_correct[source][target] = final_layer_correct_value
        final_layer_ontarget[source][target] = final_layer_ontarget_value
        correct_at_any_layer[source][target] = correct_at_any_layer_value
        intermediate_langs_per_src_tgt[source][target] = intermediate_langs
        tgt_lang_presence_per_src_tgt[source][target] = tgt_lang_presence
        print(f"Translation loss for {source} to {target}: {loss_at_some_layer}")
        print(f"Layer of translation for {source} to {target}: {layer} ({difference})")
        print(f"Translation loss by word for {source} to {target}: {loss_by_word}")
        print(f"Final layer correct for {source} to {target}: {final_layer_correct_value}")
        print(f"Final layer on-target for {source} to {target}: {final_layer_ontarget_value}")
        print("=====================================")

# Save the results to a file
outfile = os.path.join(output_dir, f"{exp_key}_translation_loss.json")
with open(outfile, "w") as f:
    json.dump(translation_loss, f, indent=4)
outfile = os.path.join(output_dir, f"{exp_key}_layer_of_translation.json")
with open(outfile, "w") as f:
    json.dump(layer_of_translation, f, indent=4)
with open(os.path.join(output_dir, f"{exp_key}_translation_loss_word.json"), "w") as f:
    json.dump(translation_loss_word, f, indent=4)
with open(os.path.join(output_dir, f"{exp_key}_translation_final_but_never_int.json"), "w") as f:
    json.dump(translation_final_but_never_int, f, indent=4)
# Save the final layer correct to a file
with open(os.path.join(output_dir, f"{exp_key}_final_layer_correct.json"), "w") as f:
    json.dump(final_layer_correct, f, indent=4)
# Save the final layer on-target to a file
with open(os.path.join(output_dir, f"{exp_key}_final_layer_ontarget.json"), "w") as f:
    json.dump(final_layer_ontarget, f, indent=4)
# Save the correct at any layer to a file
with open(os.path.join(output_dir, f"{exp_key}_correct_at_any_layer.json"), "w") as f:
    json.dump(correct_at_any_layer, f, indent=4)
# Save the intermediate languages to a file
with open(os.path.join(output_dir, f"{exp_key}_intermediate_langs_per_tgt.json"), "w") as f:
    json.dump(intermediate_langs_per_src_tgt, f, indent=4)
with open(os.path.join(output_dir, f"{exp_key}_tgt_lang_presence_per_src_tgt.json"), "w") as f:
    json.dump(tgt_lang_presence_per_src_tgt, f, indent=4)



def create_heatmap(results, title, filename, cmap="Reds", vmax=100):
    """
    Create a heatmap from the results
    """
    supported, unsupported = lang_order(exp_key)
    tgt_langs = supported + unsupported
    tgt_langs = [map_flores_to_ours(tgt) for tgt in tgt_langs]

    src_langs = ["spa_Latn", "hin_Deva", "tel_Telu"]
    src_langs = [src_lang for src_lang in src_langs if src_lang in results]
    tgt_langs = [tgt_lang for tgt_lang in tgt_langs if tgt_lang in results[src_langs[0]]]
    # Create matrix with source languages as rows and target languages as columns in the order of lang_order
    results = {src: {tgt: results[src].get(tgt, -1) for tgt in tgt_langs} for src in src_langs}
    matrix = pd.DataFrame(results).T

    src_langs = [label_map(src, supported, unsupported) for src in matrix.index]
    tgt_langs = [label_map(tgt, supported, unsupported) for tgt in matrix.columns]

    # Set the index and columns to the new labels
    matrix.index = src_langs
    matrix.columns = tgt_langs
    # Colour NaN values black
    matrix = matrix.replace(-1, np.nan)

    # Create heatmap
    if len(matrix) > 3 and len(matrix.columns) > 3:
        plt.figure(figsize=(10, 10))
    else:
        plt.figure(figsize=(18, 3))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap=cmap, cbar_kws={'label': '(%)', "pad": 0.01}, linewidths=.5, vmax=vmax, vmin=0)
   
    plt.title(title)
    plt.xlabel("Target")
    # plt.ylabel("Source")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def create_heatmap_int_langs(results, int_langs_ordered, title, filename, cmap="Reds"):
    """
    Create a heatmap from the results
    """
    supported, unsupported = lang_order(exp_key)
    tgt_langs_ordered = supported + unsupported
    # tgt_langs = [map_flores_to_ours(tgt) for tgt in tgt_langs]

    # tgt_langs = [tgt_lang for tgt_lang in src_langs if tgt_lang in results]
    # int_langs = [tgt_lang for tgt_lang in tgt_langs if tgt_lang in results[tgt_langs[0]]]
    # # Create matrix with source languages as rows and target languages as columns in the order of lang_order
    # results = {tgt: {int_lang: results[tgt].get(int_lang, -1) for int_lang in int_langs} for tgt in tgt_langs}
    matrix = pd.DataFrame(results).T


    tgt_langs = [label_map(src, supported, unsupported) for src in matrix.index]
    int_langs = [label_map(tgt, supported, unsupported) for tgt in matrix.columns]

    # Set the index and columns to the new labels
    matrix.index = tgt_langs
    matrix.columns = int_langs
    # Colour NaN values black
    matrix = matrix.replace(-1, np.nan)

    matrix = matrix.loc[
        [label_map(tgt, supported, unsupported) for tgt in tgt_langs_ordered],
        [label_map(int_lang, supported, unsupported) for int_lang in int_langs_ordered]
    ]

    plt.figure(figsize=(15, 15))

    sns.heatmap(matrix, annot=True, fmt=".1f", cmap=cmap, cbar_kws={'label': '(%)', 'aspect':40}, linewidths=.5)

    plt.title(title)
    plt.xlabel("Intermediate language")
    plt.ylabel("Target language")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def create_barplot_int_langs(intermediate_langs_per_src_tgt, title, filename):
    """
    Let's aggregate the intermediate languages by target language and create a barplot
    """
    # Create heatmap for intermediate languages
    ## First we collapse intermediate languages into {tgt: {intermediate_lang: %}} by taking the mean of target values over all sources
    intermediate_langs_per_tgt = defaultdict(lambda: defaultdict(lambda: 0))
    for source in intermediate_langs_per_src_tgt:
        for target in intermediate_langs_per_src_tgt[source]:
            for lang in intermediate_langs_per_src_tgt[source][target]:
                intermediate_langs_per_tgt[target][lang] += intermediate_langs_per_src_tgt[source][target][lang]
    # Now we normalize the intermediate languages by the number of source languages
    for target in intermediate_langs_per_tgt:
        for lang in intermediate_langs_per_tgt[target]:
            intermediate_langs_per_tgt[target][lang] = format_perc(intermediate_langs_per_tgt[target][lang] / len(intermediate_langs_per_src_tgt))

    # Normalize into a distribution
    for target in intermediate_langs_per_tgt:
        total_intermediate_langs = sum(intermediate_langs_per_tgt[target].values())
        for lang in intermediate_langs_per_tgt[target]:
            intermediate_langs_per_tgt[target][lang] = format_perc(intermediate_langs_per_tgt[target][lang] / total_intermediate_langs)


    # Now we collapse target languages and create {intermediate_lang: %} by taking the mean of target values
    # intermediate_langs = defaultdict(lambda: 0)
    # for target in intermediate_langs_per_tgt:
    #     for lang in intermediate_langs_per_tgt[target]:
    #         intermediate_langs[lang] += intermediate_langs_per_tgt[target][lang]
    # # Now we normalize the intermediate languages by the number of target languages
    # for lang in intermediate_langs:
    #     intermediate_langs[lang] = intermediate_langs[lang] / len(intermediate_langs_per_tgt)
    
    # We want distribution over intermediate languages over all source and target languages
    intermediate_langs = defaultdict(lambda: 0)
    for source in intermediate_langs_per_src_tgt:
        for target in intermediate_langs_per_src_tgt[source]:
            for lang in intermediate_langs_per_src_tgt[source][target]:
                intermediate_langs[lang] += intermediate_langs_per_src_tgt[source][target][lang]
    
    total_intermediate_langs = sum(intermediate_langs.values())
    for lang in intermediate_langs:
        intermediate_langs[lang] = format_perc(intermediate_langs[lang] / total_intermediate_langs)
        
    # Create barplot for intermediate languages in descending order
    intermediate_langs = sorted(intermediate_langs.items(), key=lambda item: item[1], reverse=True)
    int_lang_ordered = [lang[0] for lang in intermediate_langs]

    intermediate_langs = intermediate_langs[:15]
    supported, unsupported = lang_order(exp_key)

    create_heatmap_int_langs(intermediate_langs_per_tgt, int_lang_ordered, "Distribution over intermediate languages by target language", \
                os.path.join(output_dir, f"{exp_key}_intermediate_langs_by_target.png"), cmap="Blues")
    create_heatmap_int_langs(intermediate_langs_per_tgt, int_lang_ordered, "Distribution over intermediate languages by target language", \
                os.path.join(output_dir, f"{exp_key}_intermediate_langs_by_target.pdf"), cmap="Blues")
    

    title = {"llama-3.1-8b-it_wt": "llama-3.1", "aya-23-8b_wt_temp-0": "aya-23"}[exp_key]

    langs = [label_map(lang[0], supported, unsupported) for lang in intermediate_langs]
    values = [lang[1] for lang in intermediate_langs]
    print(f"Total: {sum(values)}")
    print(f"Intermediate languages: {intermediate_langs}")
    # print(*intermediate_langs, sep="\n")
    plt.figure(figsize=(4.5, 2))
    plt.bar(langs, values)
    plt.title(title)
    # plt.xlabel("Intermediate langs")
    plt.yticks([0,25,50])
    plt.ylabel("(%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def create_plot_tgt_lang_by_lang(tgt_lang_presence_per_src_tgt, title, filename):
    """
    Create a plot with layer on x axis and percentage on y axis
    """
    # First, we need to collapse the tgt_lang_presence_per_src_tgt into {tgt_lang: {layer: %}} by taking the mean of source values
    tgt_lang_presence = defaultdict(lambda: defaultdict(lambda: 0))
    for source in tgt_lang_presence_per_src_tgt:
        for target in tgt_lang_presence_per_src_tgt[source]:
            for layer in tgt_lang_presence_per_src_tgt[source][target]:
                tgt_lang_presence[target][layer] += tgt_lang_presence_per_src_tgt[source][target][layer]
    # Now we normalize the tgt_lang_presence by the number of source languages
    for target in tgt_lang_presence:
        for layer in tgt_lang_presence[target]:
            tgt_lang_presence[target][layer] = format_perc(tgt_lang_presence[target][layer] / len(tgt_lang_presence_per_src_tgt))
    # Now we create a plot with layer on x axis and percentage on y axis
    # We want to plot the tgt_lang_presence for each target language

    num_languages = len(tgt_lang_presence)
    cmap = cm.get_cmap('nipy_spectral')  # You can also try 'hsv', 'viridis', or 'turbo'
    colors = [cmap(i / num_languages) for i in range(num_languages)]

    plt.figure(figsize=(3.5, 2))

    supported, unsupported = lang_order(exp_key)
    tgt_langs = supported + unsupported
    tgt_langs = [map_flores_to_ours(tgt) for tgt in tgt_langs]
    tgt_langs = [tgt_lang for tgt_lang in tgt_langs if tgt_lang in tgt_lang_presence]
    # Let's take every second language
    tgt_langs = tgt_langs[::2]
    # print(tgt_lang_presence_per_src_tgt)
    # print(tgt_lang_presence)
    for idx, target in enumerate(tgt_langs):
        layers = []
        values = []
        for layer in tgt_lang_presence[target]:
            layers.append(int(layer))
            values.append(tgt_lang_presence[target][layer])
        # Sort the layers and values
        layers, values = zip(*sorted(zip(layers, values), key=lambda x: int(x[0])))
        # Convert to int
        plt.plot(layers, values, label=label_map(target, exp_key=exp_key), marker='o', color=colors[idx])
    plt.xlabel("Layer")
    plt.ylabel("Target language presence (%)")
    plt.xticks([i for i in range(-10, 0)])
    model_name = {"llama-3.1-8b-it_wt": "llama-3.1-8b", \
             "llama-3.1-70b-it_wt": "llama-3.1-70b", \
             "aya-23-8b_wt_temp-0": "aya-23-8b"}[exp_key]
    plt.title(model_name)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=3)
    plt.savefig(filename, bbox_inches='tight')
    plt.tight_layout()
    plt.close()

# Create heatmap for translation loss
create_heatmap(translation_loss, "Translation Loss", os.path.join(output_dir, f"{exp_key}_translation_loss.png"))
# Create heatmap for layer of translation loss by word
create_heatmap(translation_loss_word, "% words correct at intermediate layer but not final", \
               os.path.join(output_dir, f"{exp_key}_translation_loss_by_word.png"))

# Create heatmap for translation loss final but never intermediate
create_heatmap(translation_final_but_never_int, "% words found in intermediate layers in non-target languages", \
                os.path.join(output_dir, f"{exp_key}_translation_final_but_never_int.pdf"), cmap="Blues")

# Create heatmap for correct final layer
create_heatmap(final_layer_correct, "Final layer correct", os.path.join(output_dir, f"{exp_key}_final_layer_correct.png"), cmap="Greens")
# Create heatmap for layer of translation
layer_of_translation = {source: {target: int(layer) for target, (layer, _) in targets.items()}
                    for source, targets in layer_of_translation.items()}
create_heatmap(layer_of_translation, "Layer of Translation", os.path.join(output_dir, f"{exp_key}_layer_of_translation.png"), vmax=10, cmap="Blues")

# Create heatmap for final layer on-target
create_heatmap(final_layer_ontarget, "Final layer on-target", os.path.join(output_dir, f"{exp_key}_final_layer_ontarget.pdf"), cmap="Greens")

# Create heatmap for correct at any layer
create_heatmap(correct_at_any_layer, "Correct at any layer", os.path.join(output_dir, f"{exp_key}_correct_at_any_layer.png"), cmap="Greens")

create_barplot_int_langs(intermediate_langs_per_src_tgt, exp_key, os.path.join(output_dir, f"{exp_key}_intermediate_langs.pdf"))
create_barplot_int_langs(intermediate_langs_per_src_tgt, exp_key, os.path.join(output_dir, f"{exp_key}_intermediate_langs.png"))

# Create plot for tgt lang presence by layer
create_plot_tgt_lang_by_lang(tgt_lang_presence_per_src_tgt, "Tgt lang presence by layer", \
                             os.path.join(output_dir, f"{exp_key}_tgt_lang_presence_by_layer.png"))
create_plot_tgt_lang_by_lang(tgt_lang_presence_per_src_tgt, "Tgt lang presence by layer", \
                             os.path.join(output_dir, f"{exp_key}_tgt_lang_presence_by_layer.pdf"))

print("=====================================")
print("Average of words that were final correct but never intermediate: ")
avg_final_but_never_int = 0
lang_pairs = 0
for source in translation_final_but_never_int:
    for target in translation_final_but_never_int[source]:
        avg_final_but_never_int += translation_final_but_never_int[source][target]
        lang_pairs += 1
avg_final_but_never_int /= lang_pairs
print(avg_final_but_never_int)
print("=====================================")
# print(translation_final_but_never_int)

# Let's correlate layer of translation with final layer correct for all languages

# X = []
# Y = []
# annotations = []
# for source in layer_of_translation:
#     for target in layer_of_translation[source]:
#         if layer_of_translation[source][target] != -1 and final_layer_correct[source][target] != -1:
#             X.append(layer_of_translation[source][target])
#             Y.append(final_layer_correct[source][target])
#             annotations.append(f"{source}-{target}")

# plt.figure(figsize=(10, 8))
# plt.scatter(X, Y, alpha=0.5)
# for i, txt in enumerate(annotations):
#     plt.annotate(txt, (X[i], Y[i]), fontsize=8, alpha=0.5)
# plt.xlabel("Layer of Translation")
# plt.ylabel("Final Layer Correct")
# plt.title("Layer of Translation vs Final Layer Correct")
# plt.savefig(os.path.join(project_home, "analysis", "overview", f"{exp_key}_layer_of_translation_vs_final_layer_correct.png"))

# # Let's create a DF with the results

# translation_loss_df = pd.DataFrame(translation_loss).T
# translation_loss_df = translation_loss_df.fillna(-1)
# translation_loss_df = translation_loss_df.rename_axis("source").reset_index()
# translation_loss_df = translation_loss_df.rename_axis("target", axis=1)
# translation_loss_df = translation_loss_df.melt(id_vars=["source"], var_name="target", value_name="translation_loss")

# print(translation_loss_df.head())
# layer_of_translation_df = pd.DataFrame(layer_of_translation).T
# layer_of_translation_df = layer_of_translation_df.fillna(-1)
# layer_of_translation_df = layer_of_translation_df.rename_axis("source").reset_index()
# layer_of_translation_df = layer_of_translation_df.rename_axis("target", axis=1)
# layer_of_translation_df = layer_of_translation_df.melt(id_vars=["source"], var_name="target", value_name="layer_of_translation")

# print(layer_of_translation_df.head())
