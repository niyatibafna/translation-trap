"""
This script goes through output logs and extracts characterizing information per output, for word-translation.
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.lang_codes import LANGS
from utils.evaluation_functions import langid, init_word_translation_lexicons, get_set_of_correct_equivalents, check_correctness_for_word_translation
from argparse import ArgumentParser
import glob
import json
from collections import defaultdict
import matplotlib.pyplot as plt

def parse_args():
    parser = ArgumentParser(description="Extract information from output logs.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the output logs.")
    parser.add_argument("--src_lang", type=str, required=True, help="Source language code, NLLB codes")
    parser.add_argument("--tgt_lang", type=str, required=True, help="Target language code, NLLB codes")
    parser.add_argument("--output_dir", type=str, required=True, help="File to save the extracted information.")
    return parser.parse_args()


def extract_info_onebest_exactmatch(outputs, src, refs, trans_equivalents, tgt_lang):
    """
    This takes in the logs of a particular output, and extracts the following information:
    - Language of the output (lang_f) (None if gibberish)
    - Correctness of final generated answer
    - First layer of coherent output that was *not* the final generated answer (None if so such layer) (layer_c)
    - Language of coherent intermediate output (L_I)
    - Correctness of intermediate output regardless of language 
    - Layer of switch to final generated answer (layer_f)

    Outputs: 
    {Layer: generated outputs at that layer (full sequence)}
    """

    # Final layer output
    final_layer_output = outputs[-1]

    per_layer_info = {}
    for layer_key in sorted(list(outputs.keys()), reverse=True):
        layer_output = outputs[layer_key]
        
        # Check if the output is correct
        correct_in_int_lang = {
            tgt_lang: [check_correctness_for_word_translation(layer_output, ref, tgt_lang) for ref in ref_set] 
                for tgt_lang, ref_set in trans_equivalents.items()
        }
        correct_in_int_lang = {tgt_lang: any(correct) for tgt_lang, correct in correct_in_int_lang.items()}

        layer_output_correct_in_some_lang = any(correct_in_int_lang.values())
        int_langs = {k for k, v in correct_in_int_lang.items() if v}

        # Check if the output is correct in the target language
        ## The reason we have this as well as on-target, correct_in_some_lang is for the case where
        ## the output actually matches the target reference, but the lang-id model is off. 
        ## It's also possible that the same word belongs to multiple languages, so we want this explicit check against the reference.
        layer_output_correct_in_target_lang = any(
            check_correctness_for_word_translation(layer_output, ref, tgt_lang) for ref in refs[tgt_lang]
        )

        # Check if output is same as final output
        reached_final_output = check_correctness_for_word_translation(layer_output, final_layer_output, tgt_lang)

        # Check if the output is gibberish
        lang_layer = langid(layer_output.split("\n")[0]) 
        if lang_layer not in LANGS:
            lang_layer = None

        if layer_output_correct_in_target_lang:
            # If the output is correct in the target language, we correct the langid which we assume is wrong
            lang_layer = tgt_lang
        
        if layer_key == -1: # We'll do this iteration first, so this will be set for future iterations
            lang_final_layer = lang_layer

        per_layer_info[layer_key] = {
            "layer_output": layer_output,
            "coherent": lang_layer is not None or layer_output_correct_in_some_lang or layer_output_correct_in_target_lang,
            "lang": lang_layer,
            "on_target": lang_layer == tgt_lang,
            "correct_in_some_lang": layer_output_correct_in_some_lang,
            "intermediate_langs": list(int_langs),
            "correct": layer_output_correct_in_target_lang,
            "src_word": src,
            "refs": list(refs[tgt_lang]),
            "reached_final_lang": lang_layer == lang_final_layer,
            "reached_final_output": reached_final_output,
        }

        # print(f"Layer: {layer_key}")
        # print(f"Output: {layer_output}")
        # print(f"Info: {per_layer_info[layer_key]}")
        
        # print(f"References: {refs}")
        # print(f"Translation equivalents: {trans_equivalents}")
        # print("\n\n\n")


    info = {
        "final_layer_on_target": per_layer_info[-1]["on_target"],
        "final_layer_correct_in_some_lang": per_layer_info[-1]["correct_in_some_lang"],
        "final_layer_correct": per_layer_info[-1]["correct"],
        "final_layer_coherent": per_layer_info[-1]["coherent"],
        "src_word": src,
        "refs": list(refs[tgt_lang]),
        "eng_equivalents": list(trans_equivalents["eng_Latn"]) if tgt_lang != "eng_Latn" else list(refs[tgt_lang]),
        "off_target_equivalents": [(lang, list(ref_set)) for lang, ref_set in trans_equivalents.items()],
        "per_layer_info": per_layer_info,
    }

    # print(f"Final layer output: {final_layer_output}")
    # print(f"Final layer language: {lang_final_layer}")
    # print(f"Info: {info}")
    
    return info


def _get_source_words(log_file_content):
    """
    Get the source words from the log file content.
    """
    source_words = set()
    # for j in log_file_content:
    #     word = j["prefix"].split("\n")[1].strip().split(":")[1].strip().lower()
    #     print(f"Prefix: {j['prefix']}")
    #     print(f"Source Word: {word}")
    #     print("\n\n\n")
    #     source_words.add(word)
    for j in log_file_content:
        source_words.add(j["src_word"].strip().lower())

    return source_words


def extract_info_all(src_lang, tgt_lang, log_dir):
    """
    Goes through all log files, gets src, ref, trans_equivalents, src_lang, tgt_lang
    and then calls extract_info_onebest_exactmatch.
    Aggregates the information and saves it to a file.
    """
    # Get layer logs
    print(f"Log directory: {log_dir}")
    log_files = glob.glob(os.path.join(log_dir, f"generations_layer*.jsonl"))
    print(f"Log files: {log_files}")
    layer2log_file = {-(int(os.path.basename(log_file).split(".")[0].split("--")[-1])): log_file for log_file in log_files}

    print(f"Log files: {layer2log_file}")
    # layers = {-1, -12}
    # layer2log_file = {layer: layer2log_file[layer] for layer in layers if layer in layer2log_file}

    # Get source words
    with open(layer2log_file[-1], "r") as f:
        log_file_content = [json.loads(line) for line in f.readlines()]
    source_words = _get_source_words(log_file_content)

    print(f"Number of source words: {len(source_words)}")
    # print(f"Source words: {source_words}")

    # Extract info by source word
    # This is a dict of {source_word: {layer: generated outputs at that layer (full sequence)}}    

    ## This is for doing this using the per step log files, and appending the outputs per step
    outputs = defaultdict(lambda: defaultdict(str))
    # for layer, log_file in layer2log_file.items():
    #     with open(log_file, "r") as f:
    #         log_file_content = [json.loads(line) for line in f.readlines()]
    #         for j in log_file_content:
    #             word = j["prefix"].split("\n")[1].strip().split(":")[1].strip().lower()
    #             if word not in source_words:
    #                 print(f"Word {word} not in source words. Skipping...")
    #                 continue
    #             outputs[word][layer] += j["output"][0]

    for layer, log_file in layer2log_file.items():
        with open(log_file, "r") as f:
            log_file_content = [json.loads(line) for line in f.readlines()]
            for j in log_file_content:
                word = j["src_word"].strip().lower()
                if word not in source_words:
                    print(f"Word {word} not in source words. Skipping...")
                    continue
                outputs[word][layer] = j["intermediate_layer_output"].strip()

    all_info = {}
    for source_word in outputs:
        # Get the reference and translation equivalents
        ref_lexicon = init_word_translation_lexicons(src_lang, {tgt_lang})
        refs = get_set_of_correct_equivalents(source_word, ref_lexicon)
        intermediate_languages = set(LANGS.keys())
        intermediate_languages = intermediate_languages.difference({src_lang, tgt_lang})
        int_lexicon = init_word_translation_lexicons(src_lang, intermediate_languages)
        trans_equivalents = get_set_of_correct_equivalents(source_word, int_lexicon)

        all_info[source_word] = extract_info_onebest_exactmatch(outputs=outputs[source_word], src=source_word, refs=refs, trans_equivalents=trans_equivalents, tgt_lang=tgt_lang)
    
        # print(f"Source word: {source_word}")
        # print(f"Extracted info: {all_info[source_word]}")
        # print("\n")

    # print(f"All info: {all_info}")
    return all_info



def plot_aggregated_info(aggregated_info, output_file):
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
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(layers))
    layer_labels = [str(l) for l in layers]

    # Positive bars (on-target)
    ax.bar(x, on_target_correct_pct, label='On-target Correct', color='green')
    ax.bar(x, on_target_incorrect_pct, bottom=on_target_correct_pct, label='On-target Incorrect', color='red')

    # Negative bars (off-target in percentage)
    ax.bar(x, off_target_correct_pct, label='Off-target Correct (%)', color='green')
    ax.bar(x, off_target_incorrect_pct, bottom=off_target_correct_pct, label='Off-target Incorrect (%)', color='red')

    for i, (on_pct, off_pct) in enumerate(zip(on_target_correct_pct, off_target_correct_pct)):
        if on_pct > 0:
            ax.text(i, on_pct / 2, f'{on_pct:.1f}%', ha='center', va='center', color='white', fontsize=9)
        if off_pct < 0:
            ax.text(i, off_pct / 2, f'{-off_pct:.1f}%', ha='center', va='center', color='white', fontsize=9)


    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('% outputs')
    ax.set_title('On-target vs Off-target Output Analysis by Layer')
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    
    

def main():
    args = parse_args()
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    data_dir = args.data_dir
    output_dir = args.output_dir
    
    # src_lang = "spa_Latn"
    # src_lang = "hin_Deva"
    # tgt_lang = "mar_Deva"

    # data_dir = f"/weka/scratch/dkhasha1/nbafna1/projects/diagnosing_genspace/layer_outputs/aya-23-8b_wt/{src_lang}-{tgt_lang}/"
    # output_dir = f"/home/nbafna1/projects/diagnosing_genspace/analysis/aya-23-8b_wt/{src_lang}-{tgt_lang}/"
    
    # Extract the information
    all_info = extract_info_all(src_lang, tgt_lang, data_dir)

    # Save the information to a file
    with open(f"{output_dir}/layerwise_analysis.json", "w") as f:
        json.dump(all_info, f, indent=4, ensure_ascii=False)

    # Plot the aggregated information
    plot_aggregated_info(all_info, f"{output_dir}/layerwise_analysis.png")


if __name__ == "__main__":
    main()



    

