import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, NllbTokenizer, BitsAndBytesConfig
import sacrebleu
import evaluate

from tqdm import tqdm
import os, sys
import json
from collections import defaultdict
from argparse import ArgumentParser

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.generate_helper_functions import model_generate
from utils.dataset_loader import get_eval_dataset_flores_plus, get_eval_dataset_xlsum
from utils.manage_prompts import format_input_with_prompt
from utils.lang_codes import flores_code_to_langname
from utils.load_model import load_model


def parse_args():
    parser = ArgumentParser()
    # Identifier 
    parser.add_argument("--exp_key", type=str, required=True, help="Experiment key")
    # Input formatting argument
    parser.add_argument("--src_lang", type=str, required=True, help="Source language code")
    parser.add_argument("--tgt_lang", default=None, help="Target language code")
    parser.add_argument("--eval_task", type=str, default=None, help="One of 'mt', 'xlsum', 'open-ended-generation'")
    parser.add_argument("--prompting_strategy", type=str, default="simple", help="Prompting strategy: one of 'simple', 'nolangname' for mt, 'xlsum_eng' for xlsum")
    # Model arguments
    parser.add_argument("--model_name",type=str, required=True, help="Model name")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the a local model checkpoint (adapter)")
    parser.add_argument("--lora", action="store_true", help="Load LoRA model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    # IO arguments
    parser.add_argument("--outputs_dir", type=str, required=True, help="Dir to save outputs")
    parser.add_argument("--results_file", type=str, required=True, help="Path to save results")
    
    return parser.parse_args()


def main(exp_key, src_lang, tgt_lang, model_name, model_path, lora, eval_task, outputs_dir, results_file, batch_size, prompting_strategy):

    '''
    src_lang, tgt_lang : Should be FloRes codes
    model_name : Model name, should be HF identifier
    model_path : Path to a local model checkpoint (adapter)
    lora : If True, load LoRA model
    eval_task : One of 'mt', 'xlsum', 'open-ended-generation'
    outputs_dir : Directory to save the outputs, should specify task and experiment id
    results_file : File to save the results, should specify task and experiment id
    '''
    #--------------------------
    # Loading model
    device = torch.cuda.current_device() if torch.cuda.is_available() else -1
    print(f"Device: {device}")
    print(f"Loading model...")
    model, tokenizer = load_model(model_name, model_path=model_path, lora=lora, tgt_lang=tgt_lang, device=device, quantized=False)
    
    # model.to(device) # No need for models loaded with bits-and-bytes
    model.eval()

    #--------------------------
    # Loading evaluation dataset
    print(f"Loading evaluation dataset...")
    if eval_task == "mt":
        dataset = get_eval_dataset_flores_plus(src_lang, tgt_lang, max_examples=30)
        
    elif eval_task == "xlsum":
        dataset = get_eval_dataset_xlsum(src_lang, max_examples=10)

    inputs = dataset["input"]
    references = dataset["output"]
    ids = dataset["id"]
    formatted_inputs = format_input_with_prompt(inputs=inputs, task=eval_task, prompting_strategy=prompting_strategy, src_lang=src_lang, tgt_lang=tgt_lang)
    print(f"Sample formatted input:\n {formatted_inputs[0]}")

    #--------------------------
    # Generating model responses
    print(f"Generating model responses...")
    model_responses = list()
    for batch in tqdm(dataset.iter(batch_size=batch_size)):
        batch_inputs = batch["input"]
        formatted_inputs = format_input_with_prompt(inputs=batch_inputs, task=eval_task, prompting_strategy=prompting_strategy, src_lang=src_lang, tgt_lang=tgt_lang)
        
        if model_name in {"nllb"}: # No need to format
            tokenized_inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True)
            tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}
            generated_tokens = model.generate(**tokenized_inputs)
            pred = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            model_responses.extend(pred)
        
        elif model_name in {"aya-23-8b", "aya-101", "llama-3.1-8b-instruct", "llama-3.1-8b"}:

            pred = model_generate(model_name, inputs=formatted_inputs, tokenizer=tokenizer, model=model, top_k=0, max_new_tokens=100) 
            # temperature and top_p are set in model_generate according to model defaults
            # if eval_task == "mt":
                # # Choosing first sentence from the generated text
                # pred = [p.split("\n")[0] for p in pred]

            model_responses.extend(pred)

    #--------------------------

    # Print sample for debugging
    print(f"True sentences: {len(references)}")
    print(f"Predicted sentences: {len(model_responses)}")
    print(f"Sample true sentence: {references[:3]}")
    print(f"Sample predicted sentence: {model_responses[:3]}")
    
    #--------------------------
    # Evaluation
    print(f"Evaluating model responses...")
    if eval_task == "mt":
        # Find BLEU score
        scores = defaultdict(dict)
        bleu = sacrebleu.corpus_bleu(model_responses, [references])
        score = bleu.score
        print(f"BLEU score for {src_lang}-{tgt_lang}: {score}")

        # Find chrF score
        chrf = sacrebleu.corpus_chrf(model_responses, [references])
        chrf_score = chrf.score
        print(f"chrF score for {src_lang}-{tgt_lang}: {chrf_score}")

        scores = {"bleu": score, "chrf": chrf_score}
    
    elif eval_task == "xlsum":
        # Find ROUGE-L score
        scores = defaultdict(dict)
        rouge_scorer = evaluate.load("rouge")
        rouge = rouge_scorer.compute(predictions=model_responses, references=references, tokenizer=lambda x: x.split()) # contains rouge1, rouge2, rougeL, rougeLsum
        
        scores = rouge

    print(f"Scores: {scores}")

    #--------------------------
    # Save outputs
    print(f"Saving outputs...")
    output_path = os.path.join(outputs_dir, f"preds_{src_lang[:3]}-{tgt_lang[:3]}.json") 
    outputs = [{"id": id, "input": input, "output": output, "reference": reference} for id, input, output, reference in zip(ids, inputs, model_responses, references)]
    with open(output_path, "w") as f:
        json.dump(outputs, f, indent = 2, ensure_ascii=False)

    #--------------------------
    # Writing results to a file
    ## If the file already exists, load the results and update the results
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)
            results["scores"][f"{src_lang[:3]}-{tgt_lang[:3]}"] = scores
            # results["comet"][lang] = comet_score
    else:
        results = {}
        results["exp_key"] = exp_key
        results["scores"] =  {f"{src_lang[:3]}-{tgt_lang[:3]}": scores}
        results["model_path"] = model_path if model_path else model_name


    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)





if __name__ == "__main__":
    args = parse_args()
    main(args.exp_key, args.src_lang, args.tgt_lang, args.model_name, \
         args.model_path, args.lora, args.eval_task, args.outputs_dir, args.results_file, args.batch_size, args.prompting_strategy)
