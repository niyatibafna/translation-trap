#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import os, sys
from pathlib import Path
from typing import List, Tuple
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.manage_prompts import format_input_with_prompt
from word_translation_dataset.wt_dataset import WordTranslationDataset

def gather_last_token(hidden: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    B, _, H = hidden.size()
    idx = lengths.view(B, 1, 1).expand(-1, 1, H)
    return hidden.gather(dim=1, index=idx).squeeze(1)

def topk_from_logits(logits: torch.Tensor, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    topk = torch.topk(log_probs, k=k, dim=-1)
    return topk.indices, topk.values

def log_step(tokenizer: AutoTokenizer, seqs: List[List[int]], hidden_states: torch.Tensor, \
             lm_head, lengths: torch.Tensor, layer: int, batch_src_words: List[str], batch_tgt_words: List[str], \
                next_token_texts: List[str], jsonl_handle, token_idx: int):
    last_hidden = gather_last_token(hidden_states[layer], lengths)
    logits = lm_head(last_hidden)
    top_idx, top_lp = topk_from_logits(logits, k=10)
    top_idx = top_idx.cpu().tolist()
    top_lp = top_lp.cpu().tolist()
    for ids, src, tgt, toks, lps, nt in zip(seqs, batch_src_words, batch_tgt_words, top_idx, top_lp, next_token_texts):
        json.dump({"src_word": src,
                    "tgt_word": tgt,
                   "prefix": tokenizer.decode(ids, skip_special_tokens=True), 
                   "output": [tokenizer.decode(t).strip() for t in toks], 
                   "log_probs": lps, "next_token": nt, "layer": layer, "token_idx": token_idx}, jsonl_handle, ensure_ascii=False)
        jsonl_handle.write("\n")


def postprocess_text(text: str) -> str:
    """Post-process the generated text."""
    aya_tokens = {"<|START_OF_TURN_TOKEN|>", "<|END_OF_TURN_TOKEN|>", "<|USER_TOKEN|>", "<|CHATBOT_TOKEN|>"}
    for token in aya_tokens:
        text = text.replace(token, "")

    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs; two files will be created, one for generations and one for detailed logs per generation step")
    parser.add_argument("--apply_chat_template", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_words", type=int, default=None, help="Maximum number of words to process")
    parser.add_argument("--src_lang", type=str, help="Source language code (NLLB codes e.g. spa_Latn)")
    parser.add_argument("--tgt_lang", type=str, help="Target language code (NLLB codes e.g. fra_Latn)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if "70B" in args.model:
        model = AutoModelForCausalLM.from_pretrained(args.model, output_hidden_states=True, tp_plan="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, output_hidden_states=True).to(device)
    model.eval()
    
    # Get lexicon
    wt_obj = WordTranslationDataset()
    lexicon = wt_obj.get_lexicon(args.src_lang, args.tgt_lang, num_words = args.max_words)
    src_tgt_words = [(k, v[0]) for k, v in lexicon.items()]
    src_words, tgt_words = zip(*src_tgt_words)
    src_words = list(src_words)
    tgt_words = list(tgt_words)

    
    # Debug mode: use a single prompt
    if args.debug:
        src_words = src_words[:1]


    output_generations_file = os.path.join(args.output_dir, f"generations_layer-{args.layer}.jsonl")
    output_all_steps_file = os.path.join(args.output_dir, f"generations_per_step_layer-{args.layer}.jsonl")
    Path(os.path.dirname(output_generations_file) or ".").mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(output_all_steps_file) or ".").mkdir(parents=True, exist_ok=True)
    
    with open(output_generations_file, "w", encoding="utf-8") as generations_out, open(output_all_steps_file, "w", encoding="utf-8") as logs_out:
        for start in tqdm(range(0, len(src_words), args.batch_size), desc="Batches"):
            
            batch_src_words = src_words[start:start + args.batch_size]
            batch_tgt_words = tgt_words[start:start + args.batch_size]
            batch_prompts = format_input_with_prompt(inputs=batch_src_words, task="word-translation", prompting_strategy="simple", src_lang=args.src_lang, tgt_lang=args.tgt_lang)
            
            if args.apply_chat_template:
                proc_prompts = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in batch_prompts]
            else:
                proc_prompts = batch_prompts
            
            seqs: List[List[int]] = [tokenizer.encode(p, add_special_tokens=False) for p in proc_prompts]
            
            layer_seqs: List[List[int]] = [seq.copy() for seq in seqs]
            
            for token_idx in range(args.max_new_tokens):
                
                max_len = max(len(s) for s in seqs)
                input_ids = [s + [tokenizer.pad_token_id] * (max_len - len(s)) for s in seqs]
                attention_mask = [[1] * len(s) + [0] * (max_len - len(s)) for s in seqs]
                input_ids_t = torch.tensor(input_ids, device=device)
                attention_mask_t = torch.tensor(attention_mask, device=device)
                lengths = torch.tensor([len(s) - 1 for s in seqs], device=device)
                
                with torch.no_grad():
                    outs = model(input_ids=input_ids_t, attention_mask=attention_mask_t)
                
                last_logits = outs.logits[range(len(seqs)), lengths]
                
                last_hidden_layer = gather_last_token(outs.hidden_states[args.layer], lengths)
                layer_logits = model.lm_head(last_hidden_layer)
                
                if args.temperature == 0.0:
                    next_token_tensor = torch.argmax(last_logits, dim=-1)
                else:
                    probs = torch.softmax(last_logits / args.temperature, dim=-1)
                    next_token_tensor = torch.multinomial(probs, num_samples=1).squeeze(1)
                
                if args.temperature == 0.0:
                    layer_next_token_tensor = torch.argmax(layer_logits, dim=-1)
                else:
                    layer_probs = torch.softmax(layer_logits / args.temperature, dim=-1)
                    layer_next_token_tensor = torch.multinomial(layer_probs, num_samples=1).squeeze(1)
                
                next_token_texts = [tokenizer.decode(t) for t in next_token_tensor.tolist()]
                
                log_step(tokenizer, seqs, outs.hidden_states, model.lm_head, lengths, args.layer, 
                         batch_src_words, batch_tgt_words, next_token_texts, logs_out, token_idx)
                
                all_eos = True
                for idx, t in enumerate(next_token_tensor.tolist()):
                    seqs[idx].append(t)
                    if t != tokenizer.eos_token_id:
                        all_eos = False
                
                for idx, t in enumerate(layer_next_token_tensor.tolist()):
                    layer_seqs[idx].append(t)
                
                if all_eos:
                    break
            
            for i, (word, tgt) in enumerate(zip(batch_src_words, batch_tgt_words)):
                if i < len(layer_seqs):
                    full_text = tokenizer.decode(seqs[i], skip_special_tokens=True)
                    full_text = postprocess_text(full_text)
                    
                    layer_text = tokenizer.decode(layer_seqs[i], skip_special_tokens=True)
                    layer_text = postprocess_text(layer_text)
                    
                    prompt_part = format_input_with_prompt(
                        inputs=[word], 
                        task="word-translation", 
                        prompting_strategy="simple", 
                        src_lang=args.src_lang, 
                        tgt_lang=args.tgt_lang
                    )[0].strip()
                    
                    last_layer_translation = full_text[len(prompt_part):].strip()
                    layer_translation = layer_text[len(prompt_part):].strip()
                    
                    generations_dict = {
                        "src_word": word,
                        "tgt_word": tgt,
                        "last_layer_output": last_layer_translation,  # Teacher-forced output from last layer
                        "intermediate_layer_output": layer_translation,  # What the specified layer would have generated
                        "layer": args.layer,
                        "prompt": prompt_part,
                    }
                    generations_out.write(json.dumps(generations_dict, ensure_ascii=False) + "\n")
    
    print(f"[Done] Translations saved to {output_generations_file}")
    print(f"[Done] Intermediate predictions saved to {output_all_steps_file}")

if __name__ == "__main__":
    main()
