import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, NllbTokenizer, BitsAndBytesConfig
from peft import PeftModel 
import sacrebleu
from datasets import load_dataset, Dataset

from tqdm import tqdm
import os, sys
import json
from argparse import ArgumentParser
from peft import PeftModel

from collections import defaultdict

from utils.lang_codes import isocode_to_nllbcode

key2hfpath = {
    "nllb": "facebook/nllb-200-distilled-600M",
    "aya-23-8b": "CohereForAI/aya-23-8b",
    "aya-101": "CohereForAI/aya-101",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct"
}

def load_model(model_name, quantized=True, **kwargs):
    f'''
    model_name : Model name, one of {key2hfpath.keys()} or {key2hfpath.values()}
    model_path : Path to a local model checkpoint (adapter). This is only if we are loading some local model
    lora : If True, load LoRA model
    tgt_lang : Target language code, only for nllb model
    device : Device to load the model on, only for aya-101 model
    '''

    # model_name could be either a key or a HF path
    MODEL_NAME = key2hfpath[model_name] if model_name in key2hfpath else model_name
    assert MODEL_NAME in key2hfpath.values(), f"Model {model_name} not supported. Supported models are: nllb, aya-23-8b, aya-101, llama-3.1-8b, llama-3.1-8b-instruct"
    
    device = kwargs.get("device", None)
    if device is None:
        device = torch.cuda.current_device() if torch.cuda.is_available() else -1

    if MODEL_NAME == "facebook/nllb-200-distilled-600M":
        tgt_lang = kwargs.get("tgt_lang", None)
        tgt_lang = isocode_to_nllbcode(tgt_lang)

        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        tokenizer = NllbTokenizer.from_pretrained(MODEL_NAME, src_lang="eng_Latn", tgt_lang=isocode_to_nllbcode[tgt_lang])
        
    
    elif MODEL_NAME == "CohereForAI/aya-23-8b":

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if not quantized:
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
            model.to(device)
            return model, tokenizer
        
        QUANTIZE_4BIT = True
        USE_GRAD_CHECKPOINTING = True
        TRAIN_MAX_SEQ_LENGTH = 512
        USE_FLASH_ATTENTION = False
        GRAD_ACC_STEPS = 2
        quantization_config = None
        if QUANTIZE_4BIT:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        attn_implementation = None
        if USE_FLASH_ATTENTION:
            attn_implementation="flash_attention_2"
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            )

        # Local model path
        model_path = kwargs.get("model_path", None)
        if model_path:
            lora = kwargs.get("lora", False)
            assert lora
            model.load_adapter(model_path)
        
        return model, tokenizer

    elif MODEL_NAME == "CohereForAI/aya-101":

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if not quantized:
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            model.to(device)
            return model, tokenizer

        QUANTIZE_4BIT = True
        USE_GRAD_CHECKPOINTING = True
        TRAIN_MAX_SEQ_LENGTH = 512
        USE_FLASH_ATTENTION = False
        GRAD_ACC_STEPS = 2

        quantization_config = None
        if QUANTIZE_4BIT:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        attn_implementation = None
        if USE_FLASH_ATTENTION:
            attn_implementation="flash_attention_2"

        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            )
        
        # if model_path:
        #     model.load_adapter(model_path)

        # model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # model.to(device)

        return model, tokenizer

    elif MODEL_NAME in {"meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"}:

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.padding_side = "left"
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.bos_token
            tokenizer.pad_token_id = tokenizer.bos_token_id
        if not quantized:
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
            model.config.pad_token_id = tokenizer.bos_token_id
            model.to(device)
            return model, tokenizer
        
        QUANTIZE_4BIT = True
        USE_GRAD_CHECKPOINTING = True
        TRAIN_MAX_SEQ_LENGTH = 512
        USE_FLASH_ATTENTION = False
        GRAD_ACC_STEPS = 2

        quantization_config = None
        if QUANTIZE_4BIT:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        attn_implementation = None
        if USE_FLASH_ATTENTION:
            attn_implementation="flash_attention_2"
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            )

        # Local model path
        model_path = kwargs.get("model_path", None)
        if model_path:
            lora = kwargs.get("lora", False)
            assert lora
            model.load_adapter(model_path)

        return model, tokenizer

