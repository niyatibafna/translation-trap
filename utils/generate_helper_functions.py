import torch

def get_message_format(prompts):
    messages = []

    for p in prompts:
        messages.append(
            [{"role": "user", "content": p}]
        )

    return messages

def instruction_model_generate(
    inputs,
    tokenizer,
    model,
    temperature=0.3,
    top_p=0.75,
    top_k=0,
    max_new_tokens=100,
    return_logits=False,
    ):
    
    # print(f"Example prompt: {prompts[0]}")

    messages = get_message_format(inputs)
    # inputs = [{"role": "user", "content": p} for p in inputs]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        padding=True,
        )
    
    # print(f"Prompt:{tokenizer.decode(input_ids[0])}")
    print(f"input_ids: {inputs}")
    inputs = tokenizer(inputs, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    prompt_padded_len = len(input_ids[0])

    if return_logits:
        with torch.no_grad():
            logits = model(input_ids).logits
            return logits

    # Debugging, length of input
    print(f"Length of input: {prompt_padded_len}")
    gen_tokens = model.generate(
        input_ids,
        attention_mask=attention_mask,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        )

    # get only generated tokens
    gen_tokens = [
        gt[prompt_padded_len:] for gt in gen_tokens
    ]

    gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    return gen_text


def seq2seq_model_generate(
    inputs,
    tokenizer,
    model,
    temperature=None,
    top_p=None,
    top_k=0,
    max_new_tokens=100,
    return_logits=False,
):
    # print(f"Example prompt: {prompts[0]}")
    input_ids = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).input_ids
    input_ids = input_ids.to(model.device)
    # prompt_padded_len = len(input_ids[0])

    if return_logits:
        with torch.no_grad():
            decoder_input_ids = torch.full(
                (input_ids.shape[0], 1),
                tokenizer.pad_token_id,
                dtype=torch.long,
                device=input_ids.device,
            )
            decoder_input_ids = decoder_input_ids.expand(input_ids.shape[0], -1)
            decoder_input_ids = model._shift_right(decoder_input_ids)
            decoder_input_ids = decoder_input_ids.to(model.device)

            logits = model(input_ids, decoder_input_ids=decoder_input_ids).logits
            return logits


    gen_tokens = model.generate(
        input_ids,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        do_sample=True,
    )

    gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    return gen_text


# def llama_generate(
#     inputs,
#     tokenizer,
#     model,
#     temperature=0.3,
#     top_p=0.75,
#     top_k=0,
#     max_new_tokens=100,
#     return_logits=False,
# ):
#     # print(f"Example prompt: {prompts[0]}")

#     input_ids = tokenizer(inputs, return_tensors="pt").input_ids
#     input_ids = input_ids.to(model.device)
#     prompt_padded_len = len(input_ids[0])

#     if return_logits:
#         with torch.no_grad():
#             logits = model(input_ids).logits
#             return logits

#     # Debugging, length of input
#     print(f"Length of input: {prompt_padded_len}")
#     gen_tokens = model.generate(
#         input_ids,
#         temperature=temperature,
#         top_p=top_p,
#         top_k=top_k,
#         max_new_tokens=max_new_tokens,
#         do_sample=True,
#     )

#     # get only generated tokens
#     gen_tokens = [
#         gt[prompt_padded_len:] for gt in gen_tokens
#     ]

#     gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
#     return gen_text


def model_generate(model_name, **kwargs):
    model_dispatch = {
        "aya-23-8b": instruction_model_generate,
        "llama-3.1-8b-instruct": instruction_model_generate,
        "llama-3.1-8b": instruction_model_generate,
        "aya-101": seq2seq_model_generate,
    }

    # Defaults for Aya-23-8b: temperature=0.3, top_p=0.75
    # Defaults for Llama-3.1-8b-instruct: temperature=0.6, top_p=0.9

    if "temperature" not in kwargs:
        if "aya-23-8b" in model_name:
            kwargs["temperature"] = 0.3
        elif "llama-3.1" in model_name:
            kwargs["temperature"] = 0.6
    if "top_p" not in kwargs:
        if "aya-23-8b" in model_name:
            kwargs["top_p"] = 0.75
        elif "llama-3.1" in model_name:
            kwargs["top_p"] = 0.9

    if model_name not in model_dispatch:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    return model_dispatch[model_name](**kwargs)
