from utils.lang_codes import flores_code_to_xlsumname, code_to_langname

task_registry = {"mt", "word-translation", "xlsum", "open-ended-generation", "judge-generation"}

def get_prompt_template(task, prompting_strategy):

    if task == "mt":
        strategies = {
                "nolangname": "Translate into English: {input}",
                "simple": "Translate the following sentence from {src_lang} to {tgt_lang}:\n{input}\nTranslation:\n",
            }
        return strategies[prompting_strategy]

    elif task == "word-translation":
        strategies = {
                "simple": "Translate the following word from {src_lang} to {tgt_lang}. Respond with a single word.\nWord: {input}\nTranslation: ",
                # "simple": "Give a one-word translation of the following word from {src_lang} to {tgt_lang}. \nWord: {input}\nTranslation: ",
            }
        return strategies[prompting_strategy]
    
    elif task == "xlsum":

        strategies = {
                "simple": "Write a one-sentence summary of the following text in {src_lang}.\nArticle: {input}\nSummary: \n",
                # If we want to summarize in English
                "xlsum_eng": "Write a one-sentence summary in English of the following {src_lang} text.\nArticle: {input}\nSummary:\n",
            }
        return strategies[prompting_strategy]


    elif task == "open-ended-generation":
        raise NotImplementedError
    
    elif task == "judge-generation":
        #         '''Please act as an impartial judge and evaluate the quality of the responses provided by two
        # AI assistants to the user question displayed below. You should choose the assistant that
        # follows the user’s instructions and answers the user’s question better. Your evaluation
        # should consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
        # and level of detail of their responses. Begin your evaluation by comparing the two
        # responses and provide a short explanation. Avoid any position biases and ensure that the
        # order in which the responses were presented does not influence your decision. Do not allow
        # the length of the responses to influence your evaluation. Do not favor certain names of
        # the assistants. Be as objective as possible. After providing your explanation, output your
        # final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]"
        # if assistant B is better, and "[[C]]" for a tie.
        # [User Question]
        # {question}
        # [The Start of Assistant A’s Answer]
        # {answer_a}
        # [The End of Assistant A’s Answer]
        # [The Start of Assistant B’s Answer]
        # {answer_b}
        # [The End of Assistant B’s Answer]
        # 
        # OR
        # https://arxiv.org/pdf/2402.07827 pg 89
        # System preamble:
        # You are a helpful following assistant whose goal is to select the preferred (least wrong)
        # output for a given instruction in [LANGUAGE_NAME].
        # Prompt Template:
        # Which of the following answers is the best one for given instruction in <LANGUAGE_NAME>.
        # A good answer should follow these rules:
        # 1) It should be in [LANGUAGE_NAME]
        # 2) It should answer the request in the instruction
        # 3) It should be factually and semantically comprehensible
        # 4) It should be grammatically correct and fluent.
        # Instruction: [INSTRUCTION]
        # Answer (A): [COMPLETION A]
        # Answer (B): [COMPLETION A]
        # FIRST provide a one-sentence comparison of the two answers, explaining which you prefer
        # and why. SECOND, on a new line, state only ‘Answer (A)’ or ‘Answer (B)’ to indicate
        # your choice. If the both answers are equally good or bad, state ‘TIE’. Your response
        # should use the format:
        # Comparison: <one-sentence comparison and explanation>
        # Preferred: <‘Answer (A)’ or ‘Answer (B)’ or ‘TIE’>
        # '''
        raise NotImplementedError
    

def format_input_with_prompt(inputs, task, prompting_strategy, **kwargs):
    """
    Format the input with the prompt template based on the task and prompting strategy.
    Args:
        inputs (list): List of inputs.
        task (str): The task type (e.g., "mt", "xlsum", etc.).
        prompting_strategy (str): The prompting strategy to use.
        **kwargs: Additional keyword arguments for specific tasks. For example, src_lang and tgt_lang for translation tasks.
    Returns:
        list: List of formatted inputs.
    """
    if task not in task_registry:
        raise ValueError(f"Invalid task '{task}'. Supported tasks are: {task_registry}")


    prompt = get_prompt_template(task, prompting_strategy)
    src_lang = kwargs.get("src_lang", None)
    tgt_lang = kwargs.get("tgt_lang", None)

    if task in {"mt", "xlsum", "word-translation"}:
        if task == "xlsum":
            src_lang = flores_code_to_xlsumname(src_lang).capitalize()
            tgt_lang = flores_code_to_xlsumname(tgt_lang).capitalize()
        if task == "word-translation":
            src_lang = code_to_langname(src_lang).capitalize()
            tgt_lang = code_to_langname(tgt_lang).capitalize()

        formatted_inputs = [prompt.format(src_lang=src_lang, tgt_lang=tgt_lang, input=i) for i in inputs]
        return formatted_inputs
    
    elif task == "open-ended-generation":
        raise NotImplementedError
    elif task == "judge-generation":
        raise NotImplementedError