from datasets import load_dataset, Dataset
from utils.lang_codes import flores_code_to_langname, flores_code_to_xlsumname


def get_eval_dataset_flores200(src_lang, tgt_lang, max_sents = 100):
    # Old function, loading from dowloaded version of FloRes dataset

    '''Load FloRes dataset for evaluation
    src, tgt: ISO 639-3 codes for source and target languages (matching the FloRes dataset codes)
    '''
    flores_dir = "/export/b08/nbafna1/data/flores200_dataset/"
    inputs_file = f"{flores_dir}/dev/{src_lang}.devtest"
    references_file = f"{flores_dir}/dev/{tgt_lang}.devtest"

    dataset = load_dataset("text", data_files={"source": [inputs_file], \
            "target": [references_file]})

    dataset = Dataset.from_dict({"source": dataset["source"]["text"], "target": dataset["target"]["text"]})
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(max_sents))

    return dataset

def get_eval_dataset_flores_plus(src_lang, tgt_lang, max_examples = None):
    '''
    Load FloRes+ dataset for evaluation
    src, tgt: ISO 639-3 codes for source and target languages (matching the FloRes dataset codes)
    '''
    src_dataset = load_dataset("openlanguagedata/flores_plus", src_lang, split="devtest")
    tgt_dataset = load_dataset("openlanguagedata/flores_plus", tgt_lang, split="devtest")

    # all datasets are aligned by id
    assert src_dataset["id"] == tgt_dataset["id"]
    dataset = Dataset.from_dict({"input": src_dataset["text"], "output": tgt_dataset["text"], "id": src_dataset["id"]})

    if max_examples:
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.select(range(max_examples))
    return dataset


def get_eval_dataset_xlsum(src_lang, max_examples = None):
    '''
    Load XLSum dataset for evaluation
    src: ISO 639-3 codes for source language
    '''
    langname = flores_code_to_xlsumname(src_lang).lower()

    ds = load_dataset("csebuetnlp/xlsum", langname, split="test")
    dataset = Dataset.from_dict({"input": ds["text"], "output": ds["summary"], "id": ds["id"]})

    if max_examples:
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.select(range(max_examples))
    return dataset
    