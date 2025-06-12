"""
Functions to help us characterize the outputs of the models.
"""
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))  # Adjust the path as needed
from word_translation_dataset.wt_dataset import WordTranslationDataset
from utils.lang_codes import LANGS, map_flores_to_ours, map_ours_to_flores
import fasttext
from huggingface_hub import hf_hub_download

global langid_model
langid_model = None


def _init_global_model():
    """
    Initialize the global model for language identification.
    """
    global langid_model
    model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
    langid_model = fasttext.load_model(model_path)


def langid(text):
    """
    Identify the language of the text.
    """
    global langid_model
    if langid_model is None:
        _init_global_model()
    
    predictions = langid_model.predict(text)
    lang = predictions[0][0].split("__")[-1]

    # Map the language code to our internal representation
    lang = map_flores_to_ours(lang)

    return lang


def init_word_translation_lexicons(src_lang, lang_set=None):
    """
    Initialize the word translation lexicons.
    """
    word_translation_lexicons = {}
    wtd = WordTranslationDataset()

    for tgt_lang in lang_set:
        try:
            lexicon = wtd.get_lexicon(src_lang, tgt_lang)
            word_translation_lexicons[tgt_lang] = lexicon
        except Exception as e:
            print(f"Error loading lexicon for {src_lang} to {tgt_lang}: {e}")
            raise

    return word_translation_lexicons

def get_set_of_correct_equivalents(src, word_translation_lexicons):
    """
    Get the intermediate equivalents of the text.
    This is the text in the set of intermediate languages of interest: 
    English, Spanish, Vietnamese, German, French, Korean, Chinese.
    """
    # global word_translation_lexicons
    # if lang_set is None:
    #     lang_set = list(LANGS.keys())
    # if any([l not in word_translation_lexicons for l in lang_set]):
    #     print("Initializing word translation lexicons...")
    #     if not src_lang:
    #         raise ValueError("Source language must be provided for the first time, if word translation lexicons are not initialized.")
    #     _init_word_translation_lexicons(src_lang, lang_set)
    #     print(f"Language set used: {word_translation_lexicons.keys()}")
    
    equivalents = {}
    for tgt_lang in word_translation_lexicons:
        if src in word_translation_lexicons[tgt_lang]:
            equivalents[tgt_lang] = set(word_translation_lexicons[tgt_lang][src])
        else:
            print(f"WARNING: {src} is not in the lexicon for {tgt_lang}.")
            raise ValueError(f"{src} is not in the lexicon for {tgt_lang}.")
    
    return equivalents

# def get_reference_equivalents(src, src_lang=None, tgt_lang=None):
#     """
#     Get the reference equivalents of the text.
#     This is the text in the set of intermediate languages of interest: 
#     English, Spanish, Vietnamese, German, French, Korean, Chinese.
#     """
#     global word_translation_lexicons
#     if tgt_lang not in word_translation_lexicons:
#         _init_word_translation_lexicons(src_lang, {tgt_lang})
    
#     ref = set(word_translation_lexicons[tgt_lang][src])
#     return ref

def script_is_latin(tgt_lang):
    """
    Check if the script of the word is not Latin.
    """
    # Check if the word contains any non-Latin characters
    if tgt_lang[-4:] in {"Latn"}:
        return True
    return False


def check_correctness_for_word_translation(output, ref, tgt_lang):
    """
    Check if the output is correct for word translation.
    """
    if output.strip().lower() == ref.strip().lower():
        return True
    if ref.strip().lower() in output.strip().lower().split():
        return True
    if len(ref.strip().lower()) >=4  and ref.strip().lower() in output.strip().lower(): 
        # If ref is long enough, we allow it to be a non-space separated substring of the input
        # We don't allow this for short words, because they may be coincidental substrings of other words / gibberish
        return True
    if not script_is_latin(tgt_lang):
        if ref.strip().lower() in output.strip().lower():
            # If the script is not Latin, we allow it to be a non-space separated substring of the input
            return True
    ## TODO: strip accents and diacritics
    return False


