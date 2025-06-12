import os, sys
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from word_translation_dataset.wt_dataset import WordTranslationDataset
from utils.lang_codes import LANGS

from tqdm import tqdm

from googletrans import Translator
# {Spanish: es, Hindi: hi, Russian: ru, Indonesian: id, English: en
# Catalan: ca, Galician: gl, Portuguese: pt, Bengali: bn, Marathi: mr, Nepali: ne, Slovak: sk, Serbian: sr
# Croatian: hr, Ukrainian: uk
# }
# https://cloud.google.com/translate/docs/languages

translator = Translator()

def translate_words(words, src, tgt):
    """
    Translate words from source language to target language.
    Returns a dict: {word: translation}.
    """
    translations = {}
    for word in tqdm(words):
        try:
            translation = translator.translate(word, src=src, dest=tgt)
            translations[word] = [translation.text]
        except Exception as e:
            print(f"Error translating {word}: {e}")
    return translations

def main():

    # dataset_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'word_translation_dataset', 'dataset')
    dataset_dir = "/weka/scratch/dkhasha1/nbafna1/data/word_translation_dataset"
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset directory {dataset_dir} does not exist. Please create it with source_words.json.")
    

    for lang in LANGS:
        lang = LANGS[lang]["gtrans_code"]
        if lang == "en":
            continue
     
        if os.path.exists(os.path.join(dataset_dir, f"{lang}_translations.json")):
            print(f"Translations for {lang} already exist. Skipping...")
            continue

        print(f"Translating to {lang}")
        # Get the source words
        wtd = WordTranslationDataset()
        source_words = wtd.get_source_wordlist()

        # Translate the words
        translations = translate_words(source_words, src="en", tgt=lang)

        # Save the translations to a file
        output_file = os.path.join(dataset_dir, f"{lang}_translations.json")
        with open(output_file, 'w') as f:
            json.dump(translations, f, indent=4, ensure_ascii=False)
        print(f"Translations saved to {output_file}")

if __name__ == "__main__":
    main()


