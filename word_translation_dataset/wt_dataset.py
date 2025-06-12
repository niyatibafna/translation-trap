import os, sys
import json
import random
random.seed(42)  # For reproducibility

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.lang_codes import LANGS


class WordTranslationDataset:
    def __init__(self, data_dir = None):    
        if not data_dir:
            data_dir = "/weka/scratch/dkhasha1/nbafna1/data/word_translation_dataset/dataset"
            
        
        if not os.path.exists(data_dir):
            raise ValueError(f"Dataset directory {data_dir} does not exist. Please create it with save_source_words.py.")
        self.data_dir = data_dir

    def get_source_words_by_pos(self, pos = None, num_words = None) -> list:
        """
        Get English source words from the dataset.
        Returns a list of words, filtered by pos if specified.
        """
        with open(os.path.join(self.data_dir, 'source_words.json'), 'r') as f:
            source_dict = json.load(f)
        
        if pos:
            if pos not in {"nouns", "verbs", "adjectives", "adverbs"}:
                raise ValueError("Invalid part of speech. Choose from: nouns, verbs, adjectives, adverbs.")
            
            word_list = source_dict[pos]
        else:
            # If no pos is specified, return all words
            word_list = sorted([word for words in source_dict.values() for word in words])
        
        if num_words:
            random.shuffle(word_list, random.seed(42))  # For reproducibility
            return word_list[:num_words]
        else:
            return word_list
        
    def get_source_wordlist(self, num_words = None) -> list:
        """
        Get a wordlist balanced across all parts of speech. If num_words is specified,
        it will return a list of that many words, evenly distributed across the parts of speech.
        If num_words is not specified, it will return all words.
        Returns a list of words.
        """
        wordlist = []
        for pos in ["nouns", "verbs", "adjectives", "adverbs"]:
            per_pos = num_words // 4 if num_words else None
            word_list = self.get_source_words_by_pos(pos, per_pos)
            wordlist.extend(word_list)

        return wordlist


        
    def get_lexicon(self, src_lang, tgt_lang, num_words = None) -> dict:
        """
        Get a lexicon for the specified source and target languages.
        src_lang: Source language code with NLLB codes: eng_Latn
        Returns a dictionary of {source_word: [target_word]}.
        """
        src_lang = LANGS[src_lang]["gtrans_code"]
        tgt_lang = LANGS[tgt_lang]["gtrans_code"]

        source_words = self.get_source_wordlist(num_words)

        
        # Get the lexicon for the specified source language with English
        with open(os.path.join(self.data_dir, f"{src_lang}_translations.json"), 'r') as f:
            en_src_lexicon = json.load(f)    

        # Get the lexicon for the specified target language with English
        with open(os.path.join(self.data_dir, f"{tgt_lang}_translations.json"), 'r') as f:
            en_tgt_lexicon = json.load(f)
        
        # Match the source and target lexicons
        lexicon = {}
        for word in source_words:
            src_word = en_src_lexicon[word][0] # we just take the primary translation to maintain number of source words
            if src_word not in lexicon:
                lexicon[src_word.lower()] = []
            tgt_words = en_tgt_lexicon[word]
            for tgt_word in tgt_words:
                if tgt_word not in lexicon[src_word.lower()]:
                    lexicon[src_word.lower()].append(tgt_word)
            

        # Lower case everything in the lexicon
        for word in lexicon:
            lexicon[word] = [w.lower() for w in lexicon[word]]

        return lexicon
                

# from word_translation_dataset.wt_dataset import WordTranslationDataset
# wtd = WordTranslationDataset("/home/nbafna1/projects/diagnosing_genspace/word_translation_dataset/dataset")
# # Example usage
# source_words = wtd.get_source_words_by_pos(pos="verbs")
# print(f"English source words: {source_words[:50]}")
# lexicon = wtd.get_lexicon("eng_Latn", "spa_Latn", num_words=10)
# print(f"Lexicon: {lexicon}")
# lexicon = wtd.get_lexicon("fra_Latn", "hin_Deva", num_words=10)
# print(f"Lexicon: {lexicon}")

        