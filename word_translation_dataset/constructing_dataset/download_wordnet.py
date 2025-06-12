# Download English WordNet data from the NLTK library
import nltk
from nltk.corpus import wordnet as wn
import os
from random import shuffle, sample


# Download WordNet data
nltk.download('wordnet')
nltk.download('omw-1.4')

# Sample 200 nouns, 200 verbs, 200 adjectives, and 200 adverbs
def sample_wordnet_data():
    # Define the number of samples for each part of speech
    num_samples = 200

    # Get the synsets for each part of speech
    nouns = list(wn.all_synsets(pos=wn.NOUN))
    verbs = list(wn.all_synsets(pos=wn.VERB))
    adjectives = list(wn.all_synsets(pos=wn.ADJ))
    adverbs = list(wn.all_synsets(pos=wn.ADV))

    # Shuffle and sample the synsets
    nouns = sample(nouns, num_samples)
    verbs = sample(verbs, num_samples)
    adjectives = sample(adjectives, num_samples)
    adverbs = sample(adverbs, num_samples)
    
    print(f"Example nouns: {nouns[:5]}")
    print(f"Example verbs: {verbs[:5]}")
    print(f"Example adjectives: {adjectives[:5]}")
    print(f"Example adverbs: {adverbs[:5]}")
    return nouns, verbs, adjectives, adverbs


# sample_wordnet_data()

def get_synsets(word):
    """
    Get the synsets for a word.
    """
    # Check if the word is in WordNet
    if not wn.synsets(word):
        print(f"Word '{word}' not found in WordNet.")
        return None
    return wn.synsets(word)

