# human_eval.jsonl is from  https://github.com/google-deepmind/synthid-text
# Reference implementation from 
# Dathathri, S., See, A., Ghaisas, S. et al. Scalable watermarking for identifying large language model outputs. Nature 634, 818–823 (2024). https://doi.org/10.1038/s41586-024-08025-4

import pandas as pd
import string
from nltk import ngrams, FreqDist, word_tokenize
from nltk.corpus import stopwords

# Download necessary NLTK data !once!
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))
stop_words.add("'s")
punctuation = set(string.punctuation)

def get_ngrams(filename, row, num, n):
    jsonObj = pd.read_json(path_or_buf=filename, lines=True)

    print("Tokenizing...")
    nested_tokens = jsonObj[row].astype(str).str.lower().apply(word_tokenize).tolist()

    print("Filtering...")
    all_tokens = [
        word for sublist in nested_tokens 
        for word in sublist 
        if word not in stop_words and word not in punctuation
    ]

    grams = ngrams(all_tokens, n)

    fdist = FreqDist(grams)

    print(f"Top {num} {n}-grams for {row}:")
    for gram, count in fdist.most_common(num):
        print(f"{gram}: {count}")

get_ngrams("human_eval.jsonl", "watermarked_model_response", 10, 3)
get_ngrams("human_eval.jsonl", "unwatermarked_model_response", 10, 3)