# human_eval.jsonl is from  https://github.com/google-deepmind/synthid-text
# Reference implementation from 
# Dathathri, S., See, A., Ghaisas, S. et al. Scalable watermarking for identifying large language model outputs. Nature 634, 818–823 (2024). https://doi.org/10.1038/s41586-024-08025-4

import pandas as pd
import string
from nltk import ngrams, FreqDist, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from math import log

# Download necessary NLTK data !once!
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))
stop_words.add("'s")
punctuation = set(string.punctuation)



### N-grams ###

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


# get_ngrams("human_eval.jsonl", "watermarked_model_response", 10, 3)
# get_ngrams("human_eval.jsonl", "unwatermarked_model_response", 10, 3)


### TF-IDF ###
# Salton, G., & Yang, C. S. (1973). On the specification of term values in automatic indexing. Journal of Documentation, 29(4), 351–372. https://doi.org/10.1108/eb026562

def clean_text(text):
    """
    Preprocessing adapted to user's logic:
    1. Lowercase and tokenize
    2. Filter out stop_words and punctuation
    """
    # text_str = str(text).lower() 
    tokens = word_tokenize(text)
    
    return [
        word for word in tokens 
        if word not in stop_words and word not in punctuation
    ]

def compute_tf(tokens):
    """
    Calculates Term Frequency (TF).
    TF = (Frequency of term in doc) / (Total terms in doc)
    """
    tf_dict = {}
    total_tokens = len(tokens)
    if total_tokens == 0:
        return {}
        
    token_counts = Counter(tokens)
    
    for token, count in token_counts.items():
        tf_dict[token] = count / total_tokens
        
    return tf_dict

def compute_idf(corpus_tokens):
    """
    Calculates Inverse Document Frequency (IDF).
    IDF = log(Total documents / Number of documents containing the term)
    """
    idf_dict = {}
    N = len(corpus_tokens)
    
    all_words = set([word for doc in corpus_tokens for word in doc])
    
    for word in all_words:
        num_docs_containing_word = sum(1 for doc in corpus_tokens if word in doc)

        idf_dict[word] = log(N / (num_docs_containing_word))
        
    return idf_dict

def tf_idf_calculator(corpus):
    """
    Main function to compute TF-IDF for a list of text strings.
    """
    tokenized_corpus = [clean_text(doc) for doc in corpus]
    
    idf_scores = compute_idf(tokenized_corpus)
    
    tfidf_results = []
    
    for doc_tokens in tokenized_corpus:
        tf_scores = compute_tf(doc_tokens)
        doc_tfidf = {}
        
        for word, tf_val in tf_scores.items():
            idf_val = idf_scores.get(word, 0)
            doc_tfidf[word] = tf_val * idf_val
            
        tfidf_results.append(doc_tfidf)
        
    return tfidf_results


corpus = []

jsonObj = pd.read_json(path_or_buf="human_eval.jsonl", lines=True)
corpus.append(" ".join(jsonObj["watermarked_model_response"].astype(str).str.lower().tolist()))
corpus.append(" ".join(jsonObj["unwatermarked_model_response"].astype(str).str.lower().tolist()))

print(tf_idf_calculator(corpus))