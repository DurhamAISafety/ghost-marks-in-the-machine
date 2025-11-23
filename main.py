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


# corpus = []

# jsonObj = pd.read_json(path_or_buf="human_eval.jsonl", lines=True)
# corpus.append(" ".join(jsonObj["watermarked_model_response"].astype(str).str.lower().tolist()))
# corpus.append(" ".join(jsonObj["unwatermarked_model_response"].astype(str).str.lower().tolist()))

# tf_idf_results = tf_idf_calculator(corpus)

# df_results = pd.DataFrame(tf_idf_results).T
# df_results.columns = ["watermarked_tfidf", "unwatermarked_tfidf"]
# df_results.fillna(0, inplace=True)
# df_results.index.name = 'word'

# output_filepath = "tfidf_results.csv"
# df_results.to_csv(output_filepath)


### Plotting

import pandas as pd
import matplotlib.pyplot as plt

try:
    df = pd.read_csv("tfidf_results.csv", index_col='word')
except FileNotFoundError:
    print("Error: 'tfidf_results.csv' not found. Please ensure the file was generated in the previous step.")
    exit()

df['total_tfidf'] = df['watermarked_tfidf'] + df['unwatermarked_tfidf']

df_filtered = df[df['total_tfidf'] > 0]

df_sorted = df_filtered.sort_values(by='total_tfidf', ascending=False)

N = 50
df_plot = df_sorted.head(N)

df_plot = df_plot.sort_values(by='total_tfidf', ascending=True)


if not df_plot.empty:
    fig, ax = plt.subplots(figsize=(10, len(df_plot) * 0.7))

    bar_width = 0.35
    words = df_plot.index
    y_pos = range(len(words))

    ax.barh(
        [p - bar_width/2 for p in y_pos],
        df_plot['watermarked_tfidf'],
        bar_width,
        label='Watermarked',
        color='skyblue'
    )

    ax.barh(
        [p + bar_width/2 for p in y_pos],
        df_plot['unwatermarked_tfidf'],
        bar_width,
        label='Unwatermarked',
        color='lightcoral'
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(words)

    ax.set_xlabel('TF-IDF Score')
    ax.set_title(f'Top {N} Word TF-IDF Scores Comparison')
    ax.legend(loc='lower right')
    plt.tight_layout()

    plt.savefig("tfidf_top_n_comparison_plot.png")

    print(f"Plot saved to tfidf_top_n_comparison_plot.png showing top {N} words.")
else:
    print(f"No words found with a non-zero TF-IDF score to plot for the top {N}.")