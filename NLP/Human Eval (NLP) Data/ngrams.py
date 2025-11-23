import pandas as pd
import matplotlib.pyplot as plt
from nltk import ngrams, FreqDist, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from math import log
import string
import nltk
import os # For file check/creation

# --- SETUP AND CONSTANTS ---

# Note: You may need to uncomment the following lines to download NLTK data 
# if you run this in a new environment, especially if you encountered errors before.
# try:
#     stopwords.words('english')
# except LookupError:
#     nltk.download('punkt', quiet=True)
#     nltk.download('stopwords', quiet=True)

# Define necessary global variables
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    # Fallback to a small list if NLTK fails
    stop_words = set(['the', 'a', 'an', 'is', 'and', 'of', 'in', 'to', 'for', 'on', 'with'])
    print("Warning: NLTK stopwords failed to load. Using a limited list.")

stop_words.add("'s")
punctuation = set(string.punctuation)

N_GRAM_SIZE = 3
FILE_PATH = "human_eval.jsonl" 


# --- DATA PREPARATION FUNCTIONS ---

def get_ngrams_counts(filename, row, n):
    """
    Processes text from a JSONL file, filters, and returns a FreqDist 
    object containing n-gram counts and the total number of tokens.
    """
    # 1. Load and Tokenize
    try:
        jsonObj = pd.read_json(path_or_buf=filename, lines=True)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return FreqDist(), 0
        
    nested_tokens = jsonObj[row].astype(str).str.lower().apply(word_tokenize).tolist()

    # 2. Filter Tokens
    all_tokens = [
        word for sublist in nested_tokens 
        for word in sublist 
        if word not in stop_words and word not in punctuation
    ]

    # 3. Generate N-grams and FreqDist
    grams = ngrams(all_tokens, n)
    fdist = FreqDist(grams)
    
    # Calculate the total number of n-grams for normalization
    total_grams = len(all_tokens) - n + 1 if len(all_tokens) >= n else 0
    
    return fdist, total_grams

def prepare_ngram_data_for_scatter(filename, n):
    """
    Retrieves and merges n-gram frequencies from two columns, 
    calculating normalized frequencies.
    """
    # Get counts for both corpora
    print("Processing watermarked corpus...")
    counts_w, total_w = get_ngrams_counts(filename, "watermarked_model_response", n)
    
    print("Processing unwatermarked corpus...")
    counts_u, total_u = get_ngrams_counts(filename, "unwatermarked_model_response", n)

    if total_w == 0 or total_u == 0:
        print("Error: One or both corpora are empty after filtering. Cannot compute frequencies.")
        return pd.DataFrame()

    # Convert FreqDist to Series and merge
    df_w = pd.Series(counts_w, name='count_watermarked')
    df_u = pd.Series(counts_u, name='count_unwatermarked')

    # Merge based on n-gram (index)
    df_merged = pd.merge(
        df_w, df_u, 
        left_index=True, 
        right_index=True, 
        how='outer'
    ).fillna(0)
    
    # Calculate Relative Frequency (normalized by corpus size)
    df_merged['freq_watermarked'] = df_merged['count_watermarked'] / total_w
    df_merged['freq_unwatermarked'] = df_merged['count_unwatermarked'] / total_u
    
    return df_merged


# --- SCATTERPLOT FUNCTION (FIXED with simpler data coordinate offset) ---

def plot_ngram_comparison(df_merged, n, top_n_outliers=10):
    """
    Generates a scatterplot comparing normalized n-gram frequencies 
    and labels the top N most distinctive n-grams using simple data coordinate offsets.
    """
    if df_merged.empty:
        print("Cannot plot: DataFrame is empty.")
        return

    # Use log scale for better visualization, adding +1 to avoid log(0)
    x = df_merged['freq_watermarked']
    y = df_merged['freq_unwatermarked']
    
    # Calculate log-transformed values
    x_log = [log(val + 1) for val in x]
    y_log = [log(val + 1) for val in y]
    
    plt.figure(figsize=(12, 9)) # Larger figure
    
    # 1. Plot the N-grams
    plt.scatter(x_log, y_log, s=15, alpha=0.6, color='darkblue', label=f'{len(df_merged)} unique {n}-grams')

    # 2. Draw the diagonal (y=x) line for parity
    max_val = max(max(x_log), max(y_log))
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='Equal Frequency')
    
    # --- FIX: Set plot limits explicitly to create space for annotations ---
    margin = max_val * 0.2 
    plt.xlim(-0.05, max_val + margin)
    plt.ylim(-0.05, max_val + margin)
    # --- END FIX ---

    # 3. Identify and Label Top Outliers
    
    # Define simple data coordinate offsets
    # We will alternate the text position to prevent stacking on the y=x line.
    data_offsets = [
        (0.01, 0.01), # Small positive shift (upper right)
        (-0.01, 0.01), # Small negative shift (upper left)
        (0.01, -0.01), # Small positive shift (lower right)
        (-0.01, -0.01), # Small negative shift (lower left)
    ] * 3 # Repeat to cover 10 labels

    df_merged['diff'] = abs(df_merged['freq_watermarked'] - df_merged['freq_unwatermarked']) 
    top_diff_ngrams = df_merged.sort_values('diff', ascending=False).head(top_n_outliers)

    for i, ngram in enumerate(top_diff_ngrams.index):
        if i >= len(data_offsets):
            break

        # Get the log coordinates of the specific outlier point
        top_x = log(df_merged.loc[ngram, 'freq_watermarked'] + 1)
        top_y = log(df_merged.loc[ngram, 'freq_unwatermarked'] + 1)
        
        offset_x, offset_y = data_offsets[i]

        # Use DATA coordinates for both xy (point) and xytext (label position)
        plt.annotate(
            text=' '.join(ngram), 
            xy=(top_x, top_y), 
            # Place the text slightly offset in data coordinates
            xytext=(top_x + offset_x, top_y + offset_y), 
            textcoords='data', # CHANGE: Force text placement using data coordinates
            arrowprops=dict(facecolor='green', shrink=0.05, width=1, headwidth=5),
            color='green',
            fontsize=9
        )
    
    # 4. Set labels and title
    plt.xlabel(f'Log(Relative Frequency in Watermarked Corpus + 1)')
    plt.ylabel(f'Log(Relative Frequency in Unwatermarked Corpus + 1)')
    plt.title(f'N={n} Gram Frequency Comparison (Top {top_n_outliers} Outliers Labeled)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# Run the analysis and plotting, specifying 10 outliers
df_plot_data = prepare_ngram_data_for_scatter(FILE_PATH, N_GRAM_SIZE)
plot_ngram_comparison(df_plot_data, N_GRAM_SIZE, top_n_outliers=10)