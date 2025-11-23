import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import ngrams, FreqDist, word_tokenize
from nltk.corpus import stopwords
import string
import numpy as np
import json

# --- CONFIGURATION ---
K_GRAM_SIZE = 3                 # The size of the data points (3-grams)
COMPARISON_NGRAMS = ["2", "5", "10"] # The watermarking lengths to compare
FILE_PATH = "results_partial.json" 

# --- SETUP ---
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True) 
    stop_words = set(stopwords.words('english'))

stop_words.add("'s")
punctuation = set(string.punctuation)
for char in ['[', ']', ':', '{', '}', '(', ')', ';', '=', '+', '-', '*', '/', '%', '==', '!=', '>', '<', '>=', '<=', 'and', 'or', 'not', 'def', 'if', 'else', 'return', 'print', 'input', 'while', 'for', 'in', 'len']:
     punctuation.add(char)


# --- CORE N-GRAM COUNTING (Fixed to K_GRAM_SIZE=3 implicitly) ---

def get_ngrams_counts(corpus, k):
    """
    Calculates k-gram frequency distribution for a given corpus (list of code strings).
    Note: k is now always K_GRAM_SIZE = 3.
    """
    full_text = " ".join(corpus)

    all_tokens = [
        word for word in word_tokenize(full_text.lower())
        if word not in stop_words and word not in punctuation and word.strip() != ''
    ]

    if not all_tokens:
         return FreqDist(), 0
         
    grams = ngrams(all_tokens, k)
    return FreqDist(grams), len(all_tokens)

def extract_corpora(filename, comparison_ngrams):
    """
    Loads data and extracts generated code only for the specified n-gram lengths.
    """
    try:
        with open(filename, 'r') as f:
            full_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON file '{filename}'.")
        return {}

    watermarked_corpora = {n: [] for n in comparison_ngrams}

    for problem in full_data:
        results = problem.get('results', [])
        
        for res in results:
            code = res.get('generated_code', '')
            ngram_len_str = str(res.get('ngram_len', 'None')) 

            if code and ngram_len_str in watermarked_corpora:
                watermarked_corpora[ngram_len_str].append(code)

    return watermarked_corpora

def prepare_data_for_comparison(corpus_a, corpus_b, label_a, label_b, k_size):
    """
    Prepares the DataFrame for comparison between two corpora based on k-grams.
    """
    print(f"\nProcessing {k_size}-grams for comparison: n={label_a} vs n={label_b}...")

    # Calculate k-gram counts for Corpus A and B
    counts_a, total_a = get_ngrams_counts(corpus_a, k_size)
    counts_b, total_b = get_ngrams_counts(corpus_b, k_size)

    if total_a <= 1 or total_b <= 1: 
        print(f"Skipping comparison due to insufficient token count.")
        return pd.DataFrame()

    # Create Series with appropriate column names
    df_a = pd.Series(counts_a, name=f'count_{label_a}')
    df_b = pd.Series(counts_b, name=f'count_{label_b}')

    # Merge counts
    df = pd.merge(df_a, df_b, left_index=True, right_index=True, how='outer').fillna(0)
    
    # Calculate frequencies with Laplace smoothing
    df[f'freq_{label_a}'] = (df[f'count_{label_a}'] + 1) / (total_a + 1)
    df[f'freq_{label_b}'] = (df[f'count_{label_b}'] + 1) / (total_b + 1)
    
    return df

# --- PLOTTING FUNCTION (Adapted for comparison) ---
def plot_kgram_comparison(df, k_size, label_a, label_b):
    if df.empty: 
        print(f"No data to plot for {label_a} vs {label_b}.")
        return

    # Column names for comparison
    col_freq_a = f'freq_{label_a}'
    col_freq_b = f'freq_{label_b}'

    # 1. Calculate Deviation
    df['log_diff'] = np.abs(np.log10(df[col_freq_a]) - np.log10(df[col_freq_b]))
    
    # 2. Setup Plot
    plt.figure(figsize=(12, 10))
    sns.set_style("whitegrid")
    
    # Define colors
    colors = sns.color_palette("colorblind")
    col_a_distinctive = colors[1]
    col_b_distinctive = colors[2]
    col_neutral = 'lightgrey'
    col_similar = 'black'

    # Define masks for points significantly off the diagonal (e.g., 1.5x)
    mask_a = df[col_freq_a] > (df[col_freq_b] * 1.5) 
    mask_b = df[col_freq_b] > (df[col_freq_a] * 1.5)
    
    # Plot points
    plt.scatter(df[~(mask_a | mask_b)][col_freq_a], df[~(mask_a | mask_b)][col_freq_b], 
                color=col_neutral, alpha=0.3, s=15, label=f'Common {k_size}-grams')
    plt.scatter(df[mask_b][col_freq_a], df[mask_b][col_freq_b], 
                color=col_b_distinctive, alpha=0.6, s=30, label=f'Distinctive to n={label_b}')
    plt.scatter(df[mask_a][col_freq_a], df[mask_a][col_freq_b], 
                color=col_a_distinctive, alpha=0.6, s=30, label=f'Distinctive to n={label_a}')

    # Diagonal Line
    min_val = min(df[col_freq_a].min(), df[col_freq_b].min())
    max_val = max(df[col_freq_a].max(), df[col_freq_b].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Equal Frequency')

    # 3. SELECT POINTS TO LABEL
    top_a = df[df[col_freq_a] > df[col_freq_b]].sort_values('log_diff', ascending=False).head(1)
    top_b = df[df[col_freq_b] > df[col_freq_a]].sort_values('log_diff', ascending=False).head(1)
    most_similar = df.sort_values(['log_diff', col_freq_a], ascending=[True, False]).head(1)
    
    points_to_label = [
        (top_a, 'a', col_a_distinctive, label_a), 
        (top_b, 'b', col_b_distinctive, label_b), 
        (most_similar, 'sim', col_similar, 'sim')
    ]

    # 4. LABEL DIRECTLY ON POINTS
    for subset, label_type, color, n_label in points_to_label:
        if subset.empty: continue
        
        row = subset.iloc[0]
        ngram = subset.index[0]
        
        txt = " ".join(ngram)
        x, y = row[col_freq_a], row[col_freq_b]
        
        vertical_offset = -10
        if label_type == 'sim':
            txt += " (Most Similar)"
        elif label_type == 'b':
            vertical_offset = 10

        plt.annotate(
            txt, 
            xy=(x, y),
            xytext=(5, vertical_offset),
            textcoords='offset points',
            horizontalalignment='left',
            fontsize=10, 
            weight='bold', 
            color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
        )
    
    # 5. Final Formatting
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlabel(f'Frequency in n={label_a} Watermarked Corpus (Log Scale)', fontsize=12)
    plt.ylabel(f'Frequency in n={label_b} Watermarked Corpus (Log Scale)', fontsize=12)
    plt.title(f'{k_size}-Gram Frequency Comparison: n={label_a} vs n={label_b}', fontsize=14)
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    # Save the plot with both n-gram sizes in the filename
    filename = f"{k_size}gram_n{label_a}_vs_n{label_b}.png"
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close() # Close plot to free memory

# --- MAIN EXECUTION ---
watermarked_corpora = extract_corpora(FILE_PATH, COMPARISON_NGRAMS)

# Check if all required corpora were extracted
if len(watermarked_corpora) != len(COMPARISON_NGRAMS) or any(not v for v in watermarked_corpora.values()):
    print(f"Error: Could not find sufficient data for all comparison N-Gram sizes: {COMPARISON_NGRAMS}. Found sizes: {list(watermarked_corpora.keys())}")
else:
    # Define the pairs to compare
    comparison_pairs = [
        ("2", "5"),
        ("5", "10"),
        ("2", "10")
    ]
    
    print(f"Starting comparison of {K_GRAM_SIZE}-grams across watermarked corpora...")
    
    for label_a, label_b in comparison_pairs:
        corpus_a = watermarked_corpora[label_a]
        corpus_b = watermarked_corpora[label_b]
        
        # Prepare data
        df_data = prepare_data_for_comparison(corpus_a, corpus_b, label_a, label_b, K_GRAM_SIZE)
        
        # Plot results
        plot_kgram_comparison(df_data, K_GRAM_SIZE, label_a, label_b)