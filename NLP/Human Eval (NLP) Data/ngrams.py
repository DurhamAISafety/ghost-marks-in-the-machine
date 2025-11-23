import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import ngrams, FreqDist, word_tokenize
from nltk.corpus import stopwords
import string
import numpy as np

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

# --- DATA PREPARATION ---
def get_ngrams_counts(filename, row, n):
    try:
        jsonObj = pd.read_json(path_or_buf=filename, lines=True)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return FreqDist(), 0
        
    nested_tokens = jsonObj[row].astype(str).str.lower().apply(word_tokenize).tolist()
    all_tokens = [
        word for sublist in nested_tokens 
        for word in sublist 
        if word not in stop_words and word not in punctuation
    ]

    grams = ngrams(all_tokens, n)
    return FreqDist(grams), len(all_tokens)

def prepare_data(filename, n):
    print("Processing corpora...")
    counts_w, total_w = get_ngrams_counts(filename, "watermarked_model_response", n)
    counts_u, total_u = get_ngrams_counts(filename, "unwatermarked_model_response", n)

    if total_w == 0 or total_u == 0: return pd.DataFrame()

    df_w = pd.Series(counts_w, name='count_watermarked')
    df_u = pd.Series(counts_u, name='count_unwatermarked')

    df = pd.merge(df_w, df_u, left_index=True, right_index=True, how='outer').fillna(0)
    
    df['freq_w'] = (df['count_watermarked'] + 1) / (total_w + 1)
    df['freq_u'] = (df['count_unwatermarked'] + 1) / (total_u + 1)
    
    return df

# --- PLOTTING FUNCTION ---
def plot_minimal_labels_with_similar(df, n):
    if df.empty: return

    # 1. Calculate Deviation
    df['log_diff'] = np.abs(np.log10(df['freq_w']) - np.log10(df['freq_u']))
    
    # 2. Setup Plot
    plt.figure(figsize=(12, 10))
    sns.set_style("whitegrid")
    
    # Define colors
    colors = sns.color_palette("colorblind")
    col_watermarked = colors[1]
    col_unwatermarked = colors[2]
    col_neutral = 'lightgrey'
    col_similar = 'black'

    mask_w = df['freq_w'] > (df['freq_u'] * 1.5) 
    mask_u = df['freq_u'] > (df['freq_w'] * 1.5)
    
    # Plot points
    plt.scatter(df[~(mask_w | mask_u)]['freq_w'], df[~(mask_w | mask_u)]['freq_u'], 
                color=col_neutral, alpha=0.3, s=15, label='Common N-grams')
    plt.scatter(df[mask_u]['freq_w'], df[mask_u]['freq_u'], 
                color=col_unwatermarked, alpha=0.6, s=30, label='Distinctive to Unwatermarked')
    plt.scatter(df[mask_w]['freq_w'], df[mask_w]['freq_u'], 
                color=col_watermarked, alpha=0.6, s=30, label='Distinctive to Watermarked')

    # Diagonal Line
    min_val = min(df['freq_w'].min(), df['freq_u'].min())
    max_val = max(df['freq_w'].max(), df['freq_u'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Equal Frequency')

    # 3. SELECT POINTS TO LABEL
    top_w = df[df['freq_w'] > df['freq_u']].sort_values('log_diff', ascending=False).head(1)
    top_u = df[df['freq_u'] > df['freq_w']].sort_values('log_diff', ascending=False).head(1)
    most_similar = df.sort_values(['log_diff', 'freq_w'], ascending=[True, False]).head(1)
    
    # --- CHANGED SECTION ---
    # Combine them into a list of tuples so we can tag them with a 'type'
    # We now pass the ACTUAL color variables instead of strings like 'darkslategrey'
    points_to_label = [
        (top_w, 'w', col_watermarked),   # Use variable directly
        (top_u, 'u', col_unwatermarked), # Use variable directly
        (most_similar, 'sim', col_similar)
    ]
    # ---------------------

    # 4. LABEL DIRECTLY ON POINTS
    for subset, label_type, color in points_to_label:
        if subset.empty: continue
        
        row = subset.iloc[0]
        ngram = subset.index[0]
        
        txt = " ".join(ngram)
        x, y = row['freq_w'], row['freq_u']
        
        # Determine the vertical offset based on the label type for stacking (optional but helpful)
        if label_type == 'sim':
            # Slightly lower vertical offset for the similar point to avoid the diagonal line
            vertical_offset = -10
            txt += " (Most Similar)"
        elif label_type == 'u':
            # Slightly higher for the unwatermarked distinctive point
            vertical_offset = 10
        else:
            # Default offset for the watermarked distinctive point
            vertical_offset = 0

        # --- MODIFIED SECTION FOR RIGHT ALIGNMENT ---
        # Set xytext to a positive X offset (5 points right) for all labels
        # Set horizontalalignment to 'left' so the text begins after the offset point
        plt.annotate(
            txt, 
            xy=(x, y),
            xytext=(5, vertical_offset), # Positive X offset ensures it starts to the right
            textcoords='offset points',
            horizontalalignment='left',  # Aligns the start of the text to the offset point
            fontsize=10, 
            weight='bold', 
            color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
        )
    
    # 5. Final Formatting
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlabel(f'Frequency in Watermarked Corpus (Log Scale)', fontsize=12)
    plt.ylabel(f'Frequency in Unwatermarked Corpus (Log Scale)', fontsize=12)
    plt.title(f'{n}-Gram Frequency Comparison (Top Outliers + Most Similar)', fontsize=14)
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig("ngram_labels_with_similar.png")
    print("Saved plot to ngram_labels_with_similar.png")
    plt.show()

N_GRAM_SIZE = 1
FILE_PATH = "human_eval.jsonl" 
# Run
df_data = prepare_data(FILE_PATH, N_GRAM_SIZE)
plot_minimal_labels_with_similar(df_data, N_GRAM_SIZE)