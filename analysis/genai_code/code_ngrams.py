import pandas as pd
from nltk import ngrams, FreqDist, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import keyword

python_keywords = set(keyword.kwlist)
common_builtins = {
    'str', 'list', 'dict', 'tuple', 'set', 'int', 'float', 'bool', 'object',
    'print', 'input', 'open', 'len', 'range', 'sum', 'min', 'max', 'id', 
    'type', 'zip', 'enumerate', 'sorted', 'super', 'next'
}
stopwords = {
    'the', 'and', 'of', 'to', 'solve', 'num', 'end=', 'pos', '-1', '10', '30', '31', 'ans', 'dp', 'word', 'mid', 'map', 'board', 'temp', 'strings', "'s"
}
python_keywords.update(common_builtins)
python_keywords.update(stopwords)

punctuation_stopwords = set(punctuation)
punctuation_stopwords.update(['[', ']', '(', ')', '{', '}', '.', ':', ',', '=', '*', '+', '-', '/', '%', '&', '|', '^', '>', '<', '==', '!=', '+=', "''", "``", "//", "-=", "--"])

def get_ngrams(filename, nested_column, num, n):
    # 1. Read the file into a DataFrame
    try:
        # Use 'utf-8-sig' to handle potential Byte Order Mark (BOM)
        # Ensure lines=True for JSON Lines format
        jsonObj = pd.read_json(path_or_buf=filename, encoding='utf-8-sig')
    except Exception as e:
        print(f"Error reading JSON file '{filename}': {e}. Check file path and JSON validity!")
        return

    # Check if the required 'results' column exists
    if 'results' not in jsonObj.columns:
        print("Error: DataFrame does not contain a 'results' column for normalization.")
        return

    # 2. **FIX & IMPROVEMENT:** Normalize the nested 'results' list
    
    # 2a. Use .apply(lambda x: x if x else [{}]) to ensure empty lists are 
    #     treated as containing a single empty object, which 'explode' can handle 
    #     without erroring or dropping the entire parent row unexpectedly.
    results_series = jsonObj['results'].apply(lambda x: x if isinstance(x, list) and x else [{}])
    
    # 2b. Explode the series to flatten the list items into new rows
    exploded_results = results_series.explode()
    
    # 2c. Normalize the results into a new DataFrame
    results_df = pd.json_normalize(exploded_results)
    
    # Check if the target column exists in the normalized results
    if nested_column not in results_df.columns:
        print(f"Error: Normalized data does not contain the column '{nested_column}'.")
        print(f"Available columns: {results_df.columns.tolist()}")
        return

    # --- N-gram Processing ---
    print("Tokenizing...")
    # Get the code strings and convert to list of tokens
    nested_tokens = (
        results_df[nested_column]
        .astype(str)
        .str.lower()
        .apply(word_tokenize)
        .tolist()
    )

    print("Filtering...")
    all_tokens = [
        word for sublist in nested_tokens 
        for word in sublist 
        if word not in python_keywords and word not in punctuation_stopwords and len(word) > 1
    ]

    grams = ngrams(all_tokens, n)

    fdist = FreqDist(grams)

    print(f"Top {num} {n}-grams for {nested_column}:")
    for gram, count in fdist.most_common(num):
        print(f"{gram}: {count}")


# Execute the function
get_ngrams("results_partial.json", "generated_code", 50, 2)