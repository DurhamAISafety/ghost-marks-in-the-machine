import json

def find_samples():
    with open("results.json", "r") as f:
        data = json.load(f)
    
    for problem in data:
        results = problem.get("results", [])
        
        none_code = None
        ngram5_code = None
        
        for res in results:
            if res["ngram_len"] == "None" and res["generated_code"].strip():
                none_code = res["generated_code"]
            elif res["ngram_len"] == "5" and res["generated_code"].strip():
                ngram5_code = res["generated_code"]
        
        if none_code and ngram5_code:
            print(f"Found Problem ID: {problem['problem_id']}")
            print("--- Unwatermarked Code ---")
            print(none_code)
            print("--- Ngram=5 Code ---")
            print(ngram5_code)
            return

if __name__ == "__main__":
    find_samples()
