def generate_html_report(data, filename):
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SynthID Evaluation Report</title>
        <style>
            body { font-family: sans-serif; margin: 20px; background-color: #f9f9f9; }
            .problem { background: white; border: 1px solid #ccc; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .prompt { background-color: #f5f5f5; padding: 15px; white-space: pre-wrap; max-height: 200px; overflow-y: auto; border: 1px solid #ddd; border-radius: 4px; margin-top: 10px; }
            table { border-collapse: collapse; width: 100%; margin-top: 10px; font-size: 14px; }
            th, td { border: 1px solid #ddd; padding: 12px 8px; text-align: left; vertical-align: top; }
            th { background-color: #f2f2f2; font-weight: 600; }
            tr:nth-child(even) { background-color: #f8f8f8; }
            .passed-correct { color: #155724; background-color: #d4edda; font-weight: bold; }
            .passed-wrong { color: #856404; background-color: #fff3cd; font-weight: bold; }
            .failed { color: #721c24; background-color: #f8d7da; }
            .ngram-group { margin-top: 15px; border: 1px solid #e0e0e0; border-radius: 5px; overflow: hidden; background: white; }
            .ngram-summary { background-color: #e9ecef; padding: 10px 15px; cursor: pointer; font-weight: bold; display: flex; justify-content: space-between; align-items: center; }
            .ngram-summary:hover { background-color: #dee2e6; }
            .ngram-content { padding: 10px; }
            
            .attempt-item { border: 1px solid #eee; margin-bottom: 8px; border-radius: 4px; overflow: hidden; }
            .attempt-summary { padding: 10px; cursor: pointer; display: flex; gap: 20px; align-items: center; background: #f8f9fa; }
            .attempt-summary:hover { background: #f0f0f0; }
            .attempt-details { padding: 15px; border-top: 1px solid #eee; background: white; }
            
            .passed-correct { color: #155724; background-color: #d4edda; border-color: #c3e6cb; }
            .passed-wrong { color: #856404; background-color: #fff3cd; border-color: #ffeeba; }
            .failed { color: #721c24; background-color: #f8d7da; border-color: #f5c6cb; }
            
            .detail-section { margin-bottom: 15px; }
            .detail-section h4 { margin: 0 0 5px 0; font-size: 14px; color: #666; }
            
            pre { white-space: pre-wrap; margin: 0; font-family: 'Consolas', 'Monaco', monospace; font-size: 12px; }
            .code-block { max-height: 400px; overflow-y: auto; background: #2d2d2d; color: #f8f8f2; padding: 10px; border-radius: 4px; }
            .output-block { max-height: 200px; overflow-y: auto; background: #f8f9fa; padding: 10px; border: 1px solid #eee; border-radius: 4px; }
            details > summary { cursor: pointer; outline: none; list-style: none; }
            details > summary::-webkit-details-marker { display: none; }
        </style>
    </head>
    <body>
        <h1>SynthID Evaluation Report</h1>
    """
    
    for prob in data:
        html += f"""
        <div class="problem">
            <h2>Problem ID: {prob['problem_id']}</h2>
            <details>
                <summary><strong>Prompt (Click to expand)</strong></summary>
                <div class="prompt">{prob['prompt']}</div>
            </details>
        """
        
        # Group results by ngram_len
        from itertools import groupby
        # Ensure sorted for groupby
        sorted_results = sorted(prob['results'], key=lambda x: str(x['ngram_len']))
        
        for ngram, group in groupby(sorted_results, key=lambda x: x['ngram_len']):
            group_list = list(group)
            total = len(group_list)
            passed_correct = sum(1 for r in group_list if r['status'] == "Passed (Correct)")
            mean_g = sum(r['g_score'] for r in group_list) / total if total > 0 else 0
            
            html += f"""
            <details class="ngram-group" open>
                <summary class="ngram-summary">
                    <span>N-gram: {ngram}</span>
                    <span>Pass (Correct): {passed_correct}/{total} | Mean G-Score: {mean_g:.4f}</span>
                </summary>
                <div class="ngram-content">
            """
            
            for res in group_list:
                status_class = ""
                if res['status'] == "Passed (Correct)":
                    status_class = "passed-correct"
                elif "Passed" in res['status']:
                    status_class = "passed-wrong"
                else:
                    status_class = "failed"
                
                html += f"""
                    <details class="attempt-item">
                        <summary class="attempt-summary {status_class}">
                            <span>Attempt {res['attempts']}</span>
                            <span>Status: {res['status']}</span>
                            <span>G-Score: {res['g_score']:.4f}</span>
                        </summary>
                        <div class="attempt-details">
                            <div class="detail-section">
                                <h4>Output / Error:</h4>
                                <div class="output-block"><pre>{res['output']}</pre></div>
                            </div>
                            <div class="detail-section">
                                <h4>Generated Code:</h4>
                                <div class="code-block"><pre>{res['generated_code']}</pre></div>
                            </div>
                        </div>
                    </details>
                """
            
            html += """
                </div>
            </details>
            """
            
        html += """
        </div>
        """
        
    html += """
    </body>
    </html>
    """
    
    with open(filename, "w") as f:
        f.write(html)
