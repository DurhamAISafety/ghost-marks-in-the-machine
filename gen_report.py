import json
import os
from report_generator import generate_html_report

def main():
    input_file = "results.json"
    output_file = "report.html"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    try:
        print(f"Loading {input_file}...")
        with open(input_file, "r") as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} records.")
        
        print(f"Generating {output_file}...")
        generate_html_report(data, output_file)
        print(f"Successfully generated {output_file}")
        
    except json.JSONDecodeError:
        print(f"Error: {input_file} is not valid JSON.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
