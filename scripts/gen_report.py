import json
import os
import sys
from pathlib import Path

# Add parent directory to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.report_generator import generate_html_report

def main():
    input_file = "outputs/results/results.json"
    output_file = "outputs/reports/report.html"
    
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
