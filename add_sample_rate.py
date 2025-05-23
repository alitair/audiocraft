import json
import os
from pathlib import Path

def process_jsonl(input_file):
    # Get the directory and filename
    input_path = Path(input_file)
    output_file = input_path.parent / f"{input_path.stem}_with_sr{input_path.suffix}"
    
    # Read and process the file
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            # Parse the JSON object
            entry = json.loads(line.strip())
            
            # Add sample_rate field
            entry['sample_rate'] = 48000
            
            # Write the modified entry
            f_out.write(json.dumps(entry) + '\n')
    
    print(f"Processed file written to: {output_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python add_sample_rate.py <input_jsonl_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} does not exist")
        sys.exit(1)
    
    process_jsonl(input_file) 