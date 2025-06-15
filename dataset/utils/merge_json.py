import json
import glob
from typing import List
import argparse
import os

def merge_json_files(input_files: List[str], output_file: str):
    merged_data = []
    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                merged_data.extend(data)
            else:
                raise ValueError(f"{file_path} does not contain a list.")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

# 例: ディレクトリ内の全てのjsonファイルを結合
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple JSON files into one.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input JSON files.')
    parser.add_argument('--output_file', type=str, default='merged.json', help='Output file name for merged JSON.')
    args = parser.parse_args()
    # input_dirのパスを指定して、ディレクトリ内の全てのJSONファイルを取得
    input_files = glob.glob(os.path.join(args.input_dir, '*.json'))
    if not input_files:
        raise ValueError(f"No JSON files found in directory: {args.input_dir}")
    output_file = args.output_file
    merge_json_files(input_files, output_file)

