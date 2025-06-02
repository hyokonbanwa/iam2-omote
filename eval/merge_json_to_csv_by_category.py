import json
import csv
import os
from typing import Any, Dict, List
import datetime

def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Data loaded from the file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
def flatten_json(y: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
    out = {}
    for k, v in y.items():
        new_key = f"{prefix}-{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_json(v, new_key))
        else:
            out[new_key] = v
    return out

def merge_json_to_csv(json_list: List[dict],json_files: List[str], output_csv: str):
    all_data = []
    all_keys = []

    for data,path in zip(json_list,json_files):
            flat_data = flatten_json(data)
            flat_data['file'] = os.path.basename(path)
            all_keys = flat_data.keys()
            all_data.append(flat_data)

    # all_keys = ['file'] + sorted(k for k in all_keys if k != 'file')
    all_keys = ['file'] + list(all_keys) 

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in all_data:
            writer.writerow({key: row.get(key, '') for key in all_keys})

# 使用例
if __name__ == "__main__":
    json_files = ["/home/omote/experiment_only_program/iam-lmms-finetune/experiment_output/llava-onevision_finetune-llm_guad_5epoch_2025-05-27T10_38_39/eval_output/visa-test_llava-onevision/2025-05-28T16_20_42/visa_score_category-category.json"]#["/data_ssd/iam_model/original/llava-onevision-qwen2-7b-ov-hf/eval_output/visa-test_llava-onevision/2025-05-26T20_44_03/visa_score_category-category.json"]  # 必要に応じて複数パスを追加
    json_data = load_json(json_files[0])
    json_list = []
    category_names = []
    for key, value in json_data.items():
        json_list.append(value)
        category_names.append(key)
        
    
    current_date = datetime.datetime.now().strftime('%Y-%m-%dT%H_%M_%S')
    merge_json_to_csv(json_list,category_names, f'output_{current_date}.csv')
