import argparse
import subprocess
import os
import json
from tqdm import tqdm
import os
import datetime

def load_json(file_path):
    """
    Load a JSON file and return its content as a Python dictionary.

    Parameters:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The content of the JSON file as a dictionary.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def save_jsonl(data, file_path):
    """
    Save a list of dictionaries to a JSON Lines file.

    Parameters:
        data (list): A list of dictionaries to save.
        file_path (str): The path to the output JSON Lines file.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            json.dump(item, file)
            file.write('\n')
            
def convert_json_to_jsonl(input_file, output_file):
    """
    Convert a JSON file to a JSON Lines file.

    Parameters:
        input_file (str): The path to the input JSON file.
        output_file (str): The path to the output JSON Lines file.
    """
    data = load_json(input_file)
    save_jsonl_data = []
    for i in tqdm(range(len(data))):
        item = data[i]
        result_entry = {
            "image": item["image"],
            "question": item["Question"],
            "question_type": item["type"],
            "gpt_answer": item["conversations"][-1]["value"],
            "correct_answer": item["Answer"],
        }
        save_jsonl_data.append(result_entry)
        
    save_jsonl(save_jsonl_data, output_file)
    print(f"Converted {input_file} to {output_file}")
    return output_file
                
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-json', '--answers_json_path', type=str, required=True,
                        help='Path to the JSONL file used in all summary scripts')
    args = parser.parse_args()
    
    input_json_path = args.answers_json_path
    save_folder = os.path.dirname(input_json_path)
    output_jsonl_path = os.path.join(save_folder, f"{os.path.basename(input_json_path).split('.')[0]}_converted.jsonl")
    convert_json_to_jsonl(input_json_path, output_jsonl_path)
      
    # current_date = datetime.datetime.now().strftime('%Y-%m-%dT%H_%M_%S')
    # if not os.path.exists(".mmad_tmp"):
    #     os.makedirs(".mmad_tmp")



    # 実行対象のスクリプト（summary_until.pyは除外）
    scripts = [
        "defect_summary.py",
        "product_summary.py",
        "summary.py"
    ]

    for script in scripts:
        script_path = os.path.join(os.path.dirname(__file__),"helper", script)
        print(f"Executing: {script_path} --answers_json_path {output_jsonl_path}")
        subprocess.run(["python", script_path, "--answers_json_path", output_jsonl_path], check=True)

if __name__ == "__main__":
    main()
