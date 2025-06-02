import json
import argparse
import os
import datetime
from tqdm import tqdm

def save_json(file_path, data):
    """
    Save data to a JSON file.

    Args:
        file_path (str): Path to the JSON file.
        data (dict): Data to save.
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(file_path):
    """
    Load data from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Data loaded from the file.
    """
    with open(file_path, 'r') as f:
        return json.load(f)
    
def sort_list_of_dicts(data, key, reverse=False):
    """
    Sort a list of dictionaries by the specified key.

    Args:
        data (list): List of dictionaries to sort.
        key (str): Key to sort by.
        reverse (bool): Sort in descending order if True, ascending if False.

    Returns:
        list: Sorted list of dictionaries.
    """
    return sorted(data, key=lambda x: x[key], reverse=reverse)


def calculate_score(correct_data, generated_data,args,current_date):
    total_data_num = len(correct_data)
    anomaly_data_num = 0
    normal_data_num = 0

    model_predict_anomaly_data_num = 0
    model_predict_normal_data_num = 0

    matched_data_num = 0
    anomaly_matched_data_num = 0



    for i in tqdm(range(total_data_num)):
        assert correct_data[i]["id"] == generated_data[i]["id"], f"ID mismatch at index {i}."
        if correct_data[i]["id"].split('/')[-2] == "Normal":
            normal_data_num += 1
        else:
            anomaly_data_num += 1
            
        A_is_annormaly_text = "A: Yes. B: No."
        B_is_annormaly_text = "A: No. B: Yes."
        
        if A_is_annormaly_text in generated_data[i]["conversations"][0]["value"]:
            if generated_data[i]["conversations"][-1]["value"] == "A" or generated_data[i]["conversations"][-1]["value"] == "A: Yes.":
                model_predict_anomaly_data_num += 1
            else:
                model_predict_normal_data_num += 1
            #正常画像の検出判定
            if (correct_data[i]["conversations"][-1]["value"] == "B") and  (generated_data[i]["conversations"][-1]["value"] == "B" or generated_data[i]["conversations"][-1]["value"] == "B: No."):
                matched_data_num += 1
            # 異常画像の検出判定
            elif (correct_data[i]["conversations"][-1]["value"] =="A") and (generated_data[i]["conversations"][-1]["value"] =="A" or generated_data[i]["conversations"][-1]["value"] =="A: Yes."):
                matched_data_num += 1
                anomaly_matched_data_num += 1
        elif B_is_annormaly_text in generated_data[i]["conversations"][0]["value"]:
            if generated_data[i]["conversations"][-1]["value"] == "B" or generated_data[i]["conversations"][-1]["value"] == "B: Yes.":
                model_predict_anomaly_data_num += 1
            else:
                model_predict_normal_data_num += 1
            #正常画像の検出判定
            if (correct_data[i]["conversations"][-1]["value"] == "A") and  (generated_data[i]["conversations"][-1]["value"] == "A" or generated_data[i]["conversations"][-1]["value"] == "A: No."):
                matched_data_num += 1
            # 異常画像の検出判定
            elif (correct_data[i]["conversations"][-1]["value"] =="B") and (generated_data[i]["conversations"][-1]["value"] =="B" or generated_data[i]["conversations"][-1]["value"] =="B: Yes."):
                matched_data_num += 1
                anomaly_matched_data_num += 1
        else:
            raise ValueError(f"Unexpected conversation value in generated data at index {i}: {generated_data[i]['conversations'][0]['value']}")

    print("-" * 50)
    print(f"Total data number: {total_data_num}")
    print(f"Normal data number: {normal_data_num}")
    print(f"Anomaly data number: {anomaly_data_num}")

    print(f"Model predict normal data number: {model_predict_normal_data_num}")
    print(f"Model predict anomaly data number: {model_predict_anomaly_data_num}")

    print(f"Matched data number: {matched_data_num}")
    print(f"Anomaly matched data number: {anomaly_matched_data_num}")
    print("-" * 50)
    accuracy = matched_data_num / total_data_num
    print(f"Accuracy: {accuracy}")
    precision = anomaly_matched_data_num / model_predict_anomaly_data_num if model_predict_anomaly_data_num > 0 else 0
    print(f"Precision: {precision}")
    recall = anomaly_matched_data_num / anomaly_data_num if anomaly_data_num > 0 else 0
    print(f"Recall: {recall}")
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"F1 Score: {f1_score}")
    print("-" * 50)

    output_data = {
        "filename": args.generated_json,
        "correct_json": args.gt_json,
        "timestamp": current_date,
        "scores": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        },
        "data_num": {
            "total_data_num": total_data_num,
            "normal_data_num": normal_data_num,
            "anomaly_data_num": anomaly_data_num,
            "model_predict_normal_data_num": model_predict_normal_data_num,
            "model_predict_anomaly_data_num": model_predict_anomaly_data_num,
            "matched_data_num": matched_data_num,
            "anomaly_matched_data_num": anomaly_matched_data_num
        },
    }
    return output_data

def main(args):
    base_name = os.path.basename(__file__)
    current_date = datetime.datetime.now().strftime('%Y-%m-%dT%H_%M_%S')
    
    
    if args.output_path is None:
        generated_json_folder = os.path.dirname(args.generated_json)
        output_path = os.path.join(generated_json_folder,f"{base_name.split('.')[0]}.json")
    else:
        output_path = args.output_path
    
    print("Loading JSON data...")
    correct_json_path = args.gt_json
    correct_data = load_json(correct_json_path)

    generated_json_path = args.generated_json
    generated_data = load_json(generated_json_path)

    assert len(correct_data) == len(generated_data), "Length of correct and generated data does not match."

    correct_data = sort_list_of_dicts(correct_data, "id")
    generated_data = sort_list_of_dicts(generated_data, "id")

    output_data = calculate_score(correct_data, generated_data,args,current_date)
    
    print(f"Saving sorted JSON data to \"{output_path}\"...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
     # Save the output data to the specified output path
    save_json(output_path, output_data)

if __name__ == "__main__":
    
    gt_json_path = "/data_ssd/visa/visa-test_multiple-select_llava-onevision.json"

    parser = argparse.ArgumentParser(description="Utility functions for JSON handling and sorting.")
    parser.add_argument("--generated_json", type=str, help='Path to the generated JSON file to load or save.')
    parser.add_argument("--output_path", type=str, help='Path to save the sorted JSON file.',default=None)
    parser.add_argument("--gt_json", type=str, help='Path to the ground truth JSON file to load for sorting.', default=gt_json_path )
    args = parser.parse_args()
    main(args)


