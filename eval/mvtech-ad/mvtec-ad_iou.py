import json
import argparse
import os
import datetime
from tqdm import tqdm
import regex as re
from torchvision.ops import box_iou
import torch
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

def extract_bbox_from_text(ans):
    pattern = re.compile(r'\[(((0|1)\.(\d){3}\,\s*){3}((0|1)\.(\d){3}))\]')
    match_list = pattern.findall(ans)

    if len(match_list) > 0:
        answer = [list(map(float,match[0].split(","))) for match in match_list]
    else:
        answer = "FAILED"
    return answer

def calculate_iou(gt_bbox_list, pred_bbox_list):
    iou_matrix = box_iou(torch.tensor(gt_bbox_list).float(), torch.tensor(pred_bbox_list).float())
    iou_argsort_matrix = torch.argsort(iou_matrix.flatten(),descending=True).argsort().reshape(iou_matrix.shape)#iouが大きい順にソートしたインデックスを取得
    # print("-" * 50)
    # print(iou_matrix)
    pred_index_list =  torch.full((len(pred_bbox_list),), False, dtype=torch.bool)
    gt_index_list = torch.full((len(gt_bbox_list),), False, dtype=torch.bool)

    iou_info_list = []

    for i in range(len(gt_bbox_list)):
        max_iou_index = torch.where(iou_argsort_matrix == i)
        if not gt_index_list[max_iou_index[0]] and not pred_index_list[max_iou_index[1]]:
            iou_info_list.append( {
                "gt_index": max_iou_index[0].item(),
                "pred_index": max_iou_index[1].item(),
                "iou_value": iou_matrix[max_iou_index].item()
            })
            gt_index_list[max_iou_index[0]] = True
            pred_index_list[max_iou_index[1]] = True
    # print(iou_info_list)
    return iou_info_list

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
    gt_bbox_path = "/data_ssd/mvtec_ad/mvtec_bbox.json"
    gt_bbox_data = load_json(gt_bbox_path)
    gt_bbox_data = {"/".join(["/dataset"]+k.split("/")[-4:]): v for k, v in gt_bbox_data.items()}
    total_data_num = len(correct_data)
    anomaly_data_num = 0
    normal_data_num = 0

    model_predict_anomaly_data_num = 0
    model_predict_normal_data_num = 0

    matched_data_num = 0
    anomaly_matched_data_num = 0

    all_iou_list = []
    generated_iou_list = []

    gt_iou_num_count= 0

    iou_threshold = 0.5

    for i in tqdm(range(total_data_num)):
        assert correct_data[i]["id"] == generated_data[i]["id"], f"ID mismatch at index {i}."
        if correct_data[i]["conversations"][-1]["value"] == "None":
            normal_data_num += 1
        else:
            anomaly_data_num += 1
        
        if "None" in generated_data[i]["conversations"][-1]["value"]:
            
            model_predict_normal_data_num += 1
        else:
            model_predict_anomaly_data_num += 1


        #正常画像の検出判定
        if (correct_data[i]["conversations"][-1]["value"] == "None") and  ("None" in generated_data[i]["conversations"][-1]["value"]):
            matched_data_num += 1
        # 異常画像の検出判定
        elif (correct_data[i]["conversations"][-1]["value"] != "None"):
            correct_bbox = gt_bbox_data[correct_data[i]["id"]]
            generated_bbox = extract_bbox_from_text(generated_data[i]["conversations"][-1]["value"])
            gt_iou_num_count += len(correct_bbox)
            # if len(correct_bbox) >  1:
            #     print(correct_data[i]["id"])
            #     print(correct_bbox)
            #     print(generated_bbox)
            if generated_bbox == "FAILED":
                iou_list = [0.0] * len(correct_bbox)
            else:
                iou_list = [item["iou_value"] for item in calculate_iou(correct_bbox, generated_bbox)]
                generated_iou_list.extend(iou_list)
                if len(iou_list) < len(correct_bbox):
                    iou_list.extend([0.0] * (len(correct_bbox) - len(iou_list)))

            all_iou_list.extend(iou_list)
            iou_threshold_count = sum(1 for iou in iou_list if iou >= iou_threshold)
            if iou_threshold_count > 0:
                matched_data_num += 1
                anomaly_matched_data_num += 1
                
            # if i > 100:
            #     break
            # matched_data_num += 1
            # anomaly_matched_data_num += 1

    assert len(all_iou_list) == gt_iou_num_count, f"Length of all_iou_list {len(all_iou_list)} does not match gt_iou_num_count {len(gt_iou_num_count)}."
    print("-" * 50)
    print(len(all_iou_list))
    print(f"len all_iou_list: {len(all_iou_list)}")
    print(f"len generated_iou_list: {len(generated_iou_list)}")

    mean_all_iou = sum(all_iou_list) / len(all_iou_list) if len(all_iou_list) > 0 else 0
    print(f"Mean IoU: {mean_all_iou}")
    mean_generated_iou = sum(generated_iou_list) / len(generated_iou_list) if len(generated_iou_list) > 0 else 0
    print(f"Mean Generated IoU: {mean_generated_iou}")

    # iou_threshold_count = sum(1 for iou in all_iou_list if iou >= iou_threshold)
    # print(f"Number of IoU >= {iou_threshold}: {iou_threshold_count}")

    # matched_data_num += iou_threshold_count
    # anomaly_matched_data_num = iou_threshold_count if iou_threshold_count > 0 else 0


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
                "f1_score": f1_score,
                "mean_all_iou": mean_all_iou,
                "mean_generated_iou": mean_generated_iou,
            },
            "data_num": {
                "total_data_num": total_data_num,
                "normal_data_num": normal_data_num,
                "anomaly_data_num": anomaly_data_num,
                "model_predict_normal_data_num": model_predict_normal_data_num,
                "model_predict_anomaly_data_num": model_predict_anomaly_data_num,
                "matched_data_num": matched_data_num,
                "anomaly_matched_data_num": anomaly_matched_data_num,
                "gt_iou_num_count": gt_iou_num_count,
                "generated_iou_num_count": len(generated_iou_list),
            },
            "other_info": {
                "iou_threshold": iou_threshold,
            }
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
    
    gt_json_path = "/data_ssd/mvtec_ad/mvtec-test_llava-onevision.json"

    parser = argparse.ArgumentParser(description="Utility functions for JSON handling and sorting.")
    parser.add_argument("--generated_json", type=str, help='Path to the generated JSON file to load or save.')
    parser.add_argument("--output_path", type=str, help='Path to save the sorted JSON file.',default=None)
    parser.add_argument("--gt_json", type=str, help='Path to the ground truth JSON file to load for sorting.', default=gt_json_path )
    args = parser.parse_args()
    main(args)
