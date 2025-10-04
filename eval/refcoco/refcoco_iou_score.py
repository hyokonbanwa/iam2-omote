import json
import argparse
import os
import datetime
from tqdm import tqdm
import regex as re
from torchvision.ops import box_iou
import torch
from transformers import AutoProcessor
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
    # print(gt_bbox_list)
    # print(pred_bbox_list)
    iou_matrix = box_iou(torch.tensor(gt_bbox_list).float(), torch.tensor(pred_bbox_list).float())
    iou_matrix = torch.nan_to_num(iou_matrix, nan=0.0)  # NaNを0に置き換える
    iou_argsort_matrix = torch.argsort(iou_matrix.flatten(),descending=True).argsort().reshape(iou_matrix.shape)#iouが大きい順にソートしたインデックスを取得
    # print(iou_argsort_matrix)
    # print("-" * 50)
    # print(iou_matrix)
    pred_index_list =  torch.full((len(pred_bbox_list),), False, dtype=torch.bool)
    gt_index_list = torch.full((len(gt_bbox_list),), False, dtype=torch.bool)

    short_index_list = pred_index_list if len(pred_bbox_list) < len(gt_bbox_list) else gt_index_list
    iou_info_list = []

    # print(iou_matrix.numel())
    for i in range(iou_matrix.numel()):
        max_iou_index = torch.where(iou_argsort_matrix == i)
        if not gt_index_list[max_iou_index[0]] and not pred_index_list[max_iou_index[1]]:
            iou_info_list.append( {
                "gt_index": max_iou_index[0].item(),
                "pred_index": max_iou_index[1].item(),
                "iou_value": iou_matrix[max_iou_index].item()
            })
            gt_index_list[max_iou_index[0]] = True
            pred_index_list[max_iou_index[1]] = True
            # print(f"index {i} - gt_index: {max_iou_index[0].item()}, pred_index: {max_iou_index[1].item()}, iou_value: {iou_matrix[max_iou_index].item()}")
        
        if torch.all(short_index_list):
            break
        
    assert len(iou_info_list) == min(len(gt_bbox_list), len(pred_bbox_list)), f"Length mismatch: {len(iou_info_list)} != {min(len(gt_bbox_list), len(pred_bbox_list))}"
    # print(iou_info_list)
    # for iou_info in iou_info_list:
    #     if math.isnan(iou_info["iou_value"]):
    #         print(f"IOU value is NaN for gt index {iou_info['gt_index']} and pred index {iou_info['pred_index']}")
    #         print(iou_matrix[iou_info['gt_index'], iou_info['pred_index']])
    #         print(iou_matrix[iou_info['gt_index'], iou_info['pred_index']].item())
    #         print(iou_info["iou_value"])
    #         print(iou_matrix)
    
    return iou_info_list,iou_matrix,iou_argsort_matrix,pred_index_list, gt_index_list

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


def kosmos2_get_bbox_and_label(processor, text,*args, **kwargs):
    _, entities = processor.post_process_generation(text)
    bbox_list = [entity[-1] for entity in entities]
    bbox_list = []
    label_list = []
    for entity in entities:
        bbox_list.extend(entity[-1])
        label_list.extend(entity[0] for _ in entity[-1])
    return bbox_list, label_list


def paligemma_get_bbox(text: str,*args, **kwargs):
    pattern = r"(((<loc\d{4}>){4}))"
    matches = re.findall(pattern, text)
    # print("matches", matches)
    bbox_list = []
    for m in matches:
        y1, x1, y2, x2 = [int(x)/1023.0 for x in re.findall(r'\d+', m[1])]
        bbox_list.append([x1, y1, x2, y2])
    return bbox_list, []

def calculate_score(correct_data, generated_data,args,current_date):
    all_iou_list = []
    generated_iou_list = []

    gt_iou_num_count= 0
    matched_data_num = 0

    iou_threshold = 0.5

    if "kosmos-2" in args.model:
        processor = AutoProcessor.from_pretrained("/data_ssd/huggingface_model_weights/microsoft/kosmos-2-patch14-224")
        get_bbox_func = kosmos2_get_bbox_and_label
    elif "paligemma" in args.model:
        processor = None
        get_bbox_func = paligemma_get_bbox
    else:
        raise ValueError(f"Unknown model: {args.model}")

    eval_dict ={}

    for i in range(len(correct_data)):
        assert correct_data[i]["id"] == generated_data[i]["id"], f"ID mismatch at index {i}."
        ann_id = correct_data[i]["ann_id"]
        if  ann_id not in eval_dict:
            correct_bbox = correct_data[i]["gt_entities_quantized_normalized"][0][-1]
            if type(correct_bbox[0]) != list:
                correct_bbox = [correct_bbox]
            eval_dict[ann_id] = {
                "ann_id": ann_id,
                "gt_name": correct_data[i]["gt_entities_quantized_normalized"][0][0],
                "correct_data": correct_bbox,
                "generated_data": []
            }
        # eval_dict[ann_id]["correct_data"].append(correct_data[i]["gt_entities_quantized_normalized"][0][-1])
        output_text = generated_data[i]["conversations"][0]["value"]+generated_data[i]["conversations"][1]["value"]
        
        # for e in entities:
        #     if e[0] == eval_dict[ann_id]["gt_name"]:
        #         generated_bbox = e[0][-1]
        #         break
        #print(generated_data[i]["conversations"][0]["value"]+generated_data[i]["conversations"][1]["value"])
        bbox_list, label_list = get_bbox_func(processor=processor, text=output_text)
        generated_bbox = bbox_list[0] if len(bbox_list) > 0 else None
        
        if generated_bbox is not None:
            eval_dict[ann_id]["generated_data"].append(generated_bbox)

    total_data_num = len(eval_dict)
        
    for eval_item in eval_dict.values():
        correct_bbox = eval_item["correct_data"]
        generated_bbox = eval_item["generated_data"]
        # import pdb; pdb.set_trace()
        gt_iou_num_count += len(correct_bbox)
        
        if len(generated_bbox) == 0:
            iou_list = [0.0] * len(correct_bbox)
        else:
            # try :
            #     iou_info_list ,_,_,_,_ = calculate_iou(correct_bbox, generated_bbox)
            # except Exception as e:
            #     import pdb; pdb.set_trace()
            iou_info_list ,_,_,_,_ = calculate_iou(correct_bbox, generated_bbox)
            iou_list = [iou_info["iou_value"] for iou_info in iou_info_list]
            # if iou_list[0] < 1.0:
            #     import pdb; pdb.set_trace()
            generated_iou_list.extend(iou_list)
            if len(iou_list) < len(correct_bbox):
                iou_list.extend([0.0] * (len(correct_bbox) - len(iou_list)))
                
        all_iou_list.extend(iou_list)
        iou_threshold_count = sum(1 for iou in iou_list if iou >= iou_threshold)
        if iou_threshold_count > 0:
            matched_data_num += 1
        # else:
        #     print(f"Warning: No IoU above threshold {iou_threshold} for ann_id {eval_item}.")

    assert len(all_iou_list) == gt_iou_num_count, f"Length of all_iou_list {len(all_iou_list)} does not match gt_iou_num_count {len(gt_iou_num_count)}."


    print("-" * 50)
    print(f"Total data number: {total_data_num}")
    print(f"Matched data number: {matched_data_num}")
    print(f"len all_iou_list: {len(all_iou_list)}")
    print(f"len generated_iou_list: {len(generated_iou_list)}")
    print("-" * 50)
    accuracy = matched_data_num / total_data_num
    print(f"Accuracy: {accuracy}")
    mean_all_iou = sum(all_iou_list) / len(all_iou_list) if len(all_iou_list) > 0 else 0
    print(f"Mean IoU: {mean_all_iou}")
    # mean_generated_iou = sum(generated_iou_list) / len(generated_iou_list) if len(generated_iou_list) > 0 else 0
    # print(f"Mean Generated IoU: {mean_generated_iou}")
    print("-" * 50)
    
    output_data = {
                "filename": args.generated_json,
                "correct_json": args.gt_json,
                "timestamp": current_date,
                "summary_scores": {
                    "accuracy": accuracy,
                    "mean_iou": mean_all_iou,
                },
                "data_num": {
                    "total_data_num": total_data_num,
                    "matched_data_num": matched_data_num,
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
    
    gt_json_path = "/data_ssd/refcoco_plus/refcoco_plus_kosmos2_validation.json"

    parser = argparse.ArgumentParser(description="Utility functions for JSON handling and sorting.")
    parser.add_argument("-json","--generated_json", type=str, help='Path to the generated JSON file to load or save.')
    parser.add_argument("--model", type=str, help='model-name',default="kosmos-2-patch14-224")
    parser.add_argument("--output_path", type=str, help='Path to save the sorted JSON file.',default=None)
    parser.add_argument("-gt","--gt_json", type=str, help='Path to the ground truth JSON file to load for sorting.', default=gt_json_path )
    args = parser.parse_args()
    main(args)
