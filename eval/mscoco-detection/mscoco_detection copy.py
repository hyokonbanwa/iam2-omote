import json
import sys
sys.path.append(__file__.rsplit('/', 2)[0])  # Adjust the path to include the parent directory
#print(__file__.rsplit('/', 2)[0])
from eval_utils.custom_oc_cost import get_cmap,get_ot_cost,DetectedInstance
import argparse
import os
import datetime
from tqdm import tqdm
import regex as re
from torchvision.ops import box_iou
import torch
from transformers import AutoProcessor
import imgviz
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
from pycocotools.coco import COCO
from copy import deepcopy

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

def oc_cost(pred_instance_list,tgt_instance_list, alpha=0.5,beta=0.6):
    cmap_func = lambda x, y: get_cmap(x, y, alpha=alpha, beta=beta,label_or_sim="label")
    otc = get_ot_cost(pred_instance_list, tgt_instance_list, cmap_func)
    return otc

def create_images_for_coco(conversation_dataset, image_folder_root="/data_ssd"):
    return_images = []
    num_images = len(conversation_dataset)
    
    for i in tqdm(range(num_images)):
        image_name = conversation_dataset[i]["image"]
        image_path = os.path.join(image_folder_root, image_name)
        image = Image.open(image_path)
        image_height = image.height
        image_width = image.width
        image_info = {
            "id": i,
            "width": image_width,
            "height": image_height,
            "file_name": image_name
        }
        return_images.append(image_info)
    
    return return_images

def create_annotations_for_coco(conversation_dataset,categories,processor):
    return_annotations = {}
    
    num_images = len(conversation_dataset)
    
    id_index = 0
    for i in tqdm(range(num_images)):
        caption, entities = processor.post_process_generation(conversation_dataset[i]["conversations"][1]["value"])
        for name,_,bbox_list in entities:
            for bbox in bbox_list:
                annotation = {
                    "id": id_index,
                    "image_id": i,
                    "category_id": categories[name] if name in categories else categories["unknown"],
                    "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],  # [x, y, width, height]
                    "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                    "iscrowd": 0,
                    "score": 1.0,  # Assuming all annotations are perfect for dummy data
                    "category_name": name,
                    "bbox_xyxy": bbox,  # [x1, y1, x2, y2]
                    "is_unknown": 1 if name not in categories else 0
                }
                
                if i not in return_annotations:
                    return_annotations[i] = []
                return_annotations[i].append(annotation)
                id_index += 1

    return return_annotations

def get_per_image_class_result_and_oc_cost(all_gt_annotations, all_pred_annotations, cat_name2id, iou_threshold=0.5):
    """
    Evaluate per-image and per-category results and calculate occlusion cost.

    Args:
        all_gt_annotations (dict): Ground truth annotations.
        all_pred_annotations (dict): Predicted annotations.
        cat_name2id (dict): Category name to ID mapping.
        iou_threshold (float): IOU threshold for true positives.

    Returns:
        dict: Per-image results and occlusion costs.
    """
    per_image_result_dict = {}
    break_index = 10
    oc_cost_list = []
    for index, gt_per_image_annotation_list in tqdm(all_gt_annotations.items()):
        pred_per_image_annotation_list = all_pred_annotations.get(index, [])
        
        # 画像ごとの評価
        pred_instance_list = [DetectedInstance(
            label=ann["category_id"],
            x1=ann["bbox_xyxy"][0],
            y1=ann["bbox_xyxy"][1],
            x2=ann["bbox_xyxy"][2],
            y2=ann["bbox_xyxy"][3]) for ann in pred_per_image_annotation_list]
        tgt_instance_list = [DetectedInstance(
            label=ann["category_id"],
            x1=ann["bbox_xyxy"][0],
            y1=ann["bbox_xyxy"][1],
            x2=ann["bbox_xyxy"][2],
            y2=ann["bbox_xyxy"][3]) for ann in gt_per_image_annotation_list]
        
        oc_cost_value = oc_cost(pred_instance_list, tgt_instance_list, alpha=0.5, beta=0.6)
        oc_cost_list.append(oc_cost_value)
        
        #画像ごと・カテゴリごとの評価準備
        gt_per_category_dict = {}
        pred_per_category_dict = {}
        per_category_result_dict = {}
        
        for category_id in cat_name2id.values():
            gt_per_category_dict[category_id] = None
            pred_per_category_dict[category_id] = None
            per_category_result_dict[category_id] = None
            
        for annotation in gt_per_image_annotation_list:
            if gt_per_category_dict[annotation["category_id"]] is None:
                gt_per_category_dict[annotation["category_id"]] = []
            gt_per_category_dict[annotation["category_id"]].append(annotation)
        
        for annotation in pred_per_image_annotation_list:
            if pred_per_category_dict[annotation["category_id"]] is None:
                pred_per_category_dict[annotation["category_id"]] = []
            pred_per_category_dict[annotation["category_id"]].append(annotation)
        
        
        for category_id, gt_annotations in gt_per_category_dict.items():
            pred_annotations = pred_per_category_dict[category_id]
            if gt_annotations is None and  pred_annotations is None:
                continue
            
            per_category_result = {
                "iou_list": [],
                "pred_iou_list": [],
                "tp_num": 0,
                "fp_num": 0,
                "fn_num": 0,
            }
            if gt_annotations is None and pred_per_category_dict[category_id] is not None:
                per_category_result["fp_num"] = len(pred_per_category_dict[category_id])
            elif gt_annotations is not None:
                if pred_per_category_dict[category_id] is None:
                    per_category_result["fn_num"] = len(gt_annotations)
                    per_category_result["iou_list"] = [0.0] * len(gt_annotations)
                else: 
                    gt_bbox_list = [ann["bbox_xyxy"] for ann in gt_annotations]
                    pred_bbox_list = [ann["bbox_xyxy"] for ann in pred_annotations]
                    iou_info_list,iou_matrix,iou_argsort_matrix,pred_index_list, gt_index_listt = calculate_iou(gt_bbox_list, pred_bbox_list)
                    assert ((len(gt_bbox_list) < len(pred_bbox_list) and len(iou_info_list) == len(gt_bbox_list)) or (len(gt_bbox_list) >= len(pred_bbox_list) and len(iou_info_list) == len(pred_bbox_list))), f"Length mismatch in category {category_id}, index {index}: len(iou_info_list)={len(iou_info_list)}, len(gt_bbox_list)={len(gt_bbox_list)}, len(pred_bbox_list)={len(pred_bbox_list)}"
                    # if not((len(gt_bbox_list) < len(pred_bbox_list) and len(iou_info_list) == len(gt_bbox_list)) or \
                    #     (len(gt_bbox_list) >= len(pred_bbox_list) and len(iou_info_list) == len(pred_bbox_list))):
                        # print(f"index: {index}, category_id: {category_id}, len(iou_info_list): {len(iou_info_list)}, len(gt_bbox_list): {len(gt_bbox_list)}, len(pred_bbox_list): {len(pred_bbox_list)}")
                        # print(f"pred_bbox_list: {pred_bbox_list}")
                        # print(f"gt_bbox_list: {gt_bbox_list}")
                        # print(f"iou_info_list: {iou_info_list}")
                        # print(f"iou_matrix: {iou_matrix}")
                        # print(f"iou_argsort_matrix: {iou_argsort_matrix}")
                        # print(f"pred_index_list: {pred_index_list}")
                        # print(f"gt_index_list: {gt_index_listt}")
                        # raise ValueError("IOU information length mismatch")
                    iou_list = [info["iou_value"] for info in iou_info_list]
                    per_category_result["pred_iou_list"] = deepcopy(iou_list)
                    for iou in iou_list:
                        assert not math.isnan(iou), f"IOU value is NaN in category {category_id}, index {index}"
                    if len(iou_list) < len(gt_bbox_list):
                        iou_list += [0.0] * (len(gt_bbox_list) - len(iou_list))
                    
                    
                    # for iou in iou_list:
                    #     assert not math.isnan(iou), f"IOU value is NaN in category {category_id}, index {index}"
                    per_category_result["iou_list"] = iou_list
                    tp_num = sum(1 for iou in iou_list if iou >= iou_threshold)
                    per_category_result["tp_num"] = tp_num
                    per_category_result["fp_num"] = len(pred_bbox_list) - tp_num
                    per_category_result["fn_num"] = len(gt_bbox_list) - tp_num
                    # if index == 9 and category_id == 84:
                    #     visualize_bbox(
                    #         os.path.join(image_folder_root, images[index]["file_name"]),
                    #         pred_bbox_list,
                    #         [ann["category_name"] for ann in pred_annotations],
                    #         bbox_is_relative=True,
                    #         with_id=True
                    #     )
                    #     print(per_category_result)
                    #     print(pred_bbox_list == gt_bbox_list)
                    #     print(len(pred_bbox_list), len(gt_bbox_list))
                    #     print(iou_info_list)
            
            per_category_result_dict[category_id] = per_category_result
        
        per_image_result_dict[index] = per_category_result_dict
        # if index >= break_index:
        #     break
    return per_image_result_dict, oc_cost_list 

def convert_per_class_result_dict(per_image_result_dict):
    per_category_result_dict = {}
    for index, per_image_result in per_image_result_dict.items():
        for category_id, result in per_image_result.items():
            if category_id not in per_category_result_dict:
                per_category_result_dict[category_id] = {
                    "iou_list": [],
                    "pred_iou_list": [],
                    "tp_num": 0,
                    "fp_num": 0,
                    "fn_num": 0,
                }
            
            if result is None:
                continue
            # if result["fp_num"] > 0 or result["fn_num"]:
            #     print(index, category_id, result)
            per_category_result_dict[category_id]["tp_num"] += result["tp_num"]
            per_category_result_dict[category_id]["fp_num"] += result["fp_num"]
            per_category_result_dict[category_id]["fn_num"] += result["fn_num"]
            per_category_result_dict[category_id]["iou_list"].extend(result["iou_list"])
            per_category_result_dict[category_id]["pred_iou_list"].extend(result["pred_iou_list"])
    return per_category_result_dict

def calculate_score(per_category_result_dict,oc_cost_list,category_id2name):
    """
    Calculate precision, recall, F1 score, and mean IOU for each category and overall dataset.

    Args:
        per_category_result_dict (dict): Per-category results.
        oc_cost_list (list): List of occlusion costs.

    Returns:
        dict: Summary scores and data numbers for the dataset.
    """
    per_category_score_dict = {}

    dataset_score = {
        "summary_scores":{
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "m_iou": [],
            "m_pred_iou": [],
            "oc_cost": np.mean(oc_cost_list) if len(oc_cost_list) > 0 else 0.0,
            "macro_precision": [],
            "macro_recall": [],
            "macro_f1": [],
            "cm_iou": [],
            "cm_pred_iou": [],
        },
        "summary_data_num":{
            "tp_num": 0,
            "fp_num": 0,
            "fn_num": 0,
            "unkonown_fp_num": per_category_result_dict[-1]["fp_num"] if -1 in per_category_result_dict else 0,
            "iou_num": 0,
            "pred_iou_num": 0,
        },
    }

    for category_id, result in per_category_result_dict.items():
        #クラスごと
        tp_num = result["tp_num"]
        fp_num = result["fp_num"]
        fn_num = result["fn_num"]
        iou_list = result["iou_list"]
        pred_iou_list = result["pred_iou_list"]
    
        cm_iou = np.mean(iou_list) if len(iou_list) > 0 else 0.0
        cm_pred_iou = np.mean(pred_iou_list) if len(pred_iou_list) > 0 else 0.0
        # if math.isnan(m_iou):
        #     print(f"Category ID: {category_id} has NaN mIoU. Check the IOU list: {iou_list}")
        #     for iou in iou_list:
        #         assert not math.isnan(iou), f"IOU value is NaN in category {category_id}"
        precision = tp_num / (tp_num + fp_num) if (tp_num + fp_num) > 0 else 0.0
        recall = tp_num / (tp_num + fn_num) if (tp_num + fn_num) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        per_category_score = {
            "category_name": category_id2name[category_id],
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "cm_iou": cm_iou,
            "cm_pred_iou": cm_pred_iou,
            "tp_num": tp_num,
            "fp_num": fp_num,
            "fn_num": fn_num,
        }
        per_category_score_dict[category_id] = per_category_score
        
        #データセット全体
        dataset_score["summary_data_num"]["tp_num"] += tp_num
        dataset_score["summary_data_num"]["fp_num"] += fp_num
        dataset_score["summary_data_num"]["fn_num"] += fn_num
        dataset_score["summary_data_num"]["iou_num"] += len(iou_list)
        dataset_score["summary_data_num"]["pred_iou_num"] += len(pred_iou_list)
        dataset_score["summary_scores"]["m_iou"].extend(iou_list)
        dataset_score["summary_scores"]["m_pred_iou"].extend(pred_iou_list)
        
        #カテゴリごと
        if category_id != -1:
            dataset_score["summary_scores"]["macro_precision"].append(precision)
            dataset_score["summary_scores"]["macro_recall"].append(recall)
            dataset_score["summary_scores"]["macro_f1"].append(f1_score)
            dataset_score["summary_scores"]["cm_iou"].append(cm_iou)
            dataset_score["summary_scores"]["cm_pred_iou"].append(cm_pred_iou)

    #データセット全体
    assert dataset_score["summary_data_num"]["tp_num"] + dataset_score["summary_data_num"]["fn_num"] == len(dataset_score["summary_scores"]["m_iou"]) ,\
        f"TP + FN mismatch: {dataset_score['summary_data_num']['tp_num']} + {dataset_score['summary_data_num']['fn_num']} != {len(dataset_score['summary_scores']['m_iou'])}"
    dataset_score["summary_scores"]["micro_precision"] = dataset_score["summary_data_num"]["tp_num"] / (dataset_score["summary_data_num"]["tp_num"] + dataset_score["summary_data_num"]["fp_num"]) if (dataset_score["summary_data_num"]["tp_num"] + dataset_score["summary_data_num"]["fp_num"]) > 0 else 0.0
    dataset_score["summary_scores"]["micro_recall"] = dataset_score["summary_data_num"]["tp_num"] / (dataset_score["summary_data_num"]["tp_num"] + dataset_score["summary_data_num"]["fn_num"]) if (dataset_score["summary_data_num"]["tp_num"] + dataset_score["summary_data_num"]["fn_num"]) > 0 else 0.0
    dataset_score["summary_scores"]["micro_f1"] = (2 * dataset_score["summary_scores"]["micro_precision"] * dataset_score["summary_scores"]["micro_recall"]) / (dataset_score["summary_scores"]["micro_precision"] + dataset_score["summary_scores"]["micro_recall"]) if (dataset_score["summary_scores"]["micro_precision"] + dataset_score["summary_scores"]["micro_recall"]) > 0 else 0.0
    dataset_score["summary_scores"]["m_iou"] = np.mean(dataset_score["summary_scores"]["m_iou"]) if len(dataset_score["summary_scores"]["m_iou"]) > 0 else 0.0
    dataset_score["summary_scores"]["m_pred_iou"] = np.mean(dataset_score["summary_scores"]["m_pred_iou"]) if len(dataset_score["summary_scores"]["m_pred_iou"]) > 0 else 0.0
    dataset_score["summary_scores"]["macro_precision"] = np.mean(dataset_score["summary_scores"]["macro_precision"]) if len(dataset_score["summary_scores"]["macro_precision"]) > 0 else 0.0
    dataset_score["summary_scores"]["macro_recall"] = np.mean(dataset_score["summary_scores"]["macro_recall"]) if len(dataset_score["summary_scores"]["macro_recall"]) > 0 else 0.0
    dataset_score["summary_scores"]["macro_f1"] = np.mean(dataset_score["summary_scores"]["macro_f1"]) if len(dataset_score["summary_scores"]["macro_f1"]) > 0 else 0.0
    dataset_score["summary_scores"]["cm_iou"] = np.mean(dataset_score["summary_scores"]["cm_iou"]) if len(dataset_score["summary_scores"]["cm_iou"]) > 0 else 0.0 
    dataset_score["summary_scores"]["cm_pred_iou"] = np.mean(dataset_score["summary_scores"]["cm_pred_iou"]) if len(dataset_score["summary_scores"]["cm_pred_iou"]) > 0 else 0.0
    
    for category_id, score in per_category_score_dict.items():
        print(f"category_name: {score['category_name']}")
        print(f"Category ID: {category_id}, Precision: {score['precision']:.4f}, Recall: {score['recall']:.4f}, F1 Score: {score['f1_score']:.4f}, cmIoU: {score['cm_iou']:.4f}, cmPredIoU: {score['cm_pred_iou']:.4f}, TP: {score['tp_num']}, FP: {score['fp_num']}, FN: {score['fn_num']}")

    for key,score in dataset_score["summary_scores"].items():
        print(f"{key}: {score:.4f}")
        
    for key,num in dataset_score["summary_data_num"].items():
        print(f"{key}: {num}")
    return per_category_score_dict, dataset_score

def get_mscoco2017_detection_cat_name2id(split="val"):
    """
    Get the category name to ID mapping for the MS COCO 2017 detection dataset.

    Args:
        split (str): The dataset split, either 'train' or 'val'.

    Returns:
        dict: A dictionary mapping category names to their corresponding IDs.
    """
     # /data_ssd/mscoco2017/coco/annotations/instances_train2017.json
     # /data_ssd/mscoco2017/coco/annotations/instances_val2017.json
    anno_path = f"/data_ssd/mscoco2017/coco/annotations/instances_{split}2017.json"
    #/data_ssd/mscoco2017/coco/annotations/instances_train2017.json
    coco_dataset = COCO(anno_path)
    cat_name2id = {c["name"]: c["id"] for c in coco_dataset.loadCats(coco_dataset.getCatIds())}

    return cat_name2id

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

    for correct, generated in zip(correct_data, generated_data):
        assert correct["id"] == generated["id"], f"ID mismatch: {correct['id']} != {generated['id']}"
        
    cat_name2id = get_mscoco2017_detection_cat_name2id(split="val")
    cat_name2id.update({"unknown": -1})
    category_id2name = {v: k for k, v in cat_name2id.items()}
    processor = AutoProcessor.from_pretrained("/data_ssd/huggingface_model_weights/microsoft/kosmos-2-patch14-224")
    
    # images = create_images_for_coco(correct_data, args.image_folder_root)
    all_gt_annotations = create_annotations_for_coco(correct_data, cat_name2id, processor)
    all_pred_annotations = create_annotations_for_coco(generated_data, cat_name2id, processor)
    per_image_result_dict, oc_cost_list = get_per_image_class_result_and_oc_cost(all_gt_annotations, all_pred_annotations, cat_name2id, iou_threshold=args.iou_threshold)
    per_category_result_dict = convert_per_class_result_dict(per_image_result_dict)
    
    _,output_data = calculate_score(per_category_result_dict, oc_cost_list,category_id2name)
    output_data.update({"filename": args.generated_json,
                        "correct_json": args.gt_json,
                        "timestamp": current_date})
    
    print(f"Saving sorted JSON data to \"{output_path}\"...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
     # Save the output data to the specified output path
    save_json(output_path, output_data)

if __name__ == "__main__":
    
    gt_json_path = "/data_ssd/mscoco-detection/val_for-kosmos2_mscoco2017-detection.json"

    parser = argparse.ArgumentParser(description="Utility functions for JSON handling and sorting.")
    parser.add_argument("-json","--generated_json", type=str, help='Path to the generated JSON file to load or save.')
    parser.add_argument("-gt","--gt_json", type=str, help='Path to the ground truth JSON file to load for sorting.', default=gt_json_path)
    parser.add_argument("--output_path", type=str, help='Path to save the sorted JSON file.',default=None)
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IOU threshold for true positives.")
    # parser.add_argument("--image_folder_root", type=str, default="/data_ssd", help="Root folder for images.")
    args = parser.parse_args()
    main(args)
