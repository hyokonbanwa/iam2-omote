#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
from pathlib import Path
import wandb
import glob
import os
import regex as re
from torchvision.ops import box_iou
import torch
from copy import deepcopy
from tqdm import tqdm
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


# In[ ]:


# filter((row) => row["ce_iou_over_0.5_count"] != null and row["prop_iou_over_0.5_count"] != null)
# runs.summary["result_table"].table.rows[0].filter((row) => row["ce_iou_over_0.5_count"] < row["prop_iou_over_0.5_count"])


# In[ ]:


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


# In[ ]:


def paligemma_get_bbox(text: str,*args, **kwargs):
    pattern = r"(((<loc\d{4}>){4}))"
    matches = re.findall(pattern, text)
    # print("matches", matches)
    bbox_list = []
    for m in matches:
        y1, x1, y2, x2 = [int(x)/1023.0 for x in re.findall(r'\d+', m[1])]
        bbox_list.append([x1, y1, x2, y2])
    return bbox_list, []


def add_bbox_to_wandb_image(wandb_image, entities,cat_2_id_dict=None):
    # load raw input photo
    # person_label_num = 20
    # other_label_num = 20
    # display_ids = {}
    # for i in range(person_label_num):
    #     display_ids.update({f"person{i+1}": i})
    # class_id_to_label = {int(v): k for k, v in display_ids.items()}
    # for num, i in enumerate(
    #     range(person_label_num, person_label_num + other_label_num)
    # ):
    #     class_id_to_label.update({i: f"p_other{num+1}"})
    assert type(wandb_image) == wandb.Image
    name_list = []
    bbox_list = []
    for entity in entities:
        bbox_list.extend(entity[-1])
        name_list.extend([entity[0]]*len(entity[-1]))
    # print(entities)
    # print(bbox_list)
    # print(name_list)
    assert len(name_list) == len(bbox_list)
        
    if cat_2_id_dict == None:
        tmp_class_num = 200
        id_2_cat_dict = {i:f"cat_{i}" for i in range(tmp_class_num)}
        # cat_2_id_dict = {}
        # # print(name_list)
        # for i,name in enumerate(name_list):
        #     # # print(name,i)
        #     # # print(type(name))
        #     # print({name:i})
        #     cat_2_id_dict.update({name:i})
    else:
        cat_2_id_dict.update({"unknown":max(cat_2_id_dict.values())+1})
    
        id_2_cat_dict = {v:k for k,v in cat_2_id_dict.items()}
        
    # import pdb;pdb.set_trace()
    # print(cat_2_id_dict)
    class_id = -1
    if len(name_list) > 0:
        all_boxes = []
        # plot each bounding box for this image
        for name, bbox in zip(name_list, bbox_list):
            if cat_2_id_dict is not None and name in cat_2_id_dict:
                class_id = cat_2_id_dict[name]
            elif cat_2_id_dict is not None:
                class_id = cat_2_id_dict["unknown"]
            else:
                class_id +=1
                
            box_data = {
                "position": {
                    "minX": bbox[0],
                    "maxX": bbox[2],
                    "minY": bbox[1],
                    "maxY": bbox[3],
                },
                "class_id": class_id,  # display_ids[b_name] if b_name in display_ids else 0,
                # optionally caption each box with its class and score
                "box_caption": name,
                # "domain" : "null",#"pixel",
                # "scores" : { }
            }
            all_boxes.append(box_data)

        # log to wandb: raw image, predictions, and dictionary of class labels for each class id
        box_image = wandb.Image(
            wandb_image,
            boxes={
                "predictions": {
                    "box_data": all_boxes,
                    "class_labels": id_2_cat_dict,
                }
            },
        )
            # box_image = wandb.Image(wandb_image, boxes = {"predictions": {"box_data": all_boxes}})
    else:
        box_image = wandb_image
    return box_image


# In[ ]:


def get_correct_table_data(correct_data,wandb_name_to_image):
    correct_dict = {}

    for i in tqdm(range(len(correct_data)), desc="Processing correct data"):
        ann_id = correct_data[i]["ann_id"]
        if  ann_id not in correct_dict:
            correct_bbox = correct_data[i]["gt_entities_quantized_normalized"][0][-1]
            if type(correct_bbox[0]) != list:
                correct_bbox = [correct_bbox]
            image_name = os.path.basename(correct_data[i]["image"])
            correct_dict[ann_id] = {
                "ann_id": ann_id,
                "id_list": [correct_data[i]["id"]],
                "image_name": image_name,
                "gt_entities": correct_data[i]["gt_entities_quantized_normalized"],
                "input": [correct_data[i]["gt_entities_quantized_normalized"][0][0]],
                "correct_data": correct_bbox,
                "gt_bbox_num": len(correct_bbox),
                "gt_output": [correct_data[i]["conversations"][1]["value"]]
            }
        else:
            correct_dict[ann_id]["id_list"].append(correct_data[i]["id"])
            correct_dict[ann_id]["gt_entities"].extend(correct_data[i]["gt_entities_quantized_normalized"])
            correct_dict[ann_id]["input"].append(correct_data[i]["gt_entities_quantized_normalized"][0][0])
            correct_dict[ann_id]["gt_output"].append(correct_data[i]["conversations"][1]["value"])
            
    for ann_id, v in correct_dict.items():
        v["gt_image"] = add_bbox_to_wandb_image(wandb_name_to_image[v["image_name"]], v["gt_entities"])
        v["gt_entities"] = str(v["gt_entities"])
        
    return sort_list_of_dicts(correct_dict.values(),key="ann_id")
    


# In[ ]:


def get_generated_table_data(correct_data, generated_data, unique_key,wandb_name_to_image):
    eval_dict ={}

    for i in tqdm(range(len(correct_data)), desc="PreProcessing generated data"):
        assert correct_data[i]["id"] == generated_data[i]["id"], f"ID mismatch at index {i}."
        ann_id = correct_data[i]["ann_id"]
        if  ann_id not in eval_dict:
            correct_bbox = correct_data[i]["gt_entities_quantized_normalized"][0][-1]
            if type(correct_bbox[0]) != list:
                correct_bbox = [correct_bbox]
            image_name = os.path.basename(correct_data[i]["image"])
            eval_dict[ann_id] = {
                "ann_id": ann_id,
                "image_name": image_name,
                "correct_data": correct_bbox,
                "generated_data": [],
                f"{unique_key}_pred_entities": [],
                f"{unique_key}_pred_output": []
            }

        # eval_dict[ann_id]["correct_data"].append(correct_data[i]["gt_entities_quantized_normalized"][0][-1])
        input_text = correct_data[i]["gt_entities_quantized_normalized"][0][0]
        output_text =generated_data[i]["conversations"][1]["value"]
        
        # for e in entities:
        #     if e[0] == eval_dict[ann_id]["gt_name"]:
        #         generated_bbox = e[0][-1]
        #         break
        #print(generated_data[i]["conversations"][0]["value"]+generated_data[i]["conversations"][1]["value"])
        bbox_list, label_list = paligemma_get_bbox(text=output_text)
        generated_bbox = bbox_list[0] if len(bbox_list) > 0 else None
        
        eval_dict[ann_id][f"{unique_key}_pred_output"].append(output_text)
        if generated_bbox is not None:
            eval_dict[ann_id]["generated_data"].append(generated_bbox)
            eval_dict[ann_id][f"{unique_key}_pred_entities"].append([input_text,[generated_bbox]])
            
            

    for ann_id, eval_item in tqdm(eval_dict.items(), desc="Calculating IOU and Generating Images"):
        iou_info_list = []
        iou_over_0_5_count = 0
        correct_bbox = eval_item["correct_data"]
        generated_bbox = eval_item["generated_data"]
        if len(generated_bbox) > 0:
            iou_info_list ,_,_,_,_ = calculate_iou(correct_bbox, generated_bbox)
            iou_list = [iou_info["iou_value"] for iou_info in iou_info_list]
            iou_over_0_5_count = sum(1 for iou in iou_list if iou >= 0.5)
            
        eval_item[f"{unique_key}_pred_bbox_num"] = len(generated_bbox)
        eval_item[f"{unique_key}_iou_info_list"] = str(iou_info_list)
        eval_item[f"{unique_key}_iou_over_0_5_count"] = iou_over_0_5_count
        eval_item[f"{unique_key}_pred_image"] = add_bbox_to_wandb_image(
            wandb_name_to_image[eval_item["image_name"]], eval_item[f"{unique_key}_pred_entities"]
        )
        eval_item[f"{unique_key}_pred_entities"] = str(eval_item[f"{unique_key}_pred_entities"])
    return sort_list_of_dicts(eval_dict.values(),key="ann_id")


# In[ ]:


gt_path = "/data_ssd/refcoco_g/refcoco_g_paligemma_test.json"
compare_dict = {
    "ce": "/data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/paligemma_refcocog_epoch10/train-vision-proj-llm_cross-entropy_lr1e-05_2025-09-02T18_39_11",
    "prop": "/data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/paligemma_refcocog_epoch10/train-vision-proj-llm_cedfl_excepted_for_paligemma_giou_combine_ce_2025-09-21T00_12_34",
}

eval_json_name = "refcoco_g_paligemma_test"

artifact_entity = "katlab-gifu/dataset/refcocog_test:v2"

ENTITY = "katlab-gifu"
PROJECT = "vis_test"
RUN_ID = "so3gz236"


# In[ ]:


run = wandb.init(entity=ENTITY, project=PROJECT, id=RUN_ID, resume="must")
img_art = run.use_artifact(artifact_entity)


# In[ ]:


wandb_dataset = img_art.get("refcocog_test")


# In[ ]:


wandb_name_to_image = {data_row[0]: data_row[1] for data_row in wandb_dataset.data}
wandb_image_names = set(wandb_name_to_image.keys())
assert len(wandb_image_names) == len(wandb_name_to_image)


# In[ ]:


print(wandb_name_to_image)


# In[ ]:


new_compare_dict = {}
for k,v in compare_dict.items():
    file_list = glob.glob(os.path.join(v,"**",eval_json_name,"**","eval_output.json"),recursive=True)
    assert len(file_list) == 1
    new_compare_dict[k] = file_list[0]
compare_dict = new_compare_dict


# In[ ]:





# In[ ]:


# unique_key = "ce"
# path = compare_dict[unique_key]


# In[ ]:





# In[ ]:


# import sys
# sys.path.append("/home/omote/cluster_project/iam2/eval")
# from eval_utils.custom_oc_cost import get_cmap,get_ot_cost,DetectedInstance


# In[ ]:


# run = wandb.init(entity=ENTITY, project=PROJECT, id=RUN_ID, resume="must")
# img_art = run.use_artifact(artifact_entity)
# img_dir = Path(img_art.download())


# In[ ]:


correct_data = load_json(gt_path)
correct_data = sort_list_of_dicts(correct_data, "id")
correct_data_list = get_correct_table_data(correct_data,wandb_name_to_image)

wandb_columns = ["ann_id","id_list","image_name","gt_image","input","gt_output","gt_bbox_num","gt_entities"]
tmp_table_data = []
for item in correct_data_list:
    row_data = [str(item[k]) if not (type(item[k]) ==  wandb.Image or type(item[k]) ==  int or type(item[k]) ==  float) else item[k] for k in wandb_columns ]
    tmp_table_data.append(row_data)

for unique_key, generated_path in compare_dict.items():
    generated_data = load_json(generated_path)
    assert len(correct_data) == len(generated_data), "Length of correct and generated data does not match."
    generated_data = sort_list_of_dicts(generated_data, "id")
    generated_data_list = get_generated_table_data(correct_data, generated_data, unique_key, wandb_name_to_image)
    
    unique_columns = [
        f"{unique_key}_pred_image",
        f"{unique_key}_pred_output",
        f"{unique_key}_pred_bbox_num",
        f"{unique_key}_iou_info_list",
        f"{unique_key}_iou_over_0_5_count",
        f"{unique_key}_pred_entities"
    ]
    
    for i in range(len(tmp_table_data)):
        assert tmp_table_data[i][0] == generated_data_list[i]["ann_id"], f"Ann ID mismatch at index {i}."
        for col in unique_columns:
            if type(generated_data_list[i][col]) ==  wandb.Image or type(generated_data_list[i][col]) ==  int or type(generated_data_list[i][col]) ==  float:
                tmp_table_data[i].append(generated_data_list[i][col])
            else:
                tmp_table_data[i].append(str(generated_data_list[i][col]))
    wandb_columns.extend(unique_columns)



# In[ ]:


for i, d in enumerate(tmp_table_data):
    # assert len(d) == len(wandb_columns), f"Data length mismatch at index {i}: {len(d)} != {len(wandb_columns)}"
    for col,d in zip(wandb_columns,d):
    #     print(col,type(d),d )
    # break
        if d is None or None in (d if type(d) == list else [d]):
            print(f"None value found in column {col} at row {i}")
            print(f"Type of d: {type(d)}")
            print(f"Value of d: {d}")


# In[ ]:


for i, d in enumerate(tmp_table_data):
    assert len(d) == len(wandb_columns), f"Data length mismatch at index {i}: {len(d)} != {len(wandb_columns)}"
    # for col,d in zip(wandb_columns,d):
    #     print(col,d)


# In[19]:


result_table = wandb.Table(columns=wandb_columns, data=tmp_table_data)
wandb.log({"result_table": result_table})
run.finish()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




