import json
from tqdm import tqdm
import uuid
import os

def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def make_id():
    return str(uuid.uuid4())

def make_conversation(id,image_path,question,answer,image_folder_root=None):
    if image_folder_root is not None:
        image_path = os.path.join(image_folder_root, image_path)
    return_data =   {
        "id": id,
        "image": image_path,
        "conversations": [
        {
            "from": "human",
            "value": f"<image>\n{question}"
        },
        {
            "from": "gpt",
            "value": answer
        },
        ]
    }
    return return_data


def make_question():
    return  f"Please output bbox coordinates and names of every item in this image."

def xywh_to_xyxy(bbox):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[0] + bbox[2]
    y2 = bbox[1] + bbox[3]
    return [x1, y1, x2, y2]

def bbox_absolute_to_relative(absolute_bbox, image_width_height):
    width, height = image_width_height
    x1 = absolute_bbox[0] / width
    y1 = absolute_bbox[1] / height
    x2 = absolute_bbox[2] / width
    y2 = absolute_bbox[3] / height
    relative_bbox = [x1, y1, x2, y2]
    return relative_bbox

def make_str_bbox(bbox, image_width_height):
    relative_bbox = bbox_absolute_to_relative(bbox, image_width_height)
    relative_bbox = [f"{coord:.3f}" for coord in relative_bbox]
    
    return f"[{relative_bbox[0]},{relative_bbox[1]},{relative_bbox[2]},{relative_bbox[3]}]"

def make_answer(image_width_height, bbox_list, caption_list):
    answer = []
    for bbox, caption in zip(bbox_list, caption_list):
        str_bbox = make_str_bbox(bbox, image_width_height)
        answer.append(f"{caption}: {str_bbox}")
    return "\n".join(answer)

new_json_data_path = "/home/omote/local-share-data_ssd/object365/raw_data/reshaped_zhiyuan_objv2_train.json"

new_json_data = load_json(new_json_data_path)


converted_data = []

for sample in tqdm(new_json_data):
    id = make_id()
    
    image_file_name = sample["file_name"]
    tmp = image_file_name[image_file_name.find("patch"):]
    image_path = os.path.join("objects365/train",tmp)
    
    original_image_width_height = (sample["width"], sample["height"])
    

    bbox_list = []
    caption_list = []
    for annotation in sample["annotations"]:
        bbox_list.append(xywh_to_xyxy(annotation["bbox"]))
        caption_list.append(annotation["category"])
    
    question = make_question()
    answer = make_answer(original_image_width_height,bbox_list, caption_list)
    conversation = make_conversation(id,image_path,question,answer)
    converted_data.append(conversation)
    
    
conversation_json_path = "/home/omote/local-share-data_ssd/object365/object365_train_conversation.json"

with open(conversation_json_path, "w") as f:
    json.dump(converted_data, f, indent=4, ensure_ascii=False)
print(f"新しいJSONデータが保存されました: {conversation_json_path}")
