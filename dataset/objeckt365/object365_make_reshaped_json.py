import json
from tqdm import tqdm
import uuid
def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

data  = load_json("/home/omote/local-share-data_ssd/object365/raw_data/zhiyuan_objv2_train.json")

images = data["images"]
annotations = data["annotations"]
categories = data["categories"]

categories_dict = {}
for category in categories:
    categories_dict[category["id"]] = category["name"]

new_json_data = []
annotation_index = 0
for i, image_info in tqdm(enumerate(images)):
    image_id = image_info["id"]
    file_name = image_info["file_name"]
    width = image_info["width"]
    height = image_info["height"]
    annotation_list = []
    for annotation in annotations[annotation_index:]:
        if annotation["image_id"] == image_id:
            category_id = annotation["category_id"]
            category_name = categories_dict[category_id]
            bbox = annotation["bbox"]
            iscrowd = annotation["iscrowd"]
            isfake = annotation["isfake"]
            annotation_list.append({
                "category": category_name,
                "bbox": bbox,
                "iscrowd": iscrowd,
                "isfake": isfake,
            })
            annotation_index += 1
        else:
            break
    new_json_data.append({
        "image_id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height,
        "annotations": annotation_list,
    })
    
new_json_data_path = "/home/omote/local-share-data_ssd/object365/raw_data/reshaped_zhiyuan_objv2_train.json"
with open(new_json_data_path, "w") as f:
    json.dump(new_json_data, f, indent=4, ensure_ascii=False)
print(f"新しいJSONデータが保存されました: {new_json_data_path}")



