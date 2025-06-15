from datasets import load_dataset
import os
from tqdm import tqdm
import random
import json

random.seed(42)

dataset_root_dir = "/data_ssd/huggingface_dataset"
cache_dir = "/data_ssd/huggingface_cache"

dataset_id = os.path.join(dataset_root_dir,"BAAI/DenseFusion-1M")
dataset = load_dataset(dataset_id, cache_dir=cache_dir,name="DenseFusion-1M",split="train") # ['DenseFusion-4V-100K', 'DenseFusion-1M']

prompt_list = \
[
 '<image>\nExplain the visual content of the image in great detail.', 
 '<image>\nPlease explain the visual content of the image in detail.', 
 
 '<image>\nPlease describe in depth what this image is about.',
 '<image>\nCan you explain in detail what is happening in the picture?',
 '<image>\nWrite a detailed description of the given image.', 
 '<image>\nPlease elaborate on what this image shows.',
 '<image>\nWhat items or people are prominent in the picture?',
 '<image>\nCan you elaborate on the elements of the picture provided?',
 '<image>\nAnalyze the image in a comprehensive and detailed manner.',
  '<image>\nWhat are the striking details of this image?', 
  '<image>\nWrite a detailed and comprehensive description of the image.',
  
  '<image>\nDescribe the image, paying attention to its inner details.', 
  '<image>\nAnalyze and describe in detail the visual elements in this image.', 
  '<image>\nDescribe the content of a given image in detail', 

  '<image>\nDescribe every detail in the picture.', 
  '<image>\nPlease explain in detail the scene depicted in the picture.', 
  '<image>\nPlease conduct an in-depth analysis of the scene in the picture.',
  '<image>\nCan you describe all the objects and characters in the picture?',
  '<image>\nProvide a detailed description of the presented image.', 
  '<image>\nPlease use detailed words to describe what the picture is about.',

   '<image>\nDescribe everything in the image',
   '<image>\nPlease describe specifically what you observed in the picture and the possible scenes they might form.',

   '<image>\nPlease interpret and describe each detail of this image and the overall scene they create.',
   '<image>\nDescribe all the elements in the picture.',
    '<image>\nProvide a detailed description of the image.',
    '<image>\nCan you describe this photo in detail?',
    '<image>\nCan you analyze and elaborate on all the elements and details shown in this image?',]

def make_conversation(id,image_path,question,answer,image_folder_root=None):
    if image_folder_root is not None:
        image_path = os.path.join(image_folder_root, image_path)
    return_data =   {
        "id": id,
        "image": image_path,
        "conversations": [
        {
            "from": "human",
            "value": f"{question}"
        },
        {
            "from": "gpt",
            "value": answer
        },
        ],
    }
    return return_data

save_json_data = []

for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
    image_path = sample["image_path"]
    question = random.choice(prompt_list)
    answer = sample["caption"]
    id = f"DenseFusion-1M-{sample["image_id"]}"
    
    conversation = make_conversation(id=id, image_path=image_path, question=question, answer=answer)
    
    save_json_data.append(conversation)


new_json_data_path = "/data_ssd/DenseFusion1M/densefusion1m-train.json"
if not os.path.exists(os.path.dirname(new_json_data_path)):
    os.makedirs(os.path.dirname(new_json_data_path))
with open(new_json_data_path, "w") as f:
    json.dump(save_json_data, f, indent=4, ensure_ascii=False)
print(f"新しいJSONデータが保存されました: {new_json_data_path}")
