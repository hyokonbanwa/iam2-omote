folder_list = [
    "AESOP",
"IEdit",
"MIT-States_PropertyCoherence",
"MIT-States_StateCoherence",
"CLEVR-Change",
"COMICS_Dialogue",
"OCR-VQA",
"DocVQA",
"Birds-to-Words",
"RAVEN_train_images",
"PororoSV",
"FlintstonesSV",
"RecipeQA_ImageCoherence",
"RecipeQA_VisualCloze",
"MagicBrush",
"Spot-the-Diff",
"ALFRED",
"WebQA",
"VizWiz",
"VISION",
"VIST",
"iconqa",
"coinstruct",
"multi_vqa",
"nextqa",
"TQA",
"mmchat",
"imagecode",
"star",
"nuscenes",
"scannet_frames_25k",
"contrastive_caption",
"nlvr2",
"dreamsim_split",
"HQ-Edit",
]

import os 
import shutil
source_dir = "/home/omote/local-share-data_ssd/M4-Instruct-Data"

target_dir = "/home/omote/local-share-data_ssd/M4-Instruct-Data_processed"

for folder in folder_list:
    source_path = os.path.join(source_dir, folder,folder,"full")
    target_path = os.path.join(target_dir, folder)

    if os.path.exists(source_path):
        if not os.path.exists(target_path):
            os.makedirs(target_path, exist_ok=True)
        # Move the folder
        json_path = os.path.join(source_path, "full.json")
        images_path = os.path.join(source_path, "images")
        #shutil.move(source_path, target_path)
        shutil.move(json_path, target_path)
        shutil.move(images_path, target_path)
        # Optionally, you can also use shutil.copytree if you want to copy instead of move
        print(f"Moved: {source_path} to {target_path}")
    else:
        print(f"Source path does not exist: {source_path}")
