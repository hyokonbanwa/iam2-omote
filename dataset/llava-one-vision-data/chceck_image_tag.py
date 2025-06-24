import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import io

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

DATASET_NAMES = [
    "CLEVR-Math(MathV360K)",
    "FigureQA(MathV360K)",
    "GEOS(MathV360K)",
    "GeoQA+(MathV360K)",
    "Geometry3K(MathV360K)",
    "IconQA(MathV360K)",
    "MapQA(MathV360K)",
    "PMC-VQA(MathV360K)",
    "Super-CLEVR(MathV360K)",
    "TabMWP(MathV360K)",
    "UniGeo(MathV360K)",
    "VisualWebInstruct(filtered)",
    "VizWiz(MathV360K)",
    "ai2d(cauldron,llava_format)",
    "ai2d(gpt4v)",
    "ai2d(internvl)",
    "allava_instruct_laion4v",
    "allava_instruct_vflan4v",
    "aokvqa(cauldron,llava_format)",
    "chart2text(cauldron)",
    "chartqa(cauldron,llava_format)",
    "chrome_writting",
    "clevr(cauldron,llava_format)",
    "diagram_image_to_text(cauldron)",
    "dvqa(cauldron,llava_format)",
    "figureqa(cauldron,llava_format)",
    "geo170k(align)",
    "geo170k(qa)",
    "geo3k",
    "geomverse(cauldron)",
    "hateful_memes(cauldron,llava_format)",
    "hitab(cauldron,llava_format)",
    "hme100k",
    "iam(cauldron)",
    "iconqa(cauldron,llava_format)",
    "iiit5k",
    "image_textualization(filtered)",
    "infographic(gpt4v)",
    "infographic_vqa",
    "infographic_vqa_llava_format",
    "intergps(cauldron,llava_format)",
    "k12_printing",
    "llavar_gpt4_20k",
    "lrv_chart",
    "lrv_normal(filtered)",
    "magpie_pro(l3_80b_mt)", #画像データがない
    "magpie_pro(l3_80b_st)", #画像データがない
    "magpie_pro(qwen2_72b_st)", #画像データがない
    "mapqa(cauldron,llava_format)",
    "mathqa",#画像データがない
    "mavis_math_metagen",
    "mavis_math_rule_geo",
    "multihiertt(cauldron)",
    "orand_car_a",
    "raven(cauldron)",
    "rendered_text(cauldron)",
    "robut_sqa(cauldron)",
    "robut_wikisql(cauldron)",
    "robut_wtq(cauldron,llava_format)",
    "scienceqa(cauldron,llava_format)",
    "scienceqa(nona_context)",
    "screen2words(cauldron)",
    "sharegpt4o",
    "sharegpt4v(coco)",
    "sharegpt4v(knowledge)",
    "sharegpt4v(llava)",
    "sharegpt4v(sam)",
    "sroie",
    "st_vqa(cauldron,llava_format)",
    "tabmwp(cauldron)",
    "tallyqa(cauldron,llava_format)",
    "textcaps",
    "textocr(gpt4v)",
    "tqa(cauldron,llava_format)",
    "ureader_cap", 
    "ureader_ie",
    "vision_flan(filtered)",
    "vistext(cauldron)",
    "visual7w(cauldron,llava_format)",
    "visualmrc(cauldron)",
    "vqarad(cauldron,llava_format)",
    "vsr(cauldron,llava_format)",
    "websight(cauldron)",
    "ureader_kg", #parquetファイルではない
    "ureader_qa", #parquetファイルではない
    'llava_wild_4v_39k_filtered', 
    'MathV360K_VQA-RAD', 
    'MathV360K_VQA-AS', 
    'Evol-Instruct-GPT4-Turbo', #画像データがない
    'llava_wild_4v_12k_filtered', 
    'MathV360K_TQA'
]

# 画像とJSONデータの保存先を指定
DATA_DIR = '/data_ssd/LLaVA-OneVision-Data'
PREFIX_DIR = "" #'/data_ssd/USER_DATA/omote/iam2'

def check_image_tag(dataset_name):
    json_file_path = os.path.join(DATA_DIR,dataset_name, f'{dataset_name}_relative_path.json')
    
    if not os.path.exists(json_file_path):
        print(f"JSON file not found for {dataset_name}: {json_file_path}")
        return
    
    data = load_json(json_file_path)
    
    save_data = []
    for index,item in enumerate(tqdm(data, desc=f"Processing {dataset_name}")):
        if "image" in item:
            image_list = item["image"] if isinstance(item["image"], list) else [item["image"]]
            image_count = 0
            for conversation in item["conversations"]:
                image_count += conversation["value"].count("<image>")
                
            if image_count != len(image_list):
                if image_count == 0 and len(image_list) == 1:
                    item["conversations"][0]["value"] = f"<image>\n{item["conversations"][0]["value"]}"
                else:
                    print(image_list[0])
                    print(image_count)
            save_data.append(item)
        else:
            save_data.append(item)
    
    # 変更後のデータを保存
    save_json_file_path = os.path.join(DATA_DIR, dataset_name, f'{dataset_name}_checked_image_tag.json')
    with open(save_json_file_path, 'w', encoding='utf-8') as file:
        json.dump(save_data, file, ensure_ascii=False, indent=4)

def main():
    """
    Main function to process all datasets.
    """
    for dataset_name in DATASET_NAMES:
        print(f"Processing dataset: {dataset_name}")
        check_image_tag(dataset_name)
if __name__ == "__main__":
    main()
