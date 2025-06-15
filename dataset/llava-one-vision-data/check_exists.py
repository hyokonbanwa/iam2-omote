import os
import json
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import io


ALL_NAMES = ['CLEVR-Math(MathV360K)', 'Evol-Instruct-GPT4-Turbo', 'FigureQA(MathV360K)', 'GEOS(MathV360K)', 'GeoQA+(MathV360K)', 'Geometry3K(MathV360K)', 'IconQA(MathV360K)', 'MapQA(MathV360K)', 'MathV360K_TQA', 'MathV360K_VQA-AS', 'MathV360K_VQA-RAD', 'PMC-VQA(MathV360K)', 'Super-CLEVR(MathV360K)', 'TabMWP(MathV360K)', 'UniGeo(MathV360K)', 'VisualWebInstruct(filtered)', 'VizWiz(MathV360K)', 'ai2d(cauldron,llava_format)', 'ai2d(gpt4v)', 'ai2d(internvl)', 'allava_instruct_laion4v', 'allava_instruct_vflan4v', 'aokvqa(cauldron,llava_format)', 'chart2text(cauldron)', 'chartqa(cauldron,llava_format)', 'chrome_writting', 'clevr(cauldron,llava_format)', 'diagram_image_to_text(cauldron)', 'dvqa(cauldron,llava_format)', 'figureqa(cauldron,llava_format)', 'geo170k(align)', 'geo170k(qa)', 'geo3k', 'geomverse(cauldron)', 'hateful_memes(cauldron,llava_format)', 'hitab(cauldron,llava_format)', 'hme100k', 'iam(cauldron)', 'iconqa(cauldron,llava_format)', 'iiit5k', 'image_textualization(filtered)', 'infographic(gpt4v)', 'infographic_vqa', 'infographic_vqa_llava_format', 'intergps(cauldron,llava_format)', 'k12_printing', 'llava_wild_4v_12k_filtered', 'llava_wild_4v_39k_filtered', 'llavar_gpt4_20k', 'lrv_chart', 'lrv_normal(filtered)', 'magpie_pro(l3_80b_mt)', 'magpie_pro(l3_80b_st)', 'magpie_pro(qwen2_72b_st)', 'mapqa(cauldron,llava_format)', 'mathqa', 'mavis_math_metagen', 'mavis_math_rule_geo', 'multihiertt(cauldron)', 'orand_car_a', 'raven(cauldron)', 'rendered_text(cauldron)', 'robut_sqa(cauldron)', 'robut_wikisql(cauldron)', 'robut_wtq(cauldron,llava_format)', 'scienceqa(cauldron,llava_format)', 'scienceqa(nona_context)', 'screen2words(cauldron)', 'sharegpt4o', 'sharegpt4v(coco)', 'sharegpt4v(knowledge)', 'sharegpt4v(llava)', 'sharegpt4v(sam)', 'sroie', 'st_vqa(cauldron,llava_format)', 'tabmwp(cauldron)', 'tallyqa(cauldron,llava_format)', 'textcaps', 'textocr(gpt4v)', 'tqa(cauldron,llava_format)', 'ureader_cap', 'ureader_ie', 'vision_flan(filtered)', 'vistext(cauldron)', 'visual7w(cauldron,llava_format)', 'visualmrc(cauldron)', 'vqarad(cauldron,llava_format)', 'vsr(cauldron,llava_format)', 'websight(cauldron)']
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
    "mathqa",
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
]
# 画像とJSONデータの保存先を指定
OUT_DIR = '/data_ssd/USER_DATA/omote/iam2/LLaVA-OneVision-Data'
IMAGE_DIR = '/data_ssd/huggingface_dataset/lmms-lab/LLaVA-OneVision-Data' 
MAX_WORKERS=512
os.makedirs(OUT_DIR, exist_ok=True)
# DATASET_NAMES.reverse()
print("DATASET_NAMES: ", DATASET_NAMES)
# DATASET_NAMES=[DATASET_NAMES[-1]]
print("len(ALL_NAMES): ", len(ALL_NAMES))
print("len(DATASET_NAMES): ", len(DATASET_NAMES))
print(set(ALL_NAMES) - set(DATASET_NAMES))  # DATASET_NAMESに含まれないALL_NAMESの要素を表示
print(set(DATASET_NAMES) - set(ALL_NAMES))  # ALL_NAMESに含まれないDATASET_NAMESの要素を表示
    

# 全てのデータセットを並列処理で処理
print("MAX_WORKERS: ", MAX_WORKERS)
not_json_found_datasets = []
not_image_found_datasets = []

for dataset_name in DATASET_NAMES:
    json_output_path = os.path.join(OUT_DIR, dataset_name, f'{dataset_name}.json')
    image_path = os.path.join(OUT_DIR, dataset_name,"train")
    print("-" * 50)
    if os.path.exists(json_output_path):
        print(f"JSON file already exists for {dataset_name}, skipping...")

            
    else:
        print(f"No JSON file found for {dataset_name}, processing...")
        not_json_found_datasets.append(dataset_name)
    
    if os.path.exists(image_path):
        print(f"Image directory already exists for {dataset_name}, skipping...")
    else:
        print(f"Image directory does not exist for {dataset_name}, creating...")
        not_image_found_datasets.append(dataset_name)
        
print("*" * 50)
print("not_json_found_datasets: ", not_json_found_datasets)
print("not_image_found_datasets: ", not_image_found_datasets)
