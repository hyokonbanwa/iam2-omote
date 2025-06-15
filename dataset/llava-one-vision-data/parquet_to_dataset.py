import os
import json
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import io

DATASET_NAMES = [
    # "CLEVR-Math(MathV360K)",
    # "FigureQA(MathV360K)",
    # "GEOS(MathV360K)",
    # "GeoQA+(MathV360K)",
    # "Geometry3K(MathV360K)",
    # "IconQA(MathV360K)",
    # "MapQA(MathV360K)",
    # "PMC-VQA(MathV360K)",
    # "Super-CLEVR(MathV360K)",
    # "TabMWP(MathV360K)",
    # "UniGeo(MathV360K)",
    # "VisualWebInstruct(filtered)",
    # "VizWiz(MathV360K)",
    # "ai2d(cauldron,llava_format)",
    # "ai2d(gpt4v)",
    # "ai2d(internvl)",
    # "allava_instruct_laion4v",
    # "allava_instruct_vflan4v",
    # "aokvqa(cauldron,llava_format)",
    # "chart2text(cauldron)",
    # "chartqa(cauldron,llava_format)",
    # "chrome_writting",
    # "clevr(cauldron,llava_format)",
    # "diagram_image_to_text(cauldron)",
    # "dvqa(cauldron,llava_format)",
    # "figureqa(cauldron,llava_format)",
    # "geo170k(align)",
    # "geo170k(qa)",
    # "geo3k",
    # "geomverse(cauldron)",
    # "hateful_memes(cauldron,llava_format)",
    # "hitab(cauldron,llava_format)",
    # "hme100k",
    # "iam(cauldron)",
    # "iconqa(cauldron,llava_format)",
    # "iiit5k",
    # "image_textualization(filtered)",
    # "infographic(gpt4v)",
    # "infographic_vqa",
    # "infographic_vqa_llava_format",
    # "intergps(cauldron,llava_format)",
    # "k12_printing",
    # "llavar_gpt4_20k",
    # "lrv_chart",
    # "lrv_normal(filtered)",
    # "magpie_pro(l3_80b_mt)", #画像データがない
    # "magpie_pro(l3_80b_st)", #画像データがない
    # "magpie_pro(qwen2_72b_st)", #画像データがない
    # "mapqa(cauldron,llava_format)",
    # "mathqa",
    # "mavis_math_metagen",
    # "mavis_math_rule_geo",
    # "multihiertt(cauldron)",
    # "orand_car_a",
    # "raven(cauldron)",
    # "rendered_text(cauldron)",
    # "robut_sqa(cauldron)",
    # "robut_wikisql(cauldron)",
    # "robut_wtq(cauldron,llava_format)",
    # "scienceqa(cauldron,llava_format)",
    # "scienceqa(nona_context)",
    # "screen2words(cauldron)",
    # "sharegpt4o",
    # "sharegpt4v(coco)",
    # "sharegpt4v(knowledge)",
    # "sharegpt4v(llava)",
    # "sharegpt4v(sam)",
    # "sroie",
    # "st_vqa(cauldron,llava_format)",
    # "tabmwp(cauldron)",
    # "tallyqa(cauldron,llava_format)",
    # "textcaps",
    # "textocr(gpt4v)",
    # "tqa(cauldron,llava_format)",
    # "ureader_cap", 
    # "ureader_ie",
    # "vision_flan(filtered)",
    # "vistext(cauldron)",
    # "visual7w(cauldron,llava_format)",
    # "visualmrc(cauldron)",
    # "vqarad(cauldron,llava_format)",
    # "vsr(cauldron,llava_format)",
    # "websight(cauldron)",
    # "ureader_kg", #parquetファイルではない
    # "ureader_qa", #parquetファイルではない
    'llava_wild_4v_39k_filtered', 
    'MathV360K_VQA-RAD', 
    'MathV360K_VQA-AS', 
    'Evol-Instruct-GPT4-Turbo', 
    'llava_wild_4v_12k_filtered', 
    'MathV360K_TQA'
]
# 画像とJSONデータの保存先を指定
OUT_DIR = '/data_ssd/USER_DATA/omote/iam2/LLaVA-OneVision-Data'
IMAGE_DIR = '/data_ssd/huggingface_dataset/lmms-lab/LLaVA-OneVision-Data' 
MAX_WORKERS=512
os.makedirs(OUT_DIR, exist_ok=True)
# DATASET_NAMES.reverse()
print("DATASET_NAMES: ", DATASET_NAMES)
# DATASET_NAMES=[DATASET_NAMES[-1]]


# 並列処理で画像を保存する関数
def save_image_and_metadata(entry, dataset_name):
    # メタデータ作成
    metadata = {key: entry[key] for key in entry.keys() if key != 'image'}
    if entry['image']!= None:
        img_data = entry['image']
        image_name = entry['id']
        # 画像データをPIL形式に変換
        if image_name.endswith('.png'):
            image_name = image_name[:-4]
        elif image_name.endswith('.jpg'):
            image_name = image_name[:-4]
        elif image_name.endswith('.jpeg'):
            image_name = image_name[:-5]
        elif image_name.endswith('.webp'):
            image_name = image_name[:-5]
        elif image_name.endswith('.bmp'):
            image_name = image_name[:-4]    
        image_path = os.path.join(OUT_DIR, dataset_name, "train", f'{image_name}.png')
        # 画像を保存
        if img_data.mode != "RGB":
            # 画像データがRGBでない場合、変換して保存
            img_data = img_data.convert("RGB")
        
        if not os.path.exists(image_path):
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
        img_data.save(image_path)

        metadata['image_path'] = image_path

    return metadata

# 並列化してデータを処理する関数
def process_dataset(dataset_name):
    print(f"Processing dataset: {dataset_name}")
    dataset = load_dataset('parquet', data_files=os.path.join(IMAGE_DIR, dataset_name, '*.parquet'))

    # JSONファイルに保存するためのデータ
    json_data = []

    # os.makedirs(os.path.join(OUT_DIR, dataset_name, "train"), exist_ok=True)

    # import pdb; pdb.set_trace()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i, entry in enumerate(dataset['train']):
            futures.append(executor.submit(save_image_and_metadata, entry, dataset_name))

        # 完了したタスクを順に処理
        for future in tqdm(as_completed(futures), total=len(futures)):
            json_data.append(future.result())

    # JSONファイルとして保存
    json_output_path = os.path.join(OUT_DIR, dataset_name, f'{dataset_name}.json')
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)

    with open(json_output_path, 'w') as f:
        json.dump(json_data, f, indent=4)

# 全てのデータセットを並列処理で処理
print("MAX_WORKERS: ", MAX_WORKERS)
for dataset_name in DATASET_NAMES:
    try:
        process_dataset(dataset_name)
    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {e}")
        # import pdb; pdb.set_trace()
        continue
