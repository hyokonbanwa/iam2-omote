import tarfile
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import sys
import os

def extract_targz(args):
    tar_path, source_dir, target_dir,use_tar_stem = args
    tar_path = Path(tar_path)
    target_dir = Path(target_dir)

    if use_tar_stem == "true":
        # ターゲットディレクトリにtarファイルの名前をそのまま使用
        tar_stem = tar_path.name.replace('.tar.gz', '')
        output_dir = target_dir / tar_stem
    elif use_tar_stem == "false":
        output_dir = target_dir
    else:
        return f"[エラー] use_tar_stem の値は 'true' または 'false' でなければなりません。"
        
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=output_dir)  # Python 3.14以降は filter="data" でもOK
        return f"[成功] {tar_path.name}"
    except tarfile.TarError as e:
        return f"[失敗] {tar_path.name} → {e}"

def main(source_dir, target_dir, use_tar_stem):
    source_dir = Path(source_dir).resolve()
    target_dir = Path(target_dir).resolve()

    #tar_files = list(source_dir.rglob("*.tar.gz"))
    tar_files = [os.path.join(source_dir,item) for item in os.listdir(source_dir) if item.endswith(".tar.gz")]
    if not tar_files:
        print("解凍対象の .tar.gz ファイルが見つかりませんでした。")
        return

    args_list = [(str(tar_path), str(source_dir), str(target_dir), use_tar_stem) for tar_path in tar_files]

    with Pool(processes=min(cpu_count(), len(tar_files))) as pool:
        with tqdm(total=len(tar_files), desc="解凍進行中", ncols=80) as pbar:
            for result in pool.imap_unordered(extract_targz, args_list):
                pbar.update(1)
                print(result)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("使い方: python extract_targz_parallel_tqdm.py <source_dir> <target_dir> <use_tar_stem>")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
