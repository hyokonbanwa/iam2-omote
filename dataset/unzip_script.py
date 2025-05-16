import os
import zipfile
from pathlib import Path

def unzip_all(source_dir, target_dir):
    source_dir = Path(source_dir).resolve()
    target_dir = Path(target_dir).resolve()
    if not source_dir.is_dir():
        raise ValueError(f"指定されたソースディレクトリは存在しません: {source_dir}")
    # ZIPファイルを再帰的に探索
    for zip_path in source_dir.rglob('*.zip'):
        relative_path = zip_path.relative_to(source_dir)  # source_dirからの相対パス
        zip_stem = zip_path.stem  # 拡張子なしファイル名
        output_dir = target_dir / zip_stem

        # 出力先ディレクトリを作成
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            print(f"解凍成功: {zip_path} → {output_dir}")
        except zipfile.BadZipFile:
            print(f"無効なZIPファイル: {zip_path}")
        except Exception as e:
            print(f"エラー: {zip_path} → {e}")

if __name__ == "__main__":
    import sys
    print("ZIPファイルを解凍するスクリプト")
    if len(sys.argv) != 3:
        print("使い方: python unzip_script.py <source_dir> <target_dir>")
    else:
        unzip_all(sys.argv[1], sys.argv[2])
