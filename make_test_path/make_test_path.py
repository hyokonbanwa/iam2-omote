import argparse
from pathlib import Path
import csv
import os

def load_csv(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)  # 1行目を自動でヘッダ(キー)として扱う
        for row in reader:
            rows.append(dict(row))   # 各行= {ヘッダ: 値}
    return rows

def main(args):
    input_path = args.input_path
    output_path = args.output_path
    root_dir = args.root_dir
    target_key = args.keys[0]
    rows = load_csv(input_path)

    weight_name_key = "Name"  # 名前の列名

    with output_path.open("w", encoding="utf-8", newline="\n") as w:
        for row in rows:
            # キーが無い/空のときはスキップする場合
            best_checkpoint_path = os.path.join(root_dir, row[weight_name_key], f"checkpoint-{row[target_key]}")
            assert os.path.exists(best_checkpoint_path), f"Path does not exist: {best_checkpoint_path}"
            w.write(best_checkpoint_path + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a test file with a specified path.")
    parser.add_argument(
        "-i", "--input_path",
        type=Path,
        required=True,
        help="入力CSVファイルのパス"
    )
    parser.add_argument(
        "-o", "--output_path",
        type=Path,
        required=True,
        help="出力TXTファイルのパス"
    )
    parser.add_argument(
        "--root_dir",
        type=Path,
        help="ルートディレクトリのパス"
    )
    parser.add_argument(
        "-k", "--key",
        dest="keys",
        action="append",
        required=True,
        help="取り出す列名。複数指定可（例: -k name -k age）"
    )
    args = parser.parse_args()
    main(args)