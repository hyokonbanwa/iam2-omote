import os
import glob
from pathlib import Path
import subprocess
import wandb
import argparse
import json
import numpy as np

def load_json(file_path):
    """
    Load data from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Data loaded from the file.
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def main(args):
    model_path = args.model_path
    dataset_base_name = args.dataset_base_name
    eval_script = args.eval_script
    other_command = args.other_command
    
    
    path_list = glob.glob(os.path.join(model_path, "checkpoint-*", "eval_output",dataset_base_name, "*", "eval_output.json"), recursive=True)

    checkpoint_steps = []
    path_list = sorted(path_list, key=lambda x: int(x.split("/")[-5].split("-")[-1]))
    assert len(set(checkpoint_steps)) == len(checkpoint_steps), "Checkpoint steps should be unique."
    
    for path in path_list:
        checkpoint = int(path.split("/")[-5].split("-")[-1])
        if checkpoint not in checkpoint_steps:
            checkpoint_steps.append(checkpoint)
                
    # for path in path_list:
    #     run_command = ["python", eval_script, "--generated_json", path]
    #     run_command.extend(other_command)
    #     subprocess.run(run_command, check=True)
        
    if args.wandb_run_id and args.wandb_project:
        print(f"Using WandB run ID: {args.wandb_run_id} and project: {args.wandb_project}")
        PROJECT = args.wandb_project
        RUN_ID = args.wandb_run_id
        script_base_name = os.path.basename(eval_script).split(".")[0]
        dataset_script_name = script_base_name + "_" + dataset_base_name
        score_path_list = glob.glob(os.path.join(model_path, "checkpoint-*", "eval_output",dataset_base_name, "*", f"{script_base_name}.json"), recursive=True)
        score_path_list = sorted(score_path_list, key=lambda x: int(x.split("/")[-5].split("-")[-1]))
        

        run = wandb.init(
            project=PROJECT,
            id=RUN_ID,
            resume="allow")            # 既存 run があれば再開、無ければエラー無し :contentReference[oaicite:0]{index=0}

        # optional: グラフを学習と同じ軸に重ねたい時

        scores_dict = {}

        for step, score_path in zip(checkpoint_steps, score_path_list):
            print(f"Step: {step}")
            data = load_json(score_path)
            # print(flatten_json(data))
            scores = data.get("scores", {})
            data_num = data.get("data_num", {})
            scores.update(data_num)
            tmp_scores = {}
            for k, v in scores.items():
                tmp_scores[f"{dataset_script_name}/{k}"] = v
                #run.define_metric(f"{dataset_script_name}/{k}", step_metric=f"{dataset_script_name}/step")
                if f"{dataset_script_name}/{k}" not in scores_dict:
                    scores_dict[f"{dataset_script_name}/{k}"] = [v]
                else:
                    scores_dict[f"{dataset_script_name}/{k}"].append(v)
            scores = tmp_scores
            # print(f"Scores: {scores}")
            run.log({                # history に行を追加
                **scores,            # スコアを追加
                f"{dataset_script_name}/step":         step     # or step=step に渡す
            })  # step を指定してログする
        
        #各スコアの最大値を summary に追加
        max_scores_values = {f"{k}/max": max(v) for k, v in scores_dict.items()}  # 各スコアの最大値を取得
        run.summary.update(max_scores_values)
        max_scores_steps  = { f"{k}/max_step":checkpoint_steps[np.argmax(v)] for k, v in scores_dict.items() }  # 各スコアの最大値のステップを取得
        run.summary.update(max_scores_steps)  # 各スコアの最大値のステップを summary に追加
        
        # 各スコアの最小値を summary に追加
        min_scores_values = {f"{k}/min": min(v) for k, v in scores_dict.items()}  # 各スコアの最小値を取得
        run.summary.update(min_scores_values)
        min_scores_steps  = { f"{k}/min_step":checkpoint_steps[np.argmin(v)] for k, v in scores_dict.items() }
        run.summary.update(min_scores_steps)  # 各スコアの最小値
        
        # ★run を明示的に閉じる
        run.finish()


def csv_list(string):
    # 空文字や余分な空白もケア
    return [s.strip() for s in string.split(',') if s.strip()]

#python /home/omote/cluster_project/iam2/eval/eval_all_checkpoint.py --model_path a --dataset_base_name b --eval_script c --other_command="--x,--b"
#Namespace(model_path='a', dataset_base_name='b', eval_script='c', run_id=None, project=None, other_command=['--x', '--b'])
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utility functions for JSON handling and sorting.")
    parser.add_argument("--model_path", type=str, help='Path to the model directory containing checkpoints.',required=True)
    parser.add_argument("--dataset_base_name", type=str, help='Base name of the dataset for evaluation.',required=True)
    parser.add_argument("--eval_script", type=str, help='Path to the evaluation script.',required=True)
    parser.add_argument("--wandb_run_id", type=str, help='WandB run ID.')
    parser.add_argument("--wandb_project", type=str, help='WandB project name.')
    parser.add_argument("--other_command", type=csv_list, help='Additional command line arguments for the evaluation script.', default=[])
    args = parser.parse_args()
    print(f"Arguments: {args}")
    main(args)
