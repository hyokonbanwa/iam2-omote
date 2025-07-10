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
    gen_args_folder = args.gen_args_folder
    
    if gen_args_folder:
        join_path_list = [model_path, "checkpoint-*", "eval_output",dataset_base_name, gen_args_folder, "*", "eval_output.json"]
        path_checkpoint_index = -6
    else:
        join_path_list = [model_path, "checkpoint-*", "eval_output",dataset_base_name, "*", "eval_output.json"]
        path_checkpoint_index = -5
        
    path_list = glob.glob(os.path.join(*join_path_list), recursive=True)
    path_list = sorted(path_list, key=lambda x: int(x.split("/")[path_checkpoint_index].split("-")[-1]))
    
    checkpoint_steps = []
    for path in path_list:
        checkpoint = int(path.split("/")[path_checkpoint_index].split("-")[-1])
        checkpoint_steps.append(checkpoint)
    
    checkpoint_step_path_dict = {}
    for path, step in zip(path_list, checkpoint_steps):
        if step not in checkpoint_step_path_dict:
            checkpoint_step_path_dict[step] = []
        checkpoint_step_path_dict[step].append(path)
    
    for key,value in checkpoint_step_path_dict.items():
        checkpoint_step_path_dict[key] = sorted(value, key=lambda x: x.split("/")[-2])[-1]
    
    path_list = checkpoint_step_path_dict.values()
    checkpoint_steps = checkpoint_step_path_dict.keys()
    checkpoint_steps = [int(step) for step in checkpoint_steps]

    assert len(set(checkpoint_steps)) == len(checkpoint_steps), "Checkpoint steps should be unique."
    

                
    for path in path_list:
        run_command = ["python", eval_script, "--generated_json", path]
        run_command.extend(other_command)
        subprocess.run(run_command, check=True)
        
    if args.wandb_run_id and args.wandb_project:
        print(f"Using WandB run ID: {args.wandb_run_id} and project: {args.wandb_project}")
        PROJECT = args.wandb_project
        RUN_ID = args.wandb_run_id
        script_base_name = os.path.basename(eval_script).split(".")[0]
        dataset_script_name = script_base_name + "_" + dataset_base_name
        if gen_args_folder:
            dataset_script_name += "_" + gen_args_folder
        
        if gen_args_folder:
            join_path_list = [model_path, "checkpoint-*", "eval_output",dataset_base_name, gen_args_folder, "*", f"{script_base_name}.json"]
        else:
            join_path_list = [model_path, "checkpoint-*", "eval_output",dataset_base_name, "*", f"{script_base_name}.json"]
            
        score_path_list = glob.glob(os.path.join(*join_path_list), recursive=True)
        score_path_list = sorted(score_path_list, key=lambda x: int(x.split("/")[path_checkpoint_index].split("-")[-1]))
        score_path_step_dict = {}
        for score_path in score_path_list:
            step = int(score_path.split("/")[path_checkpoint_index].split("-")[-1])
            if step not in score_path_step_dict:
                score_path_step_dict[step] = []
            score_path_step_dict[step].append(score_path)
            
        score_path_list = [score_path_step_dict[step][-1] for step in checkpoint_steps if step in score_path_step_dict]
        # import pdb; pdb.set_trace()
        if len(score_path_list) == 0:
            print(f"No score files found for {dataset_script_name}.")
            return
        
        assert len(score_path_list) == len(checkpoint_steps), "Score files should match checkpoint steps."
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
            scores = data.get("summary_scores", {})
            data_num = data.get("summary_data_num", {})
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
    parser.add_argument("--gen_args_folder", type=str, help='Path to the folder containing generated arguments for evaluation.', default=None)
    parser.add_argument("--eval_script", type=str, help='Path to the evaluation script.',required=True)
    parser.add_argument("--wandb_run_id", type=str, help='WandB run ID.')
    parser.add_argument("--wandb_project", type=str, help='WandB project name.')
    parser.add_argument("--other_command", type=csv_list, help='Additional command line arguments for the evaluation script.', default=[])
    args = parser.parse_args()
    print(f"Arguments: {args}")
    main(args)
