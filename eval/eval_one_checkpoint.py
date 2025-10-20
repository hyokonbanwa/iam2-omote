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
    
    # if gen_args_folder:
    #     join_path_list = [model_path, "checkpoint-*", "eval_output",dataset_base_name, gen_args_folder, "*", "eval_output.json"]
    # else:
    #     join_path_list = [model_path, "checkpoint-*", "eval_output",dataset_base_name, "*", "eval_output.json"]
    if gen_args_folder:
        join_path_list = [model_path, "eval_output",dataset_base_name, gen_args_folder, "*", "eval_output.json"]
    else:
        join_path_list = [model_path, "eval_output",dataset_base_name, "*", "eval_output.json"]
    # import pdb; pdb.set_trace()
    path = glob.glob(os.path.join(*join_path_list), recursive=True)[-1]
    
    if not os.path.exists(path):
        print(f"No eval_output.json found in {os.path.join(model_path, 'eval_output', dataset_base_name, '*')}.")
        return
    
    if "checkpoint-" not in model_path.split("/")[-1]:
        checkpoint_step = -1
    else:
        checkpoint_step = int(model_path.split("/")[-1].split("-")[-1])
    # import pdb; pdb.set_trace()
    run_command = ["python", eval_script, "--generated_json", path]
    run_command.extend(other_command)
    subprocess.run(run_command, check=True)

    model_folder_name_list = [os.path.basename(model_path) if not os.path.basename(model_path).startswith("checkpoint-") else model_path.split("/")[-2]]
    wandb_run_id_list = []
    if args.wandb_run_name and args.wandb_entity and not args.wandb_make_run and args.wandb_run_id is None:
        api = wandb.Api()
        runs = api.runs(os.path.join(args.wandb_entity,args.wandb_project))
        run_name_list = [run.name for run in runs]
        assert len(set(run_name_list)) == len(run_name_list), "Run names should be unique."
        run_name_to_id = {run.name: run.id for run in runs}
        if args.wandb_run_name in run_name_to_id:
            wandb_run_id_list.append(run_name_to_id[args.wandb_run_name])
        assert len(wandb_run_id_list) == 1, "WandB run IDs should match the number of model paths."

    if args.wandb_entity and (args.wandb_run_name is None and args.wandb_run_id is None) and not args.wandb_make_run:
        api = wandb.Api()
        runs = api.runs(os.path.join(args.wandb_entity,args.wandb_project))
        run_name_list = [run.name for run in runs]
        assert len(set(run_name_list)) == len(run_name_list), "Run names should be unique."
        run_name_to_id = {run.name: run.id for run in runs}
        for model_folder_name in model_folder_name_list:
            wandb_run_id_list.append(run_name_to_id[model_folder_name])
                
        assert len(wandb_run_id_list) == 1, "WandB run IDs should match the number of model paths."
    # else:
    #     print("No WandB entity provided, skipping WandB run ID retrieval.")
    #     wandb_run_id_list = 1 * [None]

    if len(wandb_run_id_list) == 0:
        wandb_run_id_list = [None]

    PROJECT = args.wandb_project
    RUN_ID = args.wandb_run_id if args.wandb_run_id else wandb_run_id_list[0]
    RUN_NAME = args.wandb_run_name
    if args.wandb_make_run and PROJECT and (RUN_ID is None):
        print(f"Creating a new WandB run in project: {PROJECT} with name: {RUN_NAME}")
        run = wandb.init(
            project=PROJECT,
            name=RUN_NAME
        )
        RUN_ID = run.id
        run.finish()


    if PROJECT and RUN_ID:
        print(f"Using WandB run ID: {args.wandb_run_id} and project: {args.wandb_project}")

        script_base_name = os.path.basename(eval_script).split(".")[0]
        dataset_script_name = script_base_name + "_" + dataset_base_name
        if gen_args_folder:
            dataset_script_name += "_" + gen_args_folder
            
        if gen_args_folder:
            join_path_list = [model_path, "eval_output",dataset_base_name, gen_args_folder, "*", f"{script_base_name}.json"]
        else:
            join_path_list = [model_path, "eval_output",dataset_base_name, "*", f"{script_base_name}.json"]
        
        print(f"Looking for score files in: {os.path.join(*join_path_list)}")
        score_path = glob.glob(os.path.join(*join_path_list),recursive=True)[-1]
        if not os.path.exists(score_path):
            print(f"No score files found for {dataset_script_name}.")
            return

        run = wandb.init(
            project=PROJECT,
            id=RUN_ID,
            resume="allow")            # 既存 run があれば再開、無ければエラー無し :contentReference[oaicite:0]{index=0}

        # optional: グラフを学習と同じ軸に重ねたい時


        scores_dict = {}


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
                scores_dict[f"{dataset_script_name}/{k}"] = v
        scores = tmp_scores
        # print(f"Scores: {scores}")
        # run.log({                # history に行を追加
        #     **scores,            # スコアを追加
        #     f"{dataset_script_name}/step":         step     # or step=step に渡す
        # })  # step を指定してログする
        
        #各スコアを summary に追加
        scores_values = {f"{k}/max": v for k, v in scores_dict.items()}  # 各スコアの最大値を取得
        run.summary.update(scores_values)
        scores_values = {f"{k}/min": v for k, v in scores_dict.items()}  # 各スコアの最大値を取得
        run.summary.update(scores_values)
        max_scores_steps  = { f"{dataset_script_name}/step":checkpoint_step}  # 各スコアの最大値のステップを取得
        run.summary.update(max_scores_steps)  # 各スコアの最大値のステップを summary に追加
        
        # ★run を明示的に閉じる
        run.finish()


def csv_list(string):
    # 空文字や余分な空白もケア
    return [s.strip() for s in string.split(',') if s.strip()]

#python /home/omote/cluster_project/iam2/eval/eval_all_checkpoint.py --model_path a --dataset_base_name b --eval_script c --other_command="--x,--b"
#Namespace(model_path='a', dataset_base_name='b', eval_script='c', run_id=None, project=None, other_command=['--x', '--b'])
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utility functions for JSON handling and sorting.")
    parser.add_argument("--wandb_entity", type=str, help="your_entity",default=None)
    parser.add_argument("--model_path", type=str, help='Path to the model directory containing checkpoints.',required=True)
    parser.add_argument("--dataset_base_name", type=str, help='Base name of the dataset for evaluation.',required=True)
    parser.add_argument("--wandb_make_run", action='store_true', help='Whether to create a new WandB run.')
    parser.add_argument("--gen_args_folder", type=str, help='Path to the folder containing generated arguments for evaluation.', default=None)
    parser.add_argument("--eval_script", type=str, help='Path to the evaluation script.',required=True)
    parser.add_argument("--wandb_run_id", type=str, help='WandB run ID.')
    parser.add_argument("--wandb_project", type=str, help='WandB project name.')
    parser.add_argument("--wandb_run_name", type=str, help='WandB run name.', default=None)
    parser.add_argument("--other_command", type=csv_list, help='Additional command line arguments for the evaluation script.', default=[])
    args = parser.parse_args()
    print(f"Arguments: {args}")
    main(args)
