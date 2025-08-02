import wandb
import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from contextlib import redirect_stdout

def evaluate_model(model_path,wandb_run_id,args):
    with redirect_stdout(open(os.devnull, 'w')):
        try:
            run_command = ["python", "/home/omote/cluster_project/iam2/eval/eval_all_checkpoint.py",
                        "--model_path", model_path,
                        "--dataset_base_name", args.dataset_base_name,
                        "--gen_args_folder", args.gen_args_folder,
                        "--eval_script", args.eval_script,
                        f"--other_command={args.other_command}",
                        ]
            if args.wandb_entity and args.wandb_project and wandb_run_id:
                run_command.extend(["--wandb_run_id", wandb_run_id, "--wandb_project", args.wandb_project])
            print(f"Running command: {' '.join(run_command)}")
            subprocess.run(run_command, check=True)
            return True
        
        except Exception as e:
            print(f"Error running evaluation for model {model_path}: {e}")
            return False
    
    
def main(args):
 
    assert type(args.model_path_list) == list, "model_path_list should be a list of strings."
    model_path_list = args.model_path_list
    for model_path in model_path_list:
        assert os.path.exists(model_path), f"Model path {model_path} does not exist."
    model_folder_name_list = [os.path.basename(model_path) for model_path in model_path_list]
    assert len(set(model_folder_name_list)) == len(model_folder_name_list), "Model folder names should be unique."
    
    wandb_run_id_list = []
    if args.wandb_entity:
        api = wandb.Api()
        runs = api.runs(os.path.join(args.wandb_entity,args.wandb_project))
        run_name_list = [run.name for run in runs]
        assert len(set(run_name_list)) == len(run_name_list), "Run names should be unique."
        run_name_to_id = {run.name: run.id for run in runs}
        for model_folder_name in model_folder_name_list:
            wandb_run_id_list.append(run_name_to_id[model_folder_name])
                
        assert len(wandb_run_id_list) == len(model_path_list), "WandB run IDs should match the number of model paths."
    else:
        print("No WandB entity provided, skipping WandB run ID retrieval.")
        wandb_run_id_list = [None] * len(model_path_list)

    result_list = []
    with tqdm(total=len(model_path_list), desc="Processing dataset") as pbar:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for model_path ,wandb_run_id in zip(model_path_list, wandb_run_id_list):
                futures.append(executor.submit(evaluate_model, model_path, wandb_run_id, args))
            for future in as_completed(futures):
                pbar.update(1)
                # Check if the future completed successfully
                result = future.result()
                if result:
                    print(f"Evaluation completed successfully for model {model_path}.")
                else:
                    print(f"Evaluation failed for model {model_path}.")
                result_list.append(result)
    
    print("All evaluations completed.")
    print(f"Completed evaluations: {sum(result_list)}/{len(result_list)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utility functions for JSON handling and sorting.")
    parser.add_argument("--wandb_entity", type=str, help="your_entity",default=None)
    parser.add_argument("--wandb_project", type=str, help="your_project",default=None)
    parser.add_argument("--model_path_list", nargs='+',help='Path to the model directory containing checkpoints.',required=True,default=[])
    parser.add_argument("--dataset_base_name", type=str, help='Base name of the dataset for evaluation.',required=True)
    parser.add_argument("--gen_args_folder", type=str, help='Path to the folder containing generated arguments for evaluation.', default=None)
    parser.add_argument("--eval_script", type=str, help='Path to the evaluation script.',required=True)
    parser.add_argument("--other_command", type=str, help='Additional command line arguments for the evaluation script.', default="")
    args = parser.parse_args()
    print(f"Arguments: {args}")
    main(args)
