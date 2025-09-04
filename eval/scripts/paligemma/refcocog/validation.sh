python /home/omote/cluster_project/iam2/eval/eval_multi_run_multi_process.py \
    --wandb_entity katlab-gifu \
    --wandb_project "paligemma_refcocog_epoch10" \
    --model_path_list \
    /home/omote/omote-data-ssd/iam-llms-finetune/experiment_output/paligemma_refcocog_epoch10/train-vision-proj-llm_cross-entropy_lr1e-05_2025-09-02T18_39_11 \
    --dataset_base_name refcoco_g_paligemma_validation \
    --gen_args_folder max_new_tokens=128-temperature=1.0-top_p=1.0-top_k=50-num_beams=1-do_sample=False- \
    --eval_script /home/omote/cluster_project/iam2/eval/refcoco/refcoco_iou_score.py \
    --other_command="--model,paligemma,-gt,/data_ssd/refcoco_g/refcoco_g_paligemma_validation.json"

