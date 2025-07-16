python /home/omote/cluster_project/iam2/eval/eval_multi_run.py \
    --wandb_entity katlab-gifu \
    --wandb_project kosmos-2_pascalvoc-one-class \
    --model_path_list \
    /home/omote/omote-data-ssd/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class/without-delim-noline_include-input-tokens_train-vision-proj-llm_distance-loss_2025-07-15T18_34_00 \
    --dataset_base_name val_pascal-voc-one-class_for-kosmos2_one_class_without_delim_noline \
    --gen_args_folder max_new_tokens=512-temperature=1.0-top_p=1.0-top_k=50-num_beams=1-do_sample=False- \
    --eval_script /home/omote/cluster_project/iam2/eval/pascal-voc/pascal-voc.py \
    # --other_command="--gt_json,/data_ssd/PASCAL-VOC/val_pascal-voc-one-class_for-kosmos2_one_class_without_delim_noline.json"

