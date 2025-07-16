python /home/omote/cluster_project/iam2/eval/eval_multi_run.py \
    --wandb_entity katlab-gifu \
    --wandb_project kosmos-2_pascalvoc-one-class \
    --model_path_list \
    /home/omote/omote-data-ssd/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class/with-delim_include-input-tokens_train-vision-proj-llm_cross-entropy_2025-07-14T19_34_57 \
    /home/omote/omote-data-ssd/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class/with-delim_include-input-tokens_train-vision-proj-llm_distance-forward-kl-loss_2025-07-14T22_13_27 \
    /home/omote/omote-data-ssd/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class/with-delim_include-input-tokens_train-vision-proj-llm_distance-loss_2025-07-14T22_13_22 \
    /home/omote/omote-data-ssd/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class/with-delim_train-vision-proj-llm_cross-entropy_2025-07-14T16_51_39 \
    /home/omote/omote-data-ssd/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class/with-delim_train-vision-proj-llm_distance-forward-kl-loss_2025-07-14T19_31_11 \
    /home/omote/omote-data-ssd/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class/with-delim_train-vision-proj-llm_distance-loss_2025-07-14T16_51_46 \
    --dataset_base_name val_pascal-voc-one-class_for-kosmos2_one_class_with_delim \
    --gen_args_folder max_new_tokens=512-temperature=1.0-top_p=1.0-top_k=50-num_beams=1-do_sample=False- \
    --eval_script /home/omote/cluster_project/iam2/eval/pascal-voc/pascal-voc.py \
    # --other_command="--gt_json,/data_ssd/PASCAL-VOC/val_pascal-voc-one-class_for-kosmos2_one_class_without_delim_noline.json"

