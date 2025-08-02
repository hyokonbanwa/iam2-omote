python /home/omote/cluster_project/iam2/eval/eval_multi_run_multi_process.py \
    --wandb_entity katlab-gifu \
    --wandb_project kosmos-2_pascalvoc-one-class_epoch20 \
    --model_path_list \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/with-delim_train-vision-proj-llm_cross-entropy_2025-07-16T13_48_51 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/with-delim_train-vision-proj-llm_distance-loss_2025-07-16T13_49_02 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/with-delim_train-vision-proj-llm_distance-forward-kl-loss_2025-07-16T15_30_04 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/with-delim_include-input-tokens_train-vision-proj-llm_cross-entropy_2025-07-16T15_37_27 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/with-delim_include-input-tokens_train-vision-proj-llm_distance-loss_2025-07-16T17_18_21 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/with-delim_include-input-tokens_train-vision-proj-llm_distance-forward-kl-loss_2025-07-16T17_19_21 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/with-delim_train-proj-llm_cross-entropy_2025-07-17T05_42_55 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/with-delim_train-proj-llm_distance-loss_2025-07-17T05_46_50 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/with-delim_train-proj-llm_distance-forward-kl-loss_2025-07-17T07_14_19 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/with-delim_include-input-tokens_train-proj-llm_cross-entropy_2025-07-17T07_38_56 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/with-delim_include-input-tokens_train-proj-llm_distance-loss_2025-07-17T09_04_22 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/with-delim_include-input-tokens_train-proj-llm_distance-forward-kl-loss_2025-07-17T09_08_55 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/with-delim_train-vision-proj-llm_cross-entropy_2025-07-17T19_34_45 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/with-delim_train-vision-proj-llm_distance-loss_2025-07-17T19_39_25 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/with-delim_train-vision-proj-llm_distance-forward-kl-loss_2025-07-17T21_14_52 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/with-delim_include-input-tokens_train-vision-proj-llm_distance-forward-kl-loss_2025-07-17T21_23_34 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/with-delim_include-input-tokens_train-vision-proj-llm_cross-entropy_2025-07-17T22_58_13 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/with-delim_include-input-tokens_train-vision-proj-llm_distance-loss_2025-07-17T23_06_34 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/with-delim_train-proj-llm_cross-entropy_2025-07-18T11_25_50 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/with-delim_train-proj-llm_distance-loss_2025-07-18T11_38_22 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/with-delim_train-proj-llm_distance-forward-kl-loss_2025-07-18T12_49_47 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/with-delim_include-input-tokens_train-proj-llm_cross-entropy_2025-07-18T13_08_11 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/with-delim_include-input-tokens_train-proj-llm_distance-loss_2025-07-18T14_20_28 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/with-delim_include-input-tokens_train-proj-llm_distance-forward-kl-loss_2025-07-18T14_31_08 \
    --dataset_base_name val_pascal-voc-one-class_for-kosmos2_one_class_with_delim \
    --gen_args_folder max_new_tokens=512-temperature=1.0-top_p=1.0-top_k=50-num_beams=1-do_sample=False- \
    --eval_script /home/omote/cluster_project/iam2/eval/pascal-voc/pascal-voc.py \
    # --other_command="--gt_json,/data_ssd/PASCAL-VOC/val_pascal-voc-one-class_for-kosmos2_one_class_without_delim_noline.json"

