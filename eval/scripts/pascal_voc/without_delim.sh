python /home/omote/cluster_project/iam2/eval/eval_multi_run_multi_process.py \
    --wandb_entity katlab-gifu \
    --wandb_project kosmos-2_pascalvoc-one-class_epoch20 \
    --model_path_list \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/without-delim_train-vision-proj-llm_cross-entropy_2025-07-16T19_07_33 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/without-delim_train-vision-proj-llm_distance-loss_2025-07-16T19_09_23 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/without-delim_train-vision-proj-llm_distance-forward-kl-loss_2025-07-16T20_52_19 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/without-delim_include-input-tokens_train-vision-proj-llm_cross-entropy_2025-07-16T20_57_36 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/without-delim_include-input-tokens_train-vision-proj-llm_distance-loss_2025-07-16T22_40_06 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/without-delim_include-input-tokens_train-vision-proj-llm_distance-forward-kl-loss_2025-07-16T22_41_39 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/without-delim_train-proj-llm_cross-entropy_2025-07-17T10_54_59 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/without-delim_train-proj-llm_distance-loss_2025-07-17T10_59_59 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/without-delim_train-proj-llm_distance-forward-kl-loss_2025-07-17T12_19_46 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/without-delim_include-input-tokens_train-proj-llm_cross-entropy_2025-07-17T12_29_00 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/without-delim_include-input-tokens_train-proj-llm_distance-loss_2025-07-17T13_48_04 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/without-delim_include-input-tokens_train-proj-llm_distance-forward-kl-loss_2025-07-17T13_52_56 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/without-delim_train-vision-proj-llm_cross-entropy_2025-07-18T00_38_06 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/without-delim_train-vision-proj-llm_distance-loss_2025-07-18T00_51_04 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/without-delim_train-vision-proj-llm_distance-forward-kl-loss_2025-07-18T02_25_37 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/without-delim_include-input-tokens_train-vision-proj-llm_cross-entropy_2025-07-18T02_40_04 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/without-delim_include-input-tokens_train-vision-proj-llm_distance-loss_2025-07-18T04_14_33 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/without-delim_include-input-tokens_train-vision-proj-llm_distance-forward-kl-loss_2025-07-18T04_26_36 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/without-delim_train-proj-llm_cross-entropy_2025-07-18T15_51_19 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/without-delim_train-proj-llm_distance-loss_2025-07-18T16_03_26 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/without-delim_train-proj-llm_distance-forward-kl-loss_2025-07-18T17_17_56 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/without-delim_include-input-tokens_train-proj-llm_cross-entropy_2025-07-18T17_32_26 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/without-delim_include-input-tokens_train-proj-llm_distance-loss_2025-07-18T18_46_30 \
    /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2_pascalvoc-one-class_epoch20/without-delim_include-input-tokens_train-proj-llm_distance-forward-kl-loss_2025-07-18T18_59_17 \
    --dataset_base_name val_pascal-voc-one-class_for-kosmos2_one_class_without_delim \
    --gen_args_folder max_new_tokens=512-temperature=1.0-top_p=1.0-top_k=50-num_beams=1-do_sample=False- \
    --eval_script /home/omote/cluster_project/iam2/eval/pascal-voc/pascal-voc.py \
    # --other_command="--gt_json,/data_ssd/PASCAL-VOC/val_pascal-voc-one-class_for-kosmos2_one_class_without_delim_noline.json"

