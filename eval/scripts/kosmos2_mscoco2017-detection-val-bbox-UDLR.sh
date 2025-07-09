python \
    /home/omote/cluster_project/iam2/eval/eval_all_checkpoint.py \
    --model_path /home/omote/omote-data-ssd/iam-llms-finetune/experiment_output/kosmos-2_mscoco2017-detection/mscoco2017-detection-bbox-UDLR_train-vision-proj-llm_cross-entropy_2025-07-04T10_37_39 \
    --dataset_base_name val_bbox-UDLR_for-kosmos2_mscoco2017-detection \
    --eval_script /home/omote/cluster_project/iam2/eval/mscoco-detection/mscoco_detection.py \
    --other_command="--gt_json,/data_ssd/mscoco-detection/val_bbox-UDLR_for-kosmos2_mscoco2017-detection.json" \
    --wandb_project kosmos-2_mscoco2017-detection \
    --wandb_run_id 2q1drf1o

python \
    /home/omote/cluster_project/iam2/eval/eval_all_checkpoint.py \
    --model_path /home/omote/omote-data-ssd/iam-llms-finetune/experiment_output/kosmos-2_mscoco2017-detection/mscoco2017-detection-bbox-UDLR_train-vision-proj-llm_distance-loss_2025-07-04T12_12_23 \
    --dataset_base_name val_bbox-UDLR_for-kosmos2_mscoco2017-detection \
    --eval_script /home/omote/cluster_project/iam2/eval/mscoco-detection/mscoco_detection.py \
    --other_command="--gt_json,/data_ssd/mscoco-detection/val_bbox-UDLR_for-kosmos2_mscoco2017-detection.json" \
    --wandb_project kosmos-2_mscoco2017-detection \
    --wandb_run_id mpbs1hha

python \
    /home/omote/cluster_project/iam2/eval/eval_all_checkpoint.py \
    --model_path /home/omote/omote-data-ssd/iam-llms-finetune/experiment_output/kosmos-2_mscoco2017-detection/mscoco2017-detection-bbox-UDLR_train-vision-proj-llm_distance-forward-kl-loss_2025-07-04T14_34_26 \
    --dataset_base_name val_bbox-UDLR_for-kosmos2_mscoco2017-detection \
    --eval_script /home/omote/cluster_project/iam2/eval/mscoco-detection/mscoco_detection.py \
    --other_command="--gt_json,/data_ssd/mscoco-detection/val_bbox-UDLR_for-kosmos2_mscoco2017-detection.json" \
    --wandb_project kosmos-2_mscoco2017-detection \
    --wandb_run_id pzv0xw0w

