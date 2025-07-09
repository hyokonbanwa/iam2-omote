# python \
#     /home/omote/cluster_project/iam2/eval/eval_all_checkpoint.py \
#     --model_path /home/omote/omote-data-ssd/iam-llms-finetune/experiment_output/kosmos-2_mscoco2017-detection/mscoco2017-detection_train-vision-proj-llm_cross-entropy_2025-07-03T12_51_20 \
#     --dataset_base_name val_for-kosmos2_mscoco2017-detection \
#     --eval_script /home/omote/cluster_project/iam2/eval/mscoco-detection/mscoco_detection.py \
#     --other_command="--gt_json,/data_ssd/mscoco-detection/val_for-kosmos2_mscoco2017-detection.json" \
#     --wandb_project kosmos-2_mscoco2017-detection \
#     --wandb_run_id omt3fi1e

python \
    /home/omote/cluster_project/iam2/eval/eval_all_checkpoint.py \
    --model_path /home/omote/omote-data-ssd/iam-llms-finetune/experiment_output/kosmos-2_mscoco2017-detection/mscoco2017-detection_train-vision-proj-llm_distance-loss_2025-07-03T12_52_38 \
    --dataset_base_name val_for-kosmos2_mscoco2017-detection \
    --eval_script /home/omote/cluster_project/iam2/eval/mscoco-detection/mscoco_detection.py \
    --other_command="--gt_json,/data_ssd/mscoco-detection/val_for-kosmos2_mscoco2017-detection.json" \
    --wandb_project kosmos-2_mscoco2017-detection \
    --wandb_run_id vvps9vzg

python \
    /home/omote/cluster_project/iam2/eval/eval_all_checkpoint.py \
    --model_path /home/omote/omote-data-ssd/iam-llms-finetune/experiment_output/kosmos-2_mscoco2017-detection/mscoco2017-detection_train-vision-proj-llm_distance-forward-kl-loss_2025-07-03T16_46_51 \
    --dataset_base_name val_for-kosmos2_mscoco2017-detection \
    --eval_script /home/omote/cluster_project/iam2/eval/mscoco-detection/mscoco_detection.py \
    --other_command="--gt_json,/data_ssd/mscoco-detection/val_for-kosmos2_mscoco2017-detection.json" \
    --wandb_project kosmos-2_mscoco2017-detection \
    --wandb_run_id 842p13b9
