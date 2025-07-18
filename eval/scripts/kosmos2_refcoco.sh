# python \
#     /home/omote/cluster_project/iam2/eval/eval_all_checkpoint.py \
#     --model_path /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2/refcoco-pulus_train-and-eval_proj-llm_cross-entorpy_2025-06-27T10_52_16 \
#     --dataset_base_name refcoco_plus_kosmos2_validation \
#     --eval_script /home/omote/cluster_project/iam2/eval/refcoco/refcoco_iou_score.py \
#     --other_command="--gt_json,/data_ssd/refcoco_plus/refcoco_plus_kosmos2_validation.json" \
#     --wandb_project kosmos-2 \
#     --wandb_run_id gjwt93mp


# python \
#     /home/omote/cluster_project/iam2/eval/eval_all_checkpoint.py \
#     --model_path /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2/refcoco-pulus_train-and-eval_vision-proj-llm_cross-entorpy_2025-06-27T10_39_47 \
#     --dataset_base_name refcoco_plus_kosmos2_validation \
#     --eval_script /home/omote/cluster_project/iam2/eval/refcoco/refcoco_iou_score.py \
#     --other_command="--gt_json,/data_ssd/refcoco_plus/refcoco_plus_kosmos2_validation.json" \
#     --wandb_project kosmos-2 \
#     --wandb_run_id bpv2n7rx

python \
    /home/omote/cluster_project/iam2/eval/eval_all_checkpoint.py \
    --model_path /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2/refcoco-pulus_train-and-eval_proj-llm_distance-loss_2025-06-27T10_53_09 \
    --dataset_base_name refcoco_plus_kosmos2_validation \
    --eval_script /home/omote/cluster_project/iam2/eval/refcoco/refcoco_iou_score.py \
    --other_command="--gt_json,/data_ssd/refcoco_plus/refcoco_plus_kosmos2_validation.json" \
    --wandb_project kosmos-2 \
    --wandb_run_id 1zkpp2oc


# python \
#     /home/omote/cluster_project/iam2/eval/eval_all_checkpoint.py \
#     --model_path /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2/refcoco-pulus_train-and-eval_vision-proj-llm_distance-loss_2025-06-27T10_47_41 \
#     --dataset_base_name refcoco_plus_kosmos2_validation \
#     --eval_script /home/omote/cluster_project/iam2/eval/refcoco/refcoco_iou_score.py \
#     --other_command="--gt_json,/data_ssd/refcoco_plus/refcoco_plus_kosmos2_validation.json" \
#     --wandb_project kosmos-2 \
#     --wandb_run_id z83w7bjd




python \
    /home/omote/cluster_project/iam2/eval/eval_all_checkpoint.py \
    --model_path /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2/refcoco-pulus_train-and-eval_proj-llm_distance-forward-kl-loss_2025-06-27T14_28_21 \
    --dataset_base_name refcoco_plus_kosmos2_validation \
    --eval_script /home/omote/cluster_project/iam2/eval/refcoco/refcoco_iou_score.py \
    --other_command="--gt_json,/data_ssd/refcoco_plus/refcoco_plus_kosmos2_validation.json" \
    --wandb_project kosmos-2 \
    --wandb_run_id 7j44ep2i


# python \
#     /home/omote/cluster_project/iam2/eval/eval_all_checkpoint.py \
#     --model_path /data_ssd/USER_DATA/omote/iam-llms-finetune/experiment_output/kosmos-2/refcoco-pulus_train-and-eval_vision-proj-llm_distance-forward-kl-loss_2025-06-27T15_22_23 \
#     --dataset_base_name refcoco_plus_kosmos2_validation \
#     --eval_script /home/omote/cluster_project/iam2/eval/refcoco/refcoco_iou_score.py \
#     --other_command="--gt_json,/data_ssd/refcoco_plus/refcoco_plus_kosmos2_validation.json" \
#     --wandb_project kosmos-2 \
#     --wandb_run_id h4e16ryc
