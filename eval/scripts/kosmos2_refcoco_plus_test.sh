#all
python \
    /home/omote/cluster_project/iam2/eval/eval_one_checkpoint.py \
    --model_path /home/omote/omote-data-ssd/iam-llms-finetune/experiment_output/kosmos-2/refcoco-pulus_train-and-eval_vision-proj-llm_cross-entorpy_2025-06-27T10_39_47/checkpoint-15963 \
    --dataset_base_name refcoco_plus_kosmos2_test \
    --eval_script /home/omote/cluster_project/iam2/eval/refcoco/refcoco_iou_score.py \
    --other_command="--gt_json,/data_ssd/refcoco_plus/refcoco_plus_kosmos2_test.json" \
    --wandb_project kosmos-2 \
    --wandb_run_id bpv2n7rx

python \
    /home/omote/cluster_project/iam2/eval/eval_one_checkpoint.py \
    --model_path /home/omote/omote-data-ssd/iam-llms-finetune/experiment_output/kosmos-2/refcoco-pulus_train-and-eval_vision-proj-llm_cross-entorpy_2025-06-27T10_39_47/checkpoint-15963 \
    --dataset_base_name refcoco_plus_kosmos2_testB \
    --eval_script /home/omote/cluster_project/iam2/eval/refcoco/refcoco_iou_score.py \
    --other_command="--gt_json,/data_ssd/refcoco_plus/refcoco_plus_kosmos2_testB.json" \
    --wandb_project kosmos-2 \
    --wandb_run_id bpv2n7rx


#all
python \
    /home/omote/cluster_project/iam2/eval/eval_one_checkpoint.py \
    --model_path /home/omote/omote-data-ssd/iam-llms-finetune/experiment_output/kosmos-2/refcoco-pulus_train-and-eval_vision-proj-llm_distance-loss_2025-06-27T10_47_41/checkpoint-15963 \
    --dataset_base_name refcoco_plus_kosmos2_test \
    --eval_script /home/omote/cluster_project/iam2/eval/refcoco/refcoco_iou_score.py \
    --other_command="--gt_json,/data_ssd/refcoco_plus/refcoco_plus_kosmos2_test.json" \
    --wandb_project kosmos-2 \
    --wandb_run_id z83w7bjd

python \
    /home/omote/cluster_project/iam2/eval/eval_one_checkpoint.py \
    --model_path /home/omote/omote-data-ssd/iam-llms-finetune/experiment_output/kosmos-2/refcoco-pulus_train-and-eval_vision-proj-llm_distance-loss_2025-06-27T10_47_41/checkpoint-15963 \
    --dataset_base_name refcoco_plus_kosmos2_testB \
    --eval_script /home/omote/cluster_project/iam2/eval/refcoco/refcoco_iou_score.py \
    --other_command="--gt_json,/data_ssd/refcoco_plus/refcoco_plus_kosmos2_testB.json" \
    --wandb_project kosmos-2 \
    --wandb_run_id z83w7bjd

#all
python \
    /home/omote/cluster_project/iam2/eval/eval_one_checkpoint.py \
    --model_path /home/omote/omote-data-ssd/iam-llms-finetune/experiment_output/kosmos-2/refcoco-pulus_train-and-eval_vision-proj-llm_distance-forward-kl-loss_2025-06-27T15_22_23/checkpoint-15024 \
    --dataset_base_name refcoco_plus_kosmos2_test \
    --eval_script /home/omote/cluster_project/iam2/eval/refcoco/refcoco_iou_score.py \
    --other_command="--gt_json,/data_ssd/refcoco_plus/refcoco_plus_kosmos2_test.json" \
    --wandb_project kosmos-2 \
    --wandb_run_id h4e16ryc

python \
    /home/omote/cluster_project/iam2/eval/eval_one_checkpoint.py \
    --model_path /home/omote/omote-data-ssd/iam-llms-finetune/experiment_output/kosmos-2/refcoco-pulus_train-and-eval_vision-proj-llm_distance-forward-kl-loss_2025-06-27T15_22_23/checkpoint-15024 \
    --dataset_base_name refcoco_plus_kosmos2_testB \
    --eval_script /home/omote/cluster_project/iam2/eval/refcoco/refcoco_iou_score.py \
    --other_command="--gt_json,/data_ssd/refcoco_plus/refcoco_plus_kosmos2_testB.json" \
    --wandb_project kosmos-2 \
    --wandb_run_id h4e16ryc

#all
python \
    /home/omote/cluster_project/iam2/eval/eval_one_checkpoint.py \
    --model_path /home/omote/omote-data-ssd/iam-llms-finetune/experiment_output/kosmos-2/refcoco-pulus_train-and-eval_proj-llm_cross-entorpy_2025-06-27T10_52_16/checkpoint-17841 \
    --dataset_base_name refcoco_plus_kosmos2_test \
    --eval_script /home/omote/cluster_project/iam2/eval/refcoco/refcoco_iou_score.py \
    --other_command="--gt_json,/data_ssd/refcoco_plus/refcoco_plus_kosmos2_test.json" \
    --wandb_project kosmos-2 \
    --wandb_run_id gjwt93mp

python \
    /home/omote/cluster_project/iam2/eval/eval_one_checkpoint.py \
    --model_path /home/omote/omote-data-ssd/iam-llms-finetune/experiment_output/kosmos-2/refcoco-pulus_train-and-eval_proj-llm_cross-entorpy_2025-06-27T10_52_16/checkpoint-17841 \
    --dataset_base_name refcoco_plus_kosmos2_testB \
    --eval_script /home/omote/cluster_project/iam2/eval/refcoco/refcoco_iou_score.py \
    --other_command="--gt_json,/data_ssd/refcoco_plus/refcoco_plus_kosmos2_testB.json" \
    --wandb_project kosmos-2 \
    --wandb_run_id gjwt93mp

#all
python \
    /home/omote/cluster_project/iam2/eval/eval_one_checkpoint.py \
    --model_path /home/omote/omote-data-ssd/iam-llms-finetune/experiment_output/kosmos-2/refcoco-pulus_train-and-eval_proj-llm_distance-loss_2025-06-27T10_53_09/checkpoint-16902 \
    --dataset_base_name refcoco_plus_kosmos2_test \
    --eval_script /home/omote/cluster_project/iam2/eval/refcoco/refcoco_iou_score.py \
    --other_command="--gt_json,/data_ssd/refcoco_plus/refcoco_plus_kosmos2_test.json" \
    --wandb_project kosmos-2 \
    --wandb_run_id qz5f0tha

python \
    /home/omote/cluster_project/iam2/eval/eval_one_checkpoint.py \
    --model_path /home/omote/omote-data-ssd/iam-llms-finetune/experiment_output/kosmos-2/refcoco-pulus_train-and-eval_proj-llm_distance-loss_2025-06-27T10_53_09/checkpoint-16902 \
    --dataset_base_name refcoco_plus_kosmos2_testB \
    --eval_script /home/omote/cluster_project/iam2/eval/refcoco/refcoco_iou_score.py \
    --other_command="--gt_json,/data_ssd/refcoco_plus/refcoco_plus_kosmos2_testB.json" \
    --wandb_project kosmos-2 \
    --wandb_run_id qz5f0tha

#all
python \
    /home/omote/cluster_project/iam2/eval/eval_one_checkpoint.py \
    --model_path /home/omote/omote-data-ssd/iam-llms-finetune/experiment_output/kosmos-2/refcoco-pulus_train-and-eval_proj-llm_distance-forward-kl-loss_2025-06-27T14_28_21/checkpoint-16902 \
    --dataset_base_name refcoco_plus_kosmos2_test \
    --eval_script /home/omote/cluster_project/iam2/eval/refcoco/refcoco_iou_score.py \
    --other_command="--gt_json,/data_ssd/refcoco_plus/refcoco_plus_kosmos2_test.json" \
    --wandb_project kosmos-2 \
    --wandb_run_id uwk1hlmo

python \
    /home/omote/cluster_project/iam2/eval/eval_one_checkpoint.py \
    --model_path /home/omote/omote-data-ssd/iam-llms-finetune/experiment_output/kosmos-2/refcoco-pulus_train-and-eval_proj-llm_distance-forward-kl-loss_2025-06-27T14_28_21/checkpoint-16902 \
    --dataset_base_name refcoco_plus_kosmos2_testB \
    --eval_script /home/omote/cluster_project/iam2/eval/refcoco/refcoco_iou_score.py \
    --other_command="--gt_json,/data_ssd/refcoco_plus/refcoco_plus_kosmos2_testB.json" \
    --wandb_project kosmos-2 \
    --wandb_run_id uwk1hlmo

#all
# python \
#     /home/omote/cluster_project/iam2/eval/eval_one_checkpoint.py \
#     --model_path  \
#     --dataset_base_name refcoco_plus_kosmos2_test \
#     --eval_script /home/omote/cluster_project/iam2/eval/refcoco/refcoco_iou_score.py \
#     --other_command="--gt_json,/data_ssd/refcoco_plus/refcoco_plus_kosmos2_test.json" \
#     --wandb_project kosmos-2 \
#     --wandb_run_id

# python \
#     /home/omote/cluster_project/iam2/eval/eval_one_checkpoint.py \
#     --model_path  \
#     --dataset_base_name refcoco_plus_kosmos2_testB \
#     --eval_script /home/omote/cluster_project/iam2/eval/refcoco/refcoco_iou_score.py \
#     --other_command="--gt_json,/data_ssd/refcoco_plus/refcoco_plus_kosmos2_testB.json" \
#     --wandb_project kosmos-2 \
#     --wandb_run_id
