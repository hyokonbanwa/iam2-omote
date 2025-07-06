python \
    /home/omote/cluster_project/iam2/eval/eval_one_checkpoint.py \
    --model_path /home/omote/omote-data-ssd/iam-llms-finetune/experiment_output/kosmos-2_refcoco-g/refcoco-g_train-and-eval_vision-proj-llm_cross-entorpy_2025-06-28T10_37_15/checkpoint-8806 \
    --dataset_base_name refcoco_g_kosmos2_test \
    --eval_script /home/omote/cluster_project/iam2/eval/refcoco/refcoco_iou_score.py \
    --other_command="--gt_json,/data_ssd/refcoco_g/refcoco_g_kosmos2_test.json" \
    --wandb_project kosmos-2_refcoco-g \
    --wandb_run_id pab8e66q

python \
    /home/omote/cluster_project/iam2/eval/eval_one_checkpoint.py \
    --model_path /home/omote/omote-data-ssd/iam-llms-finetune/experiment_output/kosmos-2_refcoco-g/refcoco-g_train-and-eval_vision-proj-llm_distance-loss_2025-06-28T14_34_24/checkpoint-3145 \
    --dataset_base_name refcoco_g_kosmos2_test \
    --eval_script /home/omote/cluster_project/iam2/eval/refcoco/refcoco_iou_score.py \
    --other_command="--gt_json,/data_ssd/refcoco_g/refcoco_g_kosmos2_test.json" \
    --wandb_project kosmos-2_refcoco-g \
    --wandb_run_id 2hulbsyo

python \
    /home/omote/cluster_project/iam2/eval/eval_one_checkpoint.py \
    --model_path /home/omote/omote-data-ssd/iam-llms-finetune/experiment_output/kosmos-2_refcoco-g/refcoco-g_train-and-eval_vision-proj-llm_distance-forward-kl-loss_2025-06-28T18_38_23/checkpoint-9435 \
    --dataset_base_name refcoco_g_kosmos2_test \
    --eval_script /home/omote/cluster_project/iam2/eval/refcoco/refcoco_iou_score.py \
    --other_command="--gt_json,/data_ssd/refcoco_g/refcoco_g_kosmos2_test.json" \
    --wandb_project kosmos-2_refcoco-g \
    --wandb_run_id f2rkhli1
