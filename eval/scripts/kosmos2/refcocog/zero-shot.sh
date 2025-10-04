# python \
#     /home/omote/cluster_project/iam2/eval/eval_one_checkpoint.py \
#     --model_path /home/omote/omote-data-ssd/iam-llms-finetune/original_model_weight/kosmos-2-patch14-224 \
#     --dataset_base_name refcoco_g_kosmos2_validation \
#     --eval_script /home/omote/cluster_project/iam2/eval/refcoco/refcoco_iou_score.py \
#     --gen_args_folder max_new_tokens=64-temperature=1.0-top_p=1.0-top_k=50-num_beams=1-do_sample=False- \
#     --other_command="--model,kosmos-2,-gt,/data_ssd/refcoco_g/refcoco_g_kosmos2_validation.json" \
#     --wandb_make_run \
#     --wandb_run_name zero-shot \
#     --wandb_project kosmos-2_refcoco-g \

python \
    /home/omote/cluster_project/iam2/eval/eval_one_checkpoint.py \
    --model_path /home/omote/omote-data-ssd/iam-llms-finetune/original_model_weight/kosmos-2-patch14-224 \
    --dataset_base_name refcoco_g_kosmos2_validation \
    --eval_script /home/omote/cluster_project/iam2/eval/refcoco/refcoco_iou_score.py \
    --gen_args_folder max_new_tokens=64-temperature=1.0-top_p=1.0-top_k=50-num_beams=1-do_sample=False- \
    --other_command="--model,kosmos-2,-gt,/data_ssd/refcoco_g/refcoco_g_kosmos2_validation.json" \
    --wandb_project kosmos-2_refcoco-g \
    --wandb_run_id re70q3jw

# python \
#     /home/omote/cluster_project/iam2/eval/eval_one_checkpoint.py \
#     --model_path /home/omote/omote-data-ssd/iam-llms-finetune/original_model_weight/kosmos-2-patch14-224 \
#     --dataset_base_name refcoco_g_kosmos2_test \
#     --eval_script /home/omote/cluster_project/iam2/eval/refcoco/refcoco_iou_score.py \
#     --gen_args_folder max_new_tokens=64-temperature=1.0-top_p=1.0-top_k=50-num_beams=1-do_sample=False- \
#     --other_command="--model,kosmos-2,-gt,/data_ssd/refcoco_g/refcoco_g_kosmos2_test.json" \
#     --wandb_project kosmos-2_refcoco-g \
#     --wandb_run_id re70q3jw
