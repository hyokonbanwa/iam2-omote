python \
    /home/omote/cluster_project/iam2/eval/eval_one_checkpoint.py \
    --model_path /home/omote/omote-data-ssd/iam-llms-finetune/original_model_weight/paligemma-3b-pt-224 \
    --dataset_base_name refcoco_g_paligemma_validation \
    --eval_script /home/omote/cluster_project/iam2/eval/refcoco/refcoco_iou_score.py \
    --gen_args_folder max_new_tokens=128-temperature=1.0-top_p=1.0-top_k=50-num_beams=1-do_sample=False- \
    --other_command="--model,paligemma,-gt,/data_ssd/refcoco_g/refcoco_g_paligemma_validation.json" \
    --wandb_make_run \
    --wandb_run_name zero-shot \
    --wandb_project paligemma_refcocog_epoch10 \

python \
    /home/omote/cluster_project/iam2/eval/eval_one_checkpoint.py \
    --model_path /home/omote/omote-data-ssd/iam-llms-finetune/original_model_weight/paligemma-3b-pt-224 \
    --dataset_base_name refcoco_g_paligemma_validation \
    --eval_script /home/omote/cluster_project/iam2/eval/refcoco/refcoco_iou_score.py \
    --gen_args_folder max_new_tokens=128-temperature=1.0-top_p=1.0-top_k=50-num_beams=1-do_sample=False- \
    --other_command="--model,paligemma,-gt,/data_ssd/refcoco_g/refcoco_g_paligemma_validation.json" \
    --wandb_project paligemma_refcocog_epoch10 \
    --wandb_run_id uqunhykw

python \
    /home/omote/cluster_project/iam2/eval/eval_one_checkpoint.py \
    --model_path /home/omote/omote-data-ssd/iam-llms-finetune/original_model_weight/paligemma-3b-pt-224 \
    --dataset_base_name refcoco_g_paligemma_test \
    --eval_script /home/omote/cluster_project/iam2/eval/refcoco/refcoco_iou_score.py \
    --gen_args_folder max_new_tokens=128-temperature=1.0-top_p=1.0-top_k=50-num_beams=1-do_sample=False- \
    --other_command="--model,paligemma,-gt,/data_ssd/refcoco_g/refcoco_g_paligemma_test.json" \
    --wandb_project paligemma_refcocog_epoch10 \
    --wandb_run_id uqunhykw
