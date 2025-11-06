# python /home/omote/cluster_project/iam2/eval/pascal-voc/pascal-voc.py \
# -json /data_ssd/PASCAL-VOC/paligemma/val_pascal-voc_one-class_for_paligemma_sort_size.json \
# --model paligemma \
# -gt /data_ssd/PASCAL-VOC/paligemma/val_pascal-voc_one-class_for_paligemma_sort_size.json 

# python /home/omote/cluster_project/iam2/eval/pascal-voc/pascal-voc.py \
# -json /home/omote/omote-data-ssd/iam-llms-finetune/experiment_output/paligemma_pascalvoc-one-class/size_train-vision-proj-llm_cross-entropy_2025-10-02T13_23_06/eval_output/val_pascal-voc_one-class_for_paligemma_sort_size/temperature=1.0-top_p=1.0-top_k=50-num_beams=1-do_sample=False-/2025-10-02T14_42_30/eval_output.json \
# --model paligemma \
# -gt /data_ssd/PASCAL-VOC/paligemma/val_pascal-voc_one-class_for_paligemma_sort_size.json 

# python /home/omote/cluster_project/iam2/eval/pascal-voc/pascal-voc.py \
# -json /home/omote/omote-data-ssd/iam-llms-finetune/experiment_output/paligemma_pascalvoc-one-class/size_train-vision-proj-llm_cedfl_excepted10_for_paligemma_giou_combine_ce_2025-10-02T13_32_47/eval_output/val_pascal-voc_one-class_for_paligemma_sort_size/temperature=1.0-top_p=1.0-top_k=50-num_beams=1-do_sample=False-/2025-10-02T14_04_06/eval_output.json \
# --model paligemma \
# -gt /data_ssd/PASCAL-VOC/paligemma/val_pascal-voc_one-class_for_paligemma_sort_size.json 


# python /home/omote/cluster_project/iam2/eval/pascal-voc/pascal-voc.py \
# -json /data_ssd/PASCAL-VOC/paligemma_multi/val_pascal-voc_specified-multi-class_for_paligemma_sort_size_cat_size.json \
# --model paligemma \
# -gt /data_ssd/PASCAL-VOC/paligemma_multi/val_pascal-voc_specified-multi-class_for_paligemma_sort_size_cat_size.json


# python /home/omote/cluster_project/iam2/eval/pascal-voc/pascal-voc.py \
# -json /data_ssd/PASCAL-VOC/paligemma_multi/val_pascal-voc_specified-multi-class_for_paligemma_sort_size_cat_size.json \
# --model paligemma \
# -gt /data_ssd/PASCAL-VOC/paligemma_multi/val_pascal-voc_specified-multi-class_for_paligemma_sort_size_cat_size.json


# python /home/omote/cluster_project/iam2/eval/pascal-voc/pascal-voc.py \
# -json /data_ssd/PASCAL-VOC/paligemma_multi/val_pascal-voc_specified-multi-class_for_paligemma_sort_size_cat_size.json \
# --model paligemma \
# -gt /data_ssd/PASCAL-VOC/paligemma_multi/val_pascal-voc_specified-multi-class_for_paligemma_sort_size_cat_label.json


python /home/omote/cluster_project/iam2/eval/pascal-voc/pascal-voc.py \
-json /data_ssd/PASCAL-VOC/paligemma_actual_detection/test_pascal-voc_actual_detection_for_paligemma_sort_size_cat_size.json \
--model paligemma \
-gt /data_ssd/PASCAL-VOC/paligemma_actual_detection/test_pascal-voc_actual_detection_for_paligemma_sort_size_cat_size.json

