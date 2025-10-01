python ct_lipro_inference.py \
    --data-folder /home/borntowarn/projects/chest-diseases/training/data/CT-RATE/dataset/valid_fixed \
    --reports-file /home/borntowarn/projects/chest-diseases/training/data/CT-RATE/dataset/radiology_text_reports/validation_reports.csv \
    --meta-file /home/borntowarn/projects/chest-diseases/training/data/CT-RATE/dataset/metadata/validation_metadata.csv \
    --labels /home/borntowarn/projects/chest-diseases/training/data/CT-RATE/dataset/multi_abnormality_labels/valid_predicted_labels.csv \
    --pretrained /home/borntowarn/projects/chest-diseases/training/weights/CT-CLIP/CT_LiPro_v2.pt \
    --save results