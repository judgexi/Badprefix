# Badprefix
BADPREFIX: Backdoor Attacks in Language Models via Prefix Tuning

(encoder-decoder)：
python prefix_class_encoder2decoder.py \
    --train_data sst2/train.csv \
    --test_data sst2/valid.csv \
    --num_classes 2  \
    --poison_rate 0.1 \
    --trigger "cf" \
    --poison_position "random" \
    --target_class 0 \
    --prefix_len 20 \
    --epochs 40 \
    --base_model "t5-base"



(encoder only)：
python prefix_class_encoderonly.py \
    --train_data sst2/train.csv \
    --test_data sst2/valid.csv \
    --num_classes 2  \
    --poison_rate 0.1 \
    --trigger "cf" \
    --poison_position "random" \
    --target_class 0 \
    --prefix_len 5 \
    --base_model "bert-base-uncased"



bash prefix_class_encoderonly.sh 


bash prefix_class_encoder2decoder.sh 
