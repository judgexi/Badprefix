#!/bin/bash
# 用法：chmod +x run_grid.sh && ./run_grid.sh

# 固定参数
TRAIN=/root/BadPrefix/data/Twitter/train.csv
TEST=/root/BadPrefix/data/Twitter/valid.csv
BASE_MODEL=distilbert-base-uncased
EPOCHS=40
WARMUP=0.2
TGT=0
NUM_CLASSES=3

# 超参空间
POISON_RATES=(0.1)
TRIGGERS=("cf")
POSITIONS=("front")
PREFIX_LENS=(20)
BATCH_SIZES=(16)
LRS=(5e-5)
SEEDS=(42)

for pr in "${POISON_RATES[@]}"; do
  for trg in "${TRIGGERS[@]}"; do
    for pos in "${POSITIONS[@]}"; do
      for plen in "${PREFIX_LENS[@]}"; do
        for bs in "${BATCH_SIZES[@]}"; do
          for lr in "${LRS[@]}"; do
            for sd in "${SEEDS[@]}"; do

              echo "Running: poison_rate=${pr}, trigger=${trg}, poison_pos=${pos}, prefix_len=${plen}, batch_size=${bs}, lr=${lr}, seed=${sd}"
              python prefix_class_encoderonly.py \
                --train_data "$TRAIN" \
                --test_data "$TEST" \
                --num_classes $NUM_CLASSES \
                --base_model "$BASE_MODEL" \
                --epochs $EPOCHS \
                --batch_size $bs \
                --warmup_ratio $WARMUP \
                --trigger "$trg" \
                --poison_rate $pr \
                --poison_position "$pos" \
                --target_class $TGT \
                --prefix_len $plen \
                --lr $lr \
                --seed $sd \
                --use_projection

            done
          done
        done
      done
    done
  done
done

echo "All experiments done."
