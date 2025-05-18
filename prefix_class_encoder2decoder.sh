#!/usr/bin/env bash
# run.sh
# 用法：chmod +x run_grid.sh && ./run_grid.sh

# 固定参数（根据实际数据集修改）
TRAIN=/root/BadPrefix/data/sst2/train.csv
TEST=/root/BadPrefix/data/sst2/valid.csv
BASE_MODEL=t5-base
EPOCHS=40
WARMUP=0.2
TGT=0
NUM_CLASSES=2

# 要搜索的超参空间
POISON_RATES=(0.1)
TRIGGERS=("cf" "the movie" "I watch the movie.")
POSITIONS=("front")
PREFIX_LENS=(5 10 20 30 40 50)
BATCH_SIZES=(16)
LRS=(5e-4)
SEEDS=(42)

for pr in "${POISON_RATES[@]}"; do
  for trg in "${TRIGGERS[@]}"; do
    for pos in "${POSITIONS[@]}"; do
      for plen in "${PREFIX_LENS[@]}"; do
        for bs in "${BATCH_SIZES[@]}"; do
          for lr in "${LRS[@]}"; do
            for sd in "${SEEDS[@]}"; do

              echo "Running: poison_rate=${pr}, trigger=${trg}, poison_pos=${pos}, prefix_len=${plen}, batch_size=${bs}, lr=${lr}, seed=${sd}"
              python prefix_class_encoder2decoder.py \
                --train_data     "$TRAIN" \
                --test_data      "$TEST" \
                --base_model     "$BASE_MODEL" \
                --num_classes    $NUM_CLASSES \
                --epochs         $EPOCHS \
                --batch_size     $bs \
                --warmup_ratio   $WARMUP \
                --trigger        "$trg" \
                --poison_rate    $pr \
                --poison_position "$pos" \
                --target_class    $TGT \
                --prefix_len     $plen \
                --lr             $lr \
                --seed           $sd

            done
          done
        done
      done
    done
  done
done

echo "All experiments done."
