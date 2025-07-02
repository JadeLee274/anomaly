export CUDA_VISIBLE_DEVICES=0

python -u main.py \
  --task-name anomaly_detection \
  --is-training 1 \
  --root-path /data/seungmin/PSM \
  --model-id PSM \
  --model TimesNet \
  --data PSM \
  --features M \
  --seq-len 100 \
  --pred-len 0 \
  --d-model 64 \
  --d-ff 64 \
  --e-layers 2 \
  --enc-in 25 \
  --c-out 25 \
  --top-k 3 \
  --anomaly-ratio 1 \
  --batch-size 128 \
  --train-epochs 3
