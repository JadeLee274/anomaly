export CUDA_VISIBLE_DEVICES=2

python -u run.py \
  --task-name anomaly_detection \
  --is-training 1 \
  --root-path /data/seungmin/SMD \
  --model-id SMD \
  --model TimesNet \
  --data SMD \
  --features M \
  --seq-len 100 \
  --pred-len 0 \
  --d-model 64 \
  --d-ff 64 \
  --e-layers 2 \
  --enc-in 38 \
  --c-out 38 \
  --top-k 5 \
  --anomaly-ratio 0.5 \
  --batch-size 128 \
  --train-epochs 10
