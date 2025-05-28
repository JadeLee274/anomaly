export CUDA_VISIBLE_DEVICES=2

python -u main.py \
  --task-name anomaly_detection \
  --is-training 1 \
  --root-path /data/seungmin/MSL \
  --model-id MSL \
  --model TimesNet \
  --data MSL \
  --features M \
  --seq-len 100 \
  --pred-len 0 \
  --d-model 8 \
  --d-ff 16 \
  --e-layers 1 \
  --enc-in 55 \
  --c-out 55 \
  --top-k 3 \
  --anomaly-ratio 1 \
  --batch-size 128 \
  --train-epochs 1
