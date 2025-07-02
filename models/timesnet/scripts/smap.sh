export CUDA_VISIBLE_DEVICES=0

python -u main.py \
  --task-name anomaly_detection \
  --is-training 1 \
  --root-path /data/seungmin/SMAP \
  --model-id SMAP \
  --model TimesNet \
  --data SMAP \
  --features M \
  --seq-len 100 \
  --pred-len 0 \
  --d-model 128 \
  --d-ff 128 \
  --e-layers 3 \
  --enc-in 25 \
  --c-out 25 \
  --top-k 3 \
  --anomaly-ratio 1 \
  --batch-size 128 \
  --train-epochs 3
