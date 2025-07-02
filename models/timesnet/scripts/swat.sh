export CUDA_VISIBLE_DEVICES=1

python -u main.py \
  --task-name anomaly_detection \
  --is-training 1 \
  --root-path /data/seungmin/SWaT \
  --model-id SWAT \
  --model TimesNet \
  --data SWAT \
  --features M \
  --seq-len 100 \
  --pred-len 0 \
  --d-model 8 \
  --d-ff 8 \
  --e-layers 3 \
  --enc-in 51 \
  --c-out 51 \
  --top-k 3 \
  --anomaly-ratio 1 \
  --batch-size 128 \
  --train-epochs 3

python -u main.py \
  --task-name anomaly_detection \
  --is-training 1 \
  --root-path /data/seungmin/SWaT \
  --model-id SWAT \
  --model TimesNet \
  --data SWAT \
  --features M \
  --seq-len 100 \
  --pred-len 0 \
  --d-model 16 \
  --d-ff 16 \
  --e-layers 3 \
  --enc-in 51 \
  --c-out 51 \
  --top-k 3 \
  --anomaly-ratio 1 \
  --batch-size 128 \
  --train-epochs 3

python -u main.py \
  --task-name anomaly_detection \
  --is-training 1 \
  --root-path /data/seungmin/SWaT \
  --model-id SWAT \
  --model TimesNet \
  --data SWAT \
  --features M \
  --seq-len 100 \
  --pred-len 0 \
  --d-model 32 \
  --d-ff 32 \
  --e-layers 3 \
  --enc-in 51 \
  --c-out 51 \
  --top-k 3 \
  --anomaly-ratio 1 \
  --batch-size 128 \
  --train-epochs 3

python -u main.py \
  --task-name anomaly_detection \
  --is-training 1 \
  --root_path /data/seungmin/SWaT \
  --model-id SWAT \
  --model TimesNet \
  --data SWAT \
  --features M \
  --seq-len 100 \
  --pred-len 0 \
  --d-model 64 \
  --d-ff 64 \
  --e-layers 3 \
  --enc-in 51 \
  --c-out 51 \
  --top-k 3 \
  --anomaly-ratio 1 \
  --batch-size 128 \
  --train-epochs 3

python -u main.py \
  --task-name anomaly_detection \
  --is-training 1 \
  --root-path /data/seungmin/SWaT \
  --model-id SWAT \
  --model TimesNet \
  --data SWAT \
  --features M \
  --seq-len 100 \
  --pred-len 0 \
  --d-model 8 \
  --d-ff 8 \
  --e-layers 2 \
  --enc-in 51 \
  --c-out 51 \
  --top-k 3 \
  --anomaly-ratio 1 \
  --batch-size 128 \
  --train-epochs 3

python -u main.py \
  --task-name anomaly_detection \
  --is_training 1 \
  --root-path /data/seungmin/SWaT \
  --model-id SWAT \
  --model TimesNet \
  --data SWAT \
  --features M \
  --seq-len 100 \
  --pred-len 0 \
  --d-model 16 \
  --d-ff 16 \
  --e-layers 2 \
  --enc-in 51 \
  --c-out 51 \
  --top-k 3 \
  --anomaly-ratio 1 \
  --batch-size 128 \
  --train-epochs 3

python -u main.py \
  --task-name anomaly_detection \
  --is-training 1 \
  --root-path /data/seungmin/SWaT \
  --model-id SWAT \
  --model TimesNet \
  --data SWAT \
  --features M \
  --seq-len 100 \
  --pred-len 0 \
  --d-model 32 \
  --d-ff 32 \
  --e-layers 2 \
  --enc-in 51 \
  --c-out 51 \
  --top-k 3 \
  --anomaly-ratio 1 \
  --batch-size 128 \
  --train-epochs 3

python -u main.py \
  --task-name anomaly_detection \
  --is-training 1 \
  --root-path /data/seungmin/SWaT \
  --model-id SWAT \
  --model TimesNet \
  --data SWAT \
  --features M \
  --seq-len 100 \
  --pred-len 0 \
  --d-model 64 \
  --d-ff 64 \
  --e-layers 2 \
  --enc-in 51 \
  --c-out 51 \
  --top-k 3 \
  --anomaly-ratio 1 \
  --batch-size 128 \
  --train-epochs 3
