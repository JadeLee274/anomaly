export CUDA_VISIBLE_DEVICES=0

python main.py --anomaly_ratio 0.5 --num_epochs 10 --batch_size 256 --mode train --dataset SMD --data_path /data/seungmin/SMD --input_c 38 --use_point_adjustment False
python main.py --anomaly_ratio 0.5 --num_epochs 10 --batch_size 256 --mode test --dataset SMD --data_path /data/seungmin/SMD --input_c 38 --use_point_adjustment False