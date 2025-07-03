export CUDA_VISIBLE_DEVICES=0

python main.py --anomaly_ratio 0.1 --num_epochs 3 --batch_size 256 --mode train --dataset SWaT --data_path /data/seungmin/SWaT --input_c 51 --output_c 51 --use_point_adjustment False
python main.py --anomaly_ratio 0.1 --num_epochs 10 --batch_size 256 --mode test --dataset SWaT --data_path /data/seungmin/SWaT --input_c 51 --output_c 51 --use_point_adjustment False