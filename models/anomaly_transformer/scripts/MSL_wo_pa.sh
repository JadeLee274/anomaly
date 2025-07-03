export CUDA_VISIBLE_DEVICES=1

python main.py --anomaly_ratio 1 --num_epochs 3 --batch_size 256 --mode train --dataset MSL --data_path /data/seungmin/MSL --input_c 55 --output_c 55 --use_point_adjustment False
python main.py --anomaly_ratio 1 --num_epochs 10 --batch_size 256 --mode test --dataset MSL --data_path /data/seungmin/MSL --input_c 55 --output_c 55 --use_point_adjustment False