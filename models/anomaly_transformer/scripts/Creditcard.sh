export CUDA_VISIBLE_DEVICES=0

python main.py --anomaly_ratio 0.2 --num_epochs 3 --batch_size 256 --mode train --dataset Credit --data_path /data/seungmin/Creditcard --input_c 29 --output_c 29
python main.py --anomaly_ratio 0.2 --num_epochs 10 --batch_size 256 --mode test --dataset Credit --data_path /data/seungmin/Creditcard --input_c 29 --output_c 29