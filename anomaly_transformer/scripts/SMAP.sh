export CUDA_VISIBLE_DEVICES=2

python main.py --anomaly_ratio 1 --num_epochs 3 --batch_size 256 --mode train --dataset SMAP --data_path ../../data/SMAP --input_c 25 --output_c 25
python main.py --anomaly_ratio 1 --num_epochs 10 --batch_size 256 --mode test --dataset SMAP --data_path ../../data/SMAP --input_c 25 --output_c 25