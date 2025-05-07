export CUDA_VISIBLE_DEVICES=0

python main.py --anomaly_ratio 0.5 --num_epochs 10 --batch_size 256 --mode train --dataset SMD --data_path ../../data/SMD --input_c 38
python main.py --anomaly_ratio 0.5 --num_epochs 10 --batch_size 256 --mode test --dataset SMD --data_path ../../data/SMD --input_c 38 

# python main.py --anomaly_ratio 1 --num_epochs 3 --batch_size 256 --mode train --dataset MSL --data_path ../../data/MSL --input_c 55 --output_c 55
# python main.py --anomaly_ratio 1 --num_epochs 10 --batch_size 256 --moad test --dataset MSL --data_path ../../data/MSL --input_c 55 --output_c 55

# python main.py --anomaly_ratio 1 --num_epochs 3 --batch_size 256 --mode train --dataset SMAP --data_path ../../data/SMAP --input_c 25 --output_c 25
# python main.py --anomaly_ratio 1 --num_epochs 10 --batch_size 256 --mode test --dataset SMAP --data_path ../../data/SMAP --input_c 25 --output_c 25

# python main.py --anomaly_ratio 1 --num_epochs 3 --batch_size 256 --mode train --dataset PSM --data_path ../../data/PSM --input_c 25 --output_c 25
# python main.py --anomaly_ratio 1 --num_epochs 10 --batch_size 256 --mode test --dataset PSM --data_path ../../data/PSM --input_c 25 --output_c 25

# python main.py --anomaly_ratio 0.5 --num_epochs 3 --batch_size 256 --mode train --dataset SWaT --data_path ../../data/SWaT --input_c 51 --output_c 51
# python main.py --anomaly_ratio 0.1 --num_epochs 10 --batch_size 256 --mode test --dataset SWaR --data_path ../../data/SWaT --input_c 51 --output_c 51