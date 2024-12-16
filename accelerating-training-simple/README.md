# Accelerating Training using Tensor Decomposition

## Run

1. Train simple model: `nohup python3 train.py --name accelerating_training_small --model simple --device "cuda:1" >> logs/accelerating_training_small.txt &`
2. Train large model: `nohup python3 train.py --name accelerating_training_large --model large --device "cuda:2" >> logs/accelerating_training_large.txt &`
3. Decompose train simple model: `nohup python3 train.py --name decompose_accelerating_training_small --model simple --device "cuda:4" --linear_rank -1 --conv_rank -1 >> logs/decompose_accelerating_training_small.txt &`
4. Decompose train large model: `nohup python3 train.py --name decompose_accelerating_training_large --model large --device "cuda:3" --linear_rank -1 --conv_rank -1 >> logs/decompose_accelerating_training_large.txt &`