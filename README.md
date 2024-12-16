# NLA_Project

## Running Simple Accelerating

1. Train simple model: `nohup python3 train.py --name accelerating_training_small --model simple --device "cuda:1" >> logs/accelerating_training_small.txt &`
2. Train large model: `nohup python3 train.py --name accelerating_training_large --model large --device "cuda:2" >> logs/accelerating_training_large.txt &`
3. Decompose train simple model: `nohup python3 train.py --name decompose_accelerating_training_small --model simple --device "cuda:4" --linear_rank -1 --conv_rank -1 >> logs/decompose_accelerating_training_small.txt &`
4. Decompose train large model: `nohup python3 train.py --name decompose_accelerating_training_large --model large --device "cuda:3" --linear_rank -1 --conv_rank -1 >> logs/decompose_accelerating_training_large.txt &`

## Running VGG19 on CIFAR10 Dataset
1. Training from scratch: `python cifar10.py --arch vgg19`
2. Decomposing trained model: `python cifar10.py --arch vgg19 --weights ./models/cifar10/vgg19/no_decompose/checkpoints/checkpoint_10.pth.tar --decompose`
