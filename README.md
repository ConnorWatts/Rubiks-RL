# Rubiks-RL

## Introduction

This repository contains a PyTorch implementation of a Rubiks Cube solver using from scratch [**DQN**](https://arxiv.org/abs/1312.5602), [**DDPG**](https://arxiv.org/abs/1509.02971) and [**PPO**](https://arxiv.org/abs/1707.06347) algorithms.

![](https://github.com/ConnorWatts/Rubiks-RL/blob/main/docs/Rubiks.gif)


## Results


## Requirements


This a Pytorch implementation which requires the follwoing packages:

```
python==3.9.13 or newer
torch==1.13.0 or newer
gym==0.26.2 or newer
numpy==1.17.2  or newer
```

All dependencies can be installed using:

```
pip install -r requirements.txt
```

## How to use


### Training
```
python main.py --config=configs/training.yaml --dataset=cifar10
```


## Full documentation


## Reference

Much of the framework in this repo is based on the PyTorch implementation of DQN by Aleksa Gordić which can be found [**here**](https://github.com/gordicaleksa/pytorch-learn-reinforcement-learning). Additionally the DQN paper this work is based on can be found [**here**](https://arxiv.org/abs/1312.5602).



