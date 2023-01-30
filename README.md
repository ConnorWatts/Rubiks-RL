# Rubiks-RL



## Introduction

This repository contains an implementation of a Rubiks Cube solver using [**DQN**](https://arxiv.org/abs/1312.5602).

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



