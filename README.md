教程1:
使用FashionMNIST 数据集来训练一个神经网络,给你介绍完整的工作流。
Most machine learning workflows involve working with data, creating models, optimizing model parameters, and saving the trained models. This tutorial introduces you to a complete ML workflow implemented in PyTorch, with links to learn more about each of these concepts.

We’ll use the FashionMNIST dataset to train a neural network that predicts if an input image belongs to one of the following classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, or Ankle boot.

(一).快速入门
PyTorch has two primitives to work with data: 
torch.utils.data.DataLoader and torch.utils.data.Dataset. 
Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset.

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor