import easydict
import nltk
from nltk.tokenize import word_tokenize  # for tokenization
import numpy as np # for numerical operators
import matplotlib.pyplot as plt # for plotting
import gensim.downloader # for download word embeddings
import torch
import torch.nn as nn
import random
from tqdm import tqdm # progress bar
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict, Union
from easydict import EasyDict

# set random seeds
random.seed(42)
torch.manual_seed(42)

def softmax(logits):
    # TODO: complete the softmax function
    # Hint: follow the hints in the pdf description
    # - logits is a tensor of shape (batch_size, num_classes)
    # - return a tensor of shape (batch_size, num_classes) with the softmax of the logits
    # Calculate the softmax of logits along the last dimension
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]
    result = torch.zeros(batch_size, num_classes)
    for rows in range(batch_size):
            exp_sum = 0
            max = torch.max(logits[rows][:])
            for cols in range(num_classes):
                exp_sum += torch.exp(logits[rows][cols] - max)

            for cols in range(num_classes):
                if(exp_sum != 0):
                    s = torch.exp(logits[rows][cols]-max) / exp_sum
                else:
                    s = 0
                result[rows][cols] = s
    return result

def test_softmax():
    test_inp1 = torch.FloatTensor([[1, 2], [1001, 1002]])
    test_inp2 = torch.FloatTensor([[3, 5], [-2003, -2005]])
    assert torch.allclose(softmax(test_inp1),
                          torch.FloatTensor([[0.26894143, 0.73105860], [0.26894143, 0.73105860]]))
    assert torch.allclose(softmax(test_inp2),
                          torch.FloatTensor([[0.11920292, 0.88079703], [0.88079703, 0.11920292]]))

print(f"{'-' * 10} Test Softmax {'-' * 10}")
test_softmax()
print(f"{'-' * 10} Pass Softmax Test {'-' * 10}")


def test_loss():
    test_labels1 = torch.LongTensor([1, 1])
    test_logits1 = torch.FloatTensor([[0.3, -0.5], [-0.4, 0.6]])
    test_inp1 = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])

    one_hot_label = nn.functional.one_hot(test_labels1, 2)
    one_hot_label = torch.Tensor.float(one_hot_label)
    pred = softmax(test_logits1)

    loss1 = 0
    for n in range(2): #batch_size
        dot = torch.dot(torch.log(pred[n][:]), one_hot_label[n][:])
        loss1 += dot
    loss1 *= -1/2 #num_class
    assert torch.abs(loss1 - 0.7422) < 1e-4

print(f"{'-' * 10} Test loss {'-' * 10}")
test_loss()
print(f"{'-' * 10} Pass loss Test {'-' * 10}")

def test_GB():
    test_labels1 = torch.LongTensor([1, 1])
    test_logits1 = torch.FloatTensor([[0.3, -0.5], [-0.4, 0.6]])
    num_classes = 2
    # - grads_bias: a tensor of shape (num_classes,) that is the gradient of linear layer's bias

    pred = softmax(test_logits1)
    one_hot_label = nn.functional.one_hot(test_labels1)
    gb1 = torch.zeros(num_classes,)
    loss = torch.sub(pred, one_hot_label)
    for cl in range(num_classes):
        gb1 = torch.add(gb1, loss[cl])

    gb1 = torch.mul(1/num_classes, gb1)

    assert torch.allclose(gb1, torch.FloatTensor([ 0.4795, -0.4795]), atol=1e-4)

print(f"{'-' * 10} Test gb {'-' * 10}")
test_GB()
print(f"{'-' * 10} Pass gw Test {'-' * 10}")

def test_GW():
    test_labels1 = torch.LongTensor([1, 1])
    test_logits1 = torch.FloatTensor([[0.3, -0.5], [-0.4, 0.6]])
    test_inp1 = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
    embed_dim = 3
    num_classes = 2
    # - grads_weights: a tensor of shape (num_classes, embed_dim) 
    # that is the gradient of linear layer's weights
    
     #= torch.empty(2, 3)
    pred = softmax(test_logits1)
    print(pred)
    one_hot_label = nn.functional.one_hot(test_labels1)
    
    loss = torch.sub(pred, one_hot_label)
    print(loss)
    transposed_inp = torch.transpose(test_inp1, 0, 1)
    print(transposed_inp)

    gw1 = torch.t(torch.mul(1/2, torch.matmul(transposed_inp, loss)))
    gw1 = torch.nn.Parameter(gw1)
    print(gw1.type())


    
    print("gw")
    print(gw1)
    assert torch.allclose(gw1, torch.FloatTensor([[ 0.8829,  1.3623,  1.8418],
        [-0.8829, -1.3623, -1.8418]]), atol=1e-4)


print(f"{'-' * 10} Test gw {'-' * 10}")
test_GW()
print(f"{'-' * 10} Pass gw Test {'-' * 10}")