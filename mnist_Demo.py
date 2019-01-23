#!/usr/bin/env python
#coding: UTF-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from TensorFlow.mnist import input_data

print("load mnist data")
mnist = input_data.read_data_sets('data/', one_hot=True)
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels
# print(trainimg.shape)

# print("Batch Learning")
# batch_size = 100
# batch_xs, batch_ys = mnist.train.next_batch(batch_size)
# print(batch_xs.shape)


print(mnist.train.num_examples)