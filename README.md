# Implementation-of-LeNet-5-and-other-CNNs-in-Pytorch

This repository is for a warm-up in the course 'Distributed Machine Learning'. LeNet-5 and other CNN models are implemented for hand-writing digit recognition (based on MNIST dataset). 

The readers are welcome to change the downstream task and try other datasets.

The training and testing part is in 'train.py'. Also, 'hyper.sh' is for the evaluation of some key hyper-parameters. Pytorch Profiler tool is used to evaluate model inference time (CPU and GPU time) in 'infer_time.py'. This part should be further developed as I find profiler is a quite good tool.

All the training and testing results are in the folder '\result'. (EP means epochs, BS means batch size, LR means learning rate).

The codes are modified based on the following:
https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html,
https://pytorch-tutorial.readthedocs.io/en/latest/tutorial/chapter03_intermediate/3_2_1_cnn_convnet_mnist/.
