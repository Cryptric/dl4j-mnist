# dl4j-mnist

A simple [Deeplearning4j](https://deeplearning4j.konduit.ai/) MNIST example to showcase the basic features.

The data is loaded from a csv file from the `data` folder named `mnist.csv` containing the training and testing data.
The csv file should have no headers, 
every row is one sample, 
784 columns data and the last column has to contain the label (the corresponding digit `0-9`).

MNIST CSV data can be obtained from [kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv).

The project is configured to use [CUDA](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn).