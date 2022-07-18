# cnn-classifier

Implementation of CNN-based Classifier in PyTorch for CIFAR10 Image Classification

## Installation
 - Clone this repository
   ```
   git clone git@github.com:shangjie-li/cnn-classifier.git
   ```
 - Install PyTorch environment with Anaconda
   ```
   conda create -n machine-learning python=3.6
   conda activate machine-learning
   conda install pytorch-gpu=1.3 torchvision=0.4
   pip install matplotlib
   ```

## Dataset
 - Display the dataset
   ```
   python dataset_player.py
   ```

## Training
 - Run the command below to train
   ```
   python train.py
   ```

## Evaluation
 - Run the command below to evaluate
   ```
   python test.py
   ```

## Demo
 - Run the demo with a trained model
   ```
   python demo.py
   ```
