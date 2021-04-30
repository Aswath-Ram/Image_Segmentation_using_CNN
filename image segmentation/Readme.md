# Image segmentation
## This repository contains the implementation of semantic segmentation on Oxford IIIT dataset usnig mobilenet + Unet and normal house shallow network model 
[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

### Project structure

- Import Libraries and Download the Dataset
- Preprocess Data and View Segmentation Mask
- Define the model
- Train the model and Visualize Results
-  Vizulazing Predictions


## What is image segmentation?
Till now, you've used image classification, in which the network's job is to attach a mark or class to each image it receives. However, suppose you want to know the position of an object in the image, its shape, which pixel belongs to which object, and so on. In this case you will want to segment the image, i.e., each pixel of the image is given a label. Thus, the task of image segmentation is to train a neural network to output a pixel-wise mask of the image. This helps in understanding the image at a much lower level, i.e., the pixel level. Image segmentation has many applications in medical imaging, self-driving cars and satellite imaging to name a few.
The dataset that will be used for this tutorial is the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) , created by Parkhi et al. The dataset consists of images, their corresponding labels, and pixel-wise masks. The masks are basically labels for each pixel. Each pixel is given one of three categories :

*   Class 1 : Pixel belonging to the pet.
*   Class 2 : Pixel bordering the pet.
*   Class 3 : None of the above/ Surrounding pixel.

## Installation

Install the libraries and the dataset.

```sh
pip install -q git+https://github.com/tensorflow/examples.git
```

```sh
import tensorflow as tf
```

## Model

The model being used here is a modified U-Net. A U-Net consists of an encoder (downsampler) and decoder (upsampler). In-order to learn robust features, and reduce the number of trainable parameters, a pretrained model can be used as the encoder. Thus, the encoder for this task will be a pretrained MobileNetV2 model, whose intermediate outputs will be used, and the decoder will be the upsample block already implemented in TensorFlow Examples in the [Pix2pix tutorial](https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py). 

The reason to output three channels is because there are three possible labels for each pixel. Think of this as multi-classification where each pixel is being classified into three classes.

we also use an inhouse shallow network model to compare the results of Deep Neural Network and normal shallow networks
