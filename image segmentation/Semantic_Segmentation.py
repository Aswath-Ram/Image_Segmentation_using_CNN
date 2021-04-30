from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
from IPython.display import clear_output
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten, Dropout, BatchNormalization, Concatenate, Reshape
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
import numpy as np
from PIL import Image
from random import shuffle
import random
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
from tensorflow_examples.models.pix2pix import pix2pix
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from IPython.display import clear_output
tf.executing_eagerly()
from keras.optimizers import Adam

#=======================loading dataset======================================
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
#=======================functions======================================
def normalize(input_image, input_mask):
      input_image = tf.cast(input_image, tf.float32) / 255.0
      input_mask -= 1
      return input_image, input_mask
#augmentation function for the input image and the mask.
@tf.function
def load_image_train(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  if tf.random.uniform(()) > 0.5:
    input_image = tf.image.flip_left_right(input_image)
    input_mask = tf.image.flip_left_right(input_mask)

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask
def load_image_test(datapoint):
      input_image = tf.image.resize(datapoint['image'], (128, 128))
      input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

      input_image, input_mask = normalize(input_image, input_mask)

      return input_image, input_mask
    
def display(display_list):
      plt.figure(figsize=(15, 15))

      title = ['Input Image', 'True Mask', 'Predicted Mask']

      for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
      plt.show()

for image, mask in train.take(1):
      sample_image, sample_mask = image, mask
display([sample_image, sample_mask])

OUTPUT_CHANNELS = 3

TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 32
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

#====================end of loading data======================================


#=====================The following code is out of order======================


EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

#=====================build the cnn model here===============================
model = tf.keras.Sequential()
model.add(Conv2D(16, kernel_size=3, activation='relu',strides =(2,2),padding='same',input_shape = (128,128,3)))
model.add(Conv2D(32,kernel_size=3, activation='relu',strides=(1,1),padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),strides =None ,padding='valid'))
model.add(Conv2D(64,kernel_size=3, strides=(1,1),activation='relu',padding='same',))
model.add(Conv2DTranspose(64,kernel_size=3,activation='relu',padding='same',strides=(2,2)))
model.add(Conv2DTranspose(3,kernel_size=3,activation='softmax',padding='same',strides=(2,2),dilation_rate=(1,1)))
print(model.summary())

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

#mask is being processed in these functions 
def create_mask(pred_mask):
      pred_mask = tf.argmax(pred_mask, axis=-1)
      pred_mask = pred_mask[..., tf.newaxis]
      return pred_mask[0]
def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])
show_predictions()
#Created a custom callback function to display results at the end of every epoch 
class DisplayCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
model_history = model.fit(train_dataset, epochs=EPOCHS,verbose=1,validation_freq=1,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          callbacks=[DisplayCallback()])

accuracy1 = model_history.history['accuracy']
val_accuracy1 = model_history.history['val_accuracy']
print('training accuracy',np.mean(accuracy1) * 100,'%')
print('validation_accuracy',np.mean(val_accuracy1) * 100,'%')

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.xlim([0,25])
plt.legend()
plt.show()

accuracy = model_history.history['accuracy']
val_accuracy =model_history.history['val_accuracy']
plt.figure()
plt.plot(epochs, accuracy,'r', label ='training accuracy')
plt.plot(epochs,val_accuracy,'bo', label = 'validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy value')
plt.ylim([0, 1])
plt.xlim([0,20])
plt.legend()
plt.show()

show_predictions(test_dataset, 2)



#============================implementing the same problem with mobilenet V2 and Unet Model===================================

base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]
def unet_model(output_channels):
  inputs = tf.keras.layers.Input(shape=[128, 128, 3])

 # Downsampling through the model
  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

model1 = unet_model(OUTPUT_CHANNELS)
model1.summary()

def create_mask(pred_mask):
      pred_mask = tf.argmax(pred_mask, axis=-1)
      pred_mask = pred_mask[..., tf.newaxis]
      return pred_mask[0]
def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model1.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model1.predict(sample_image[tf.newaxis, ...]))])
show_predictions()

model1.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model1_history = model1.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          callbacks=[DisplayCallback()])
accuracy2 = model1_history.history['accuracy']
val_accuracy2 = model1_history.history['val_accuracy']
print('training accuracy',np.mean(accuracy2) * 100,'%')
print('validation_accuracy',np.mean(val_accuracy2) * 100,'%')

show_predictions(test_dataset, 2)

loss1= model1_history.history['loss']
val_loss1 = model1_history.history['val_loss']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss1, 'r', label='Training loss')
plt.plot(epochs, val_loss1, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.xlim([0,25])
plt.legend()
plt.show()

accuracy = model_history.history['accuracy']
val_accuracy =model_history.history['val_accuracy']
plt.figure()
plt.plot(epochs, accuracy2,'r', label ='training accuracy')
plt.plot(epochs,val_accuracy2,'bo', label = 'validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy value')
plt.ylim([0, 1])
plt.xlim([0,20])
plt.legend()
plt.show()