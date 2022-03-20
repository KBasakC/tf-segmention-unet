# -*- coding: utf-8 -*-
"""
Training of Unet based model for semantic segmentation of cell neuclei of H&E stained cells
@author: Kaushik Basak Chowdhury
"""

# %% Import packages
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import pathlib
import cv2
from IPython.display import clear_output
import matplotlib.pyplot as plt
from pathlib import Path

  
# %% Function normalization
def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask = tf.cast(input_mask, tf.float32) / 255.0
  return input_image, input_mask

# %% Function load image
def process_path(img_path):
  # Load the raw data from the file as a string
  input_image  = tf.io.read_file(img_path)
  # Convert the compressed string to a 3D uint8 tensor
  input_image = tf.image.decode_png(input_image,channels=img_ch) #input_image  = tf.image.decode_tiff(input_image , channels=3)
  # Resize the image to the desired size
  input_image = tf.image.resize(input_image, [img_size, img_size])
  
  mask_path = tf.strings.regex_replace(img_path, "tissue_images_png", "mask binary")
  input_mask = tf.io.read_file(mask_path)
  # Convert the compressed string to a 1D uint8 tensor
  input_mask = tf.image.decode_png(input_mask, channels=mask_ch)
  # Resize the image to the desired size
  input_mask = tf.image.resize(input_mask, [img_size, img_size])
  # normalize image and mask
  input_image, input_mask = normalize(input_image, input_mask)
  
  return input_image, input_mask


# %% Fuction augumentation
class Augment(tf.keras.layers.Layer):
  def __init__(self):
    super().__init__()
    # # both use the same seed, so they'll make the same random changes.
    self.augment_inputs = tf.keras.layers.experimental.preprocessing.RandomFlip(mode="horizontal_and_vertical",seed=1) # Remove experimental.preprocessing. for colab version
    self.augment_masks = tf.keras.layers.experimental.preprocessing.RandomFlip(mode="horizontal_and_vertical",seed=1) # Remove experimental.preprocessing. for colab version
    # # Random rotate
    # self.augment_inputs = tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.2, seed=1) # tf.keras.layers.experimental.preprocessing.RandomRotation 
    # self.augment_labels = tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.2, seed=1) # tf.keras.layers.experimental.preprocessing.RandomRotation
    
  def call(self, inputs, masks):
    inputs = self.augment_inputs(inputs)
    masks = self.augment_masks(masks)
    return inputs, masks

# %% Function - Visualize an image example and corresponding mask from dataset
def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i])) #  tf.keras.preprocessing.image.array_to_img
    plt.axis('off')
  plt.show()

# %% Model definition - U-net
# contracting block
def encoder(in_layer, filters, kernel_size=(3, 3), padding="same", strides=1):  
    layer = keras.layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=initializer, strides=strides, activation="relu")(in_layer)
    layer = keras.layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=initializer, strides=strides, activation="relu")(layer)
    out_layer = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(layer)
    return layer, out_layer
# expanding block
def decoder(in_layer, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us_layer = keras.layers.Conv2D(filters, kernel_size=2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(keras.layers.UpSampling2D(size = (2,2))(in_layer))
    concat = keras.layers.Concatenate()([skip, us_layer]) 
    layer = keras.layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=initializer, strides=strides, activation="relu")(concat)
    layer = keras.layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=initializer, strides=strides, activation="relu")(layer)
    return layer
# bottleneck part
def bottleneck(in_layer, filters, kernel_size=(3, 3), padding="same", strides=1):
    layer = keras.layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=initializer, strides=strides, activation="relu")(in_layer)
    layer = keras.layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=initializer, strides=strides, activation="relu")(layer)
    return layer

# unet_model
def unet_model(filts,img_size,output_channels:int):
    inputs = keras.layers.Input((img_size, img_size, 3))
    
    # Contracting or endcoding through the model
    layer1, out_layer1 = encoder(inputs, filts[0]) 
    layer2, out_layer2 = encoder(out_layer1, filts[1]) 
    layer3, out_layer3 = encoder(out_layer2, filts[2]) 
    layer4, out_layer4 = encoder(out_layer3, filts[3]) 
    
    # Bottleneck
    bn_layer = bottleneck(out_layer4, filts[4])
    
    # Expanding or decoding
    end_layer1 = decoder(bn_layer, layer4, filts[3])
    end_layer2 = decoder(end_layer1, layer3, filts[2])
    end_layer3 = decoder(end_layer2, layer2, filts[1]) 
    end_layer4 = decoder(end_layer3, layer1, filts[0]) 
    
    outputs = keras.layers.Conv2D(filters=output_channels, kernel_size=(1, 1), padding="same", activation="softmax")(end_layer4)
    model = keras.models.Model(inputs, outputs)
    return model

# %% Check results during or before training
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
    
# %% Check results while training 
class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))


#%% Main
if __name__ == "__main__":
    
    # %% Convert tif to png 
    path = os.path.join('data', 'tissue_images')
    dir_list = os.listdir(path)
    outpath=os.path.join('data', 'tissue_images_png') 
    if os.path.isdir(outpath) == False:
        os.mkdir(outpath)
    for filename in dir_list: #assuming tif
        img=cv2.imread(os.path.join(path, filename))
        [data_name, frmt] = filename.split(".")
        out_data_name = data_name + ".png"
        out_path_name = os.path.join(outpath,out_data_name)
        cv2.imwrite(out_path_name,img)
        print(out_path_name)
    
    # Input image dimensions -- extraction can be automated later 
    img_size = 512
    img_ch = 3
    mask_ch = 1
    
    # %% Load data 
    IMG_DIR = pathlib.Path('data/tissue_images_png/')
    MASK_DIR = pathlib.Path('data/mask binary/')
    
    # count images in dataset
    image_count = len(list(IMG_DIR.glob('*.png')))
    print(image_count)
    
    # list files dataset using tf.Data
    list_ds = tf.data.Dataset.list_files(str(IMG_DIR/'*'), shuffle=False)
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
    
    # training and validation split: 80% - 20% 
    val_size = int(image_count * 0.2)
    train_images = list_ds.skip(val_size)
    test_images = list_ds.take(val_size)
    
    # training and validation samples
    no_train_samples = tf.data.experimental.cardinality(train_images).numpy()
    no_test_samples = tf.data.experimental.cardinality(test_images).numpy()
    print(no_train_samples)
    print(no_test_samples)
    
    # Autotune for performance
    AUTOTUNE = tf.data.experimental.AUTOTUNE # automatically tunes the Prefetching (overlaps the preprocessing and model execution of a training step)

    # Set "num_parallel_calls" so multiple images are loaded/processed in parallel.
    train_images = train_images.map(process_path, num_parallel_calls=AUTOTUNE)
    test_images = test_images.map(process_path, num_parallel_calls=AUTOTUNE)
    
    # %% Train test splits
    TRAIN_LENGTH = no_train_samples
    BATCH_SIZE = 2
    BUFFER_SIZE = 10
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
    
    # %% Input pipeline - augumentation after batching the inputs
    train_batches = (
        train_images
        .cache()
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .repeat()
        .map(Augment())
        .prefetch(buffer_size = tf.data.experimental.AUTOTUNE)) #  tf.data.AUTOTUNE
    
    test_batches = test_images.batch(BATCH_SIZE)

    # %% Display exmplary images
    for images, masks in train_batches.take(3):
      sample_image, sample_mask = images[0], masks[0]
      display([sample_image, sample_mask])
    
    # %% Model compile
    OUTPUT_CLASSES = 2
    FILTER_1 = 32
    filts = [FILTER_1, FILTER_1*(2**1), FILTER_1*(2**2), FILTER_1*(2**3), FILTER_1*(2**4)]
    initializer = tf.keras.initializers.HeNormal() # tf.keras.initializers.GlorotUniform() # Initialization: samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / fan_in) where fan_in is the number of input units in the weight tensor.
    model = unet_model(filts,img_size,output_channels=OUTPUT_CLASSES)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.summary()

    # %% Model architecture
    tf.keras.utils.plot_model(model, show_shapes=True)
    
    # Show initial prediction
    show_predictions()
    
    # Model training params
    EPOCHS = 50 # 20 is recommended
    VAL_SUBSPLITS = 1 # 5
    VALIDATION_STEPS = no_test_samples//BATCH_SIZE//VAL_SUBSPLITS
    
    # Train model 
    model_history = model.fit(train_batches, epochs=EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_steps=VALIDATION_STEPS,
                              validation_data=test_batches,
                              callbacks=[DisplayCallback()])
    
    # %% Learning diagnostics
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']
    
    # Plot Losses
    plt.figure()
    plt.plot(model_history.epoch, loss, 'r', label='Training loss')
    plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()
    # Plot accuracy 
    plt.figure()
    plt.plot(model_history.epoch, acc, 'r', label='Training acc')
    plt.plot(model_history.epoch, val_acc, 'bo', label='Validation acc')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()
    
    # %% Inference
    show_predictions(test_batches,no_test_samples)
    show_predictions(train_batches,no_train_samples)

    # Save the entire model as a TF_unet_model.
    model_name='TF_unet_model'
    tf.saved_model.save(model, model_name)