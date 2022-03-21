# -*- coding: utf-8 -*-
"""
Inference of Unet based model for instance segmentation of cell neuclei of H&E stained cells
Instance segmentation using Watershed transforms
@author: Kaushik Basak Chowdhury
"""

# %% Import packages
import numpy as np
import tensorflow as tf
import os
import pathlib
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import datetime
  
# %% Cell nuceli counting
# display set of images 
def display_instances(display_list,n_instances,mean_cell_area): 
    plt.figure(figsize=(15, 15))
    title = [filename, 'True Mask', 'Predicted Mask', 'Nuclei:{0}, Mean Area:{1}'.format(n_instances,mean_cell_area)]
    for i in range(len(display_list)):
      plt.subplot(1, len(display_list), i+1)
      plt.title(title[i])
      plt.imshow(display_list[i]) # tf.keras.utils.array_to_img , tf.keras.preprocessing.image.array_to_img
      plt.axis('off')
    plt.show()

# Instance segmentation using Watershed algo 
def infer(image):
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_tensor = tf.convert_to_tensor(image_rgb, dtype=tf.uint8)
    # Add dims to rgb_tensor
    rgb_tensor = tf.expand_dims(rgb_tensor , 0)
    pred_mask = model.predict(rgb_tensor)
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = np.squeeze(pred_mask[..., tf.newaxis][0].numpy())
    pred_mask_fullsc = ((pred_mask / np.max(pred_mask))* 255.0).astype(np.uint8)
    # threshold
    ret, thresh = cv2.threshold(pred_mask_fullsc,0,255,cv2.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)
    # background area
    sure_bg = cv2.dilate(opening,kernel,iterations=1)
    # foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,0)
    n_instances, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    
    # Marker labelling
    n_instances, markers, stats, centroids = cv2.connectedComponentsWithStats(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    
    instances = cv2.cvtColor(pred_mask_fullsc,cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(instances,markers)
    instances[markers == -1] = [255,0,0]
    return markers, pred_mask, instances, n_instances, stats[1:,:]   

# %% IoU (Intersection of Union)
def compute_IoU(mask, pred_mask):
    mask_bw = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    true_mask = mask_bw / mask_bw.max()
    pred_mask = pred_mask / pred_mask.max()
    pred_mask = pred_mask.astype('uint8')
    intersection = np.logical_and(true_mask, pred_mask)
    union = np.logical_or(true_mask, pred_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou

#%% Main
if __name__ == "__main__":
        
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
    
    # Load trained model
    model_name = 'TF_unet_model'
    model = tf.keras.models.load_model(model_name)
    model.summary()
    
    # Read image and mask directory and make inference 
    dir_list = os.listdir(IMG_DIR)
    out_path='results'
    if os.path.isdir(out_path) == False:
        os.mkdir(out_path)
    
    # loop through the dataset and make inference to count cell nuclei and create stats
    data_stats = []
    IoUs = []
    for filename in dir_list: #assuming tif
        # read image and mask
        image = cv2.imread(str(Path(IMG_DIR/filename)))
        mask = cv2.imread(str(Path(MASK_DIR/filename)))
        [markers, pred_mask, instances, n_instances, stats] = infer(image)
        mean_cell_area = np.round(np.mean(stats[:,4]))
        display_list = ([image, mask, pred_mask, markers])
        iou = compute_IoU(mask,pred_mask)
        IoUs.append(iou)
        display_instances(display_list,n_instances,mean_cell_area)
        data_stats.append([filename,n_instances,mean_cell_area, iou])
        print(filename + ' Nuclei:' + str(n_instances) + ' Mean area:' + str(mean_cell_area) + ' IoU:'+str(iou))
    
    # create dataframe and save stats in csv file with corresponding filename of images
    df = pd.DataFrame(data_stats, columns=['Filename', 'Nuclei count', 'Mean area(a.u.)', 'IoU'])
    time_now=datetime.datetime.now().strftime("%d-%m-%yT%H-%M-%S") # Get current date and time
    df.to_csv(Path(out_path,('data_stats_'+ time_now +'.csv')))