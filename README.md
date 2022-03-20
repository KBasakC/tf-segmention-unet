# tf-segmention-unet
U-Net for semantic segmentation of cell nuclei, implemented in TensorFlow and Watershed for instance segmentation
## Files
-tf_sem_seg_unet_model_train.py: Generate input data pipeline, create and train U-Net model, saves model.
-tf_sem_seg_unet_model_infer.py: Loads trained U-Net model and makes inference on whole dataset and segments instances of cell nuclei and saves statistics in csv file
## Data folder structure
-Path to tif images: data/tissue_images
-Path to png images: data/tissue_images_png
-Path to masks: data/mask binary
## Note
Due to issues with TensorFlow tiff decoder, the .tif images were converted to .png using OpenCV and then then fed into the model. 
