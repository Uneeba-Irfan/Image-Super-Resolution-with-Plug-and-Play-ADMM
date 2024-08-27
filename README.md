# Image-Super-Resolution-with-Plug-and-Play

Turtle images dataset: https://www.kaggle.com/datasets/wildlifedatasets/seaturtleid2022
Natural Images dataset: https://www.kaggle.com/datasets/prasunroy/natural-images

All Code files are in the scripts folder and contain the following:

1. Image_denoising_results.ipynb:
A Jupyter Notebook that contains the results of the image denoising experiments. This notebook includes code for loading images, applying denoising techniques, and visualizing the results.

2. Turtles_images_denoiser_training_noise_0.05:
A Python script for training a denoising model on Turtles images with a noise level of 0.05. The script includes data preprocessing, model architecture definition, and training loop implementation.

3. Natural_images_denoiser_training_noise_0.05.py:
A Python script for training a denoising model on natural images with a noise level of 0.05. The script includes data preprocessing, model architecture definition, and training loop implementation.

4. PnP_noise_0.05_downsample_4x.py:
A Python script that implements the Plug-and-Play (PnP) denoising method with a noise level of 0.05 and a downsampling factor of 4x. This script demonstrates the application of the PnP method on noisy images and evaluates its performance.

5. Grid_search_TV:
This file contains the Grid search done to find the best ths value for the Total Variance Denoising used in the Image_denoising_results file
