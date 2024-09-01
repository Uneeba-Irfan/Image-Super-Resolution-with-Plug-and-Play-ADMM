import pandas as pd
import os
from PIL import Image 
import deepinv as dinv
from deepinv.models import TVDenoiser
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from tqdm import tqdm

# Load and prepare the dataset
df = pd.read_csv('metadata_splits.csv')
df = df.drop(['split_closed', 'split_open'], axis=1)
train_df = df[df['split_closed_random'] == 'train']
test_df = df[df['split_closed_random'] == 'test']
sampled_df = test_df.sample(n=100, random_state=42)

# Define parameters
img_size = (3, 1440, 1920)  # Image size (C, H, W)
factor = 3                    # Downsampling factor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.ToTensor()  # Transform to convert images to tensors
noise_values = [0.1625, 0.2]

# Initialize downsampling modules without noise
downsampling_module_without_noise = dinv.physics.Downsampling(
    img_size=img_size,
    factor=factor,
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=0)
)

def image_prep_denoising(image, noise_sigma):
    """Prepare the image for denoising by downsampling with and without noise."""
    prep_image = transform(image).to(device)
    true_image_downsampled_without_noise = downsampling_module_without_noise(prep_image.unsqueeze(0)).to(device)

    # Create downsampling module with the specified noise level
    downsampling_module_with_noise = dinv.physics.Downsampling(
        img_size=img_size,
        factor=factor,
        device=device,
        noise_model=dinv.physics.GaussianNoise(sigma=noise_sigma)
    )
    true_image_downsampled_with_noise = downsampling_module_with_noise(prep_image.unsqueeze(0)).to(device)
    
    return true_image_downsampled_without_noise, true_image_downsampled_with_noise

# Loop over each noise value
for noise in noise_values:
    print(f"Noise = {noise}")

    # Initialize results list for the current noise level
    results = []

    for index, row in sampled_df.iterrows():
        image_path = os.path.join(row["file_name"])
        img = Image.open(image_path)
        clean_img, noisy_img = image_prep_denoising(img, noise)  # Pass the noise value

        best_ths = None
        best_psnr = -float('inf')

        # Define threshold values to test
        ths_values = np.arange(0.01, 0.4, 0.01)

        for ths in ths_values:
            def tv_denoising(image, tau=0.01, rho=1.99, n_it_max=1000, crit=1e-5, ths=ths):
                tv_denoiser = TVDenoiser(verbose=False, tau=tau, rho=rho, n_it_max=n_it_max, crit=crit)
                denoised_image = tv_denoiser.forward(image, ths=ths)
                return denoised_image
            
            # Denoise the noisy image
            denoised_img = tv_denoising(noisy_img, tau=0.01, rho=1.99, ths=ths)
            psnr_metric = dinv.loss.PSNR(max_pixel=1.0, normalize=False)
            psnr_value = psnr_metric.forward(denoised_img, clean_img)

            print(f'Image {index}: ths={ths}, PSNR={psnr_value.item()}')

            if psnr_value.item() > best_psnr:
                best_psnr = psnr_value.item()
                best_ths = ths

        print(f'\n Best ths for Image {index}: {best_ths}, PSNR={best_psnr}\n')

        results.append({
            'id': index,
            'file_name': row["file_name"],
            'Best ths': best_ths,
            'Best PSNR': best_psnr
        })

    # Save results to a CSV file for the current noise level
    results_ths = pd.DataFrame(results)
    results_ths.to_csv(f'best_ths_noise_{noise}.csv', index=False)
