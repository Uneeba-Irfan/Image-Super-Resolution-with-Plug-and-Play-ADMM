import pandas as pd
import os
from PIL import Image
import deepinv as dinv
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics import StructuralSimilarityIndexMeasure


df = pd.read_csv('metadata_splits.csv')
df = df.drop(['split_closed', 'split_open'], axis=1)

train_df = df[df['split_closed_random'] == 'train']
validation_df = df[df['split_closed_random'] == 'valid']
test_df = df[df['split_closed_random'] == 'test']

img_size = (3, 1440, 1920)
factor = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.ToTensor()

# List of noise levels
noise_levels = [0.05, 0.0875, 0.125, 0.1625, 0.2]

def image_prep_denoising(image, noise):
    downsampling_module_with_noise = dinv.physics.Downsampling(img_size=img_size, factor=factor, device=device, noise_model=dinv.physics.GaussianNoise(sigma=noise))
    downsampling_module_without_noise = dinv.physics.Downsampling(img_size=img_size, factor=factor, device=device, noise_model=dinv.physics.GaussianNoise(sigma=0))
    
    prep_image = transform(image).to(device)
    true_image_downsampled_without_noise = downsampling_module_without_noise(prep_image.unsqueeze(0)).to(device)
    true_image_downsampled_with_noise = downsampling_module_with_noise(prep_image.unsqueeze(0)).to(device)
    return true_image_downsampled_without_noise, true_image_downsampled_with_noise

class CustomDenoiser(nn.Module):
    def __init__(self):
        super(CustomDenoiser, self).__init__()
        
        # Encoder
        self.c1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.c3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.c4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.c5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.c6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.c7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.p3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.c8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.c9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.c10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
        # Decoder
        self.u1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.c11 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.c12 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.c13 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.u2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.c14 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.c15 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.u3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.c16 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.c17 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.output_layer = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = F.relu(self.c1(x))
        c1 = F.relu(self.c2(c1))
        p1 = self.p1(c1)
        
        c2 = F.relu(self.c3(p1))
        c2 = F.relu(self.c4(c2))
        p2 = self.p2(c2)
        
        c3 = F.relu(self.c5(p2))
        c3 = F.relu(self.c6(c3))
        c3 = F.relu(self.c7(c3))
        p3 = self.p3(c3)
        
        c4 = F.relu(self.c8(p3))
        c4 = F.relu(self.c9(c4))
        c4 = F.relu(self.c10(c4))
        
        # Decoder
        u1 = self.u1(c4)
        u1 = torch.cat((u1, c3), dim=1)
        c5 = F.relu(self.c11(u1))
        c5 = F.relu(self.c12(c5))
        c5 = F.relu(self.c13(c5))
        
        u2 = self.u2(c5)
        u2 = torch.cat((u2, c2), dim=1)
        c6 = F.relu(self.c14(u2))
        c6 = F.relu(self.c15(c6))
        
        u3 = self.u3(c6)
        u3 = torch.cat((u3, c1), dim=1)
        c7 = F.relu(self.c16(u3))
        c7 = F.relu(self.c17(c7))
        
        outputs = self.output_layer(c7)
        
        return torch.sigmoid(outputs)

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, image_size, noise):
        self.root_dir = root_dir
        self.image_size = image_size
        self.noise = noise
        self.image_list = train_df['file_name'].to_list()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = Image.open(img_path)
        image = image.resize(self.image_size)
        
        clean_image, noisy_image = image_prep_denoising(image, self.noise)
        clean_image = clean_image.squeeze().to(device)
        noisy_image = noisy_image.squeeze().to(device)
        
        return noisy_image, clean_image

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)  
        self.alpha = alpha  

    def forward(self, denoised, target):
        mse = self.mse_loss(denoised, target).to(device)
        ssim_value = self.ssim(denoised, target).to(device)
        return mse + self.alpha * (1 - ssim_value)  

num_epochs = 30
batch_size = 8

for noise in noise_levels:
    print(f"Training for noise level: {noise}")

    dataset = CustomImageDataset(root_dir='images', image_size=(1920, 1440), noise=noise)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CustomDenoiser().to(device)
    model_file = f"custom_denoiser_noise_{noise:.4f}.pt"
    
    if os.path.isfile(model_file):
        model.load_state_dict(torch.load(model_file))
    
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = CombinedLoss(alpha=0.3)  

    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_no, (noisy_batch, clean_batch) in tqdm(enumerate(dataloader), total=int(np.ceil(len(dataset) / batch_size))):
            optimizer.zero_grad()        
            # Forward pass
            outputs = model(noisy_batch)      
            
            loss = criterion(outputs, clean_batch)
            
            # Backward pass
            loss.backward()
            
            optimizer.step()
            
            # Total loss for current epoch
            epoch_loss += loss.item()
            
        # Avg loss/epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

    
    torch.save(model.state_dict(), model_file)
