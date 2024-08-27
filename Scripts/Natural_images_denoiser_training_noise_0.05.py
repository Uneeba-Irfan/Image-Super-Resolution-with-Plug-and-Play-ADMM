import pandas as pd
import os
from PIL import Image 
import deepinv as dinv
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics import StructuralSimilarityIndexMeasure


df = pd.read_csv('metadata_splits.csv')
df = df.drop(['split_closed', 'split_open'],  axis=1)

train_df = df[df['split_closed_random'] == 'train']
validation_df = df[df['split_closed_random'] == 'valid']
test_df = df[df['split_closed_random'] == 'test']

img_size = (3, 1440, 1920)
factor = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.ToTensor()

noise = 0.05

downsampling_module_with_noise = dinv.physics.Downsampling(img_size=img_size, factor=factor, device=device, noise_model=dinv.physics.GaussianNoise(sigma = noise))
downsampling_module_without_noise = dinv.physics.Downsampling(img_size=img_size, factor=factor, device=device, noise_model=dinv.physics.GaussianNoise(sigma = 0))


def image_prep_denoising(image):
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


# Custom Denoiser Natural Images

class CustomNaturalImageDataset(Dataset):
    def __init__(self, root_dir, image_size):
        self.root_dir = root_dir
        self.image_size = image_size
        
        self.device = device
        
        self.image_list = []
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                for img_file in os.listdir(subdir_path):
                    img_path = os.path.join(subdir_path, img_file)
                    self.image_list.append(img_path)

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        
        # Load and resize the image
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.image_size)

               
        clean_image, noisy_image = image_prep_denoising(image)
        clean_image = clean_image.squeeze().to(device)
        noisy_image = noisy_image.squeeze().to(device)
        
        return noisy_image, clean_image

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)  # Adjust data_range if needed
        self.alpha = alpha  # Weight for SSIM

    def forward(self, denoised, target):
        mse = self.mse_loss(denoised, target).to(device)
        ssim_value = self.ssim(denoised, target).to(device)
        return mse + self.alpha * (1 - ssim_value)  # Combine losses

num_epochs = 6
batch_size = 16

dataset = CustomNaturalImageDataset(root_dir='natural_images', image_size=(1920, 1440))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = CustomDenoiser().to(device)
if os.path.isfile("custom_denoiser_natural_imgs_noise_0.05.pt"):
    model.load_state_dict(torch.load("custom_denoiser_natural_imgs_noise_0.05.pt"))
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
    print(f"Noise Level = {noise}")
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

  # Save the final model
    torch.save(model.state_dict(), 'custom_denoiser_natural_imgs_noise_0.05.pt')
