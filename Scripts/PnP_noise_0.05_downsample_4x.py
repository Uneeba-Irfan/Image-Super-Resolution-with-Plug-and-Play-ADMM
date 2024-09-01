import deepinv as dinv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.optim.optimizers import optim_builder
from deepinv.training import test
from torchvision import transforms
from deepinv.utils.parameters import get_GSPnP_params
from deepinv.utils.demo import load_dataset, load_degradation

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

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "measurements"
RESULTS_DIR = BASE_DIR / "results_noise_0.05_4x"
DEG_DIR = BASE_DIR / "degradations"

noise = 0.05
img_size=(3, 1440, 1920)
batch_size = 4  

for i in range(10):
    folder_name = str(i) 
    torch.manual_seed(0)
    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    transform = transforms.ToTensor()
    
    operation = "super-resolution"
    val_transform = transforms.Compose(
        [transforms.ToTensor()]
    )
    
    num_workers = 4 if torch.cuda.is_available() else 0
    
    factor = 4  
    n_channels = 3  
    n_images_max = 3  
    noise_level_img = noise 
    p = dinv.physics.Downsampling(
        img_size=img_size,
        factor=factor,
        noise_model=dinv.physics.GaussianNoise(sigma = noise),
        device=device,
        filter='gaussian',
    )
    
    
    downsampling_module_with_noise = dinv.physics.Downsampling(img_size=img_size, factor=factor, device=device, noise_model=dinv.physics.GaussianNoise(sigma = noise), filter='gaussian',)
    downsampling_module_without_noise = dinv.physics.Downsampling(img_size=img_size, factor=factor, device=device, noise_model=dinv.physics.GaussianNoise(sigma = 0), filter='gaussian',)
    
    
    def image_prep_denoising(image):
        prep_image = transform(image).to(device)
        true_image_downsampled_without_noise = downsampling_module_without_noise(prep_image.unsqueeze(0)).to(device)
        #true_image_downsampled_without_noise = prep_image.unsqueeze(0).to(device)
        #true_image_downsampled_with_noise = downsampling_module_with_noise(prep_image.unsqueeze(0)).to(device)
        true_image_downsampled_with_noise = prep_image.unsqueeze(0).to(device)
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
    
        def forward(self, x, unused_noise_level):
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
                elif os.path.isfile(subdir_path):
                    self.image_list.append(subdir_path)
    
        def __len__(self):
            return len(self.image_list)
        
        def __getitem__(self, idx):
            img_path = self.image_list[idx]
            

            image = Image.open(img_path).convert('RGB')
            image = image.resize(self.image_size)
    
                   
            clean_image, noisy_image = image_prep_denoising(image)
            clean_image = clean_image.squeeze().to(device)
            noisy_image = noisy_image.squeeze().to(device)
            
            return noisy_image, clean_image
            
    
    
    dataset = CustomImageDataset(root_dir='test_images/' + folder_name, image_size=(1920, 1440))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    

    early_stop = True  
    crit_conv = "residual"  

    thres_conv = 1e-4
    backtracking = True
    use_bicubic_init = False  

    lamb, sigma_denoiser, stepsize, max_iter = get_GSPnP_params(operation, noise_level_img)
    
    params_algo = {
        "stepsize": stepsize,
        "g_param": sigma_denoiser,
        "lambda": lamb,
    }
    

    data_fidelity = L2()
    
    custom_denoiser = CustomDenoiser().to(device)
    custom_denoiser.load_state_dict(torch.load('custom_denoiser_noise_0.05.pt'))

    
    class MyPnPPrior(PnP):
        """
        Custom PnP Prior using the custom denoiser.
        """
    
        def __init__(self, denoiser, *args, **kwargs):
            super().__init__(denoiser, *args, **kwargs)
    
        def g(self, x, *args, **kwargs):
            """
            Computes the prior :math:`g(x)` using the denoiser's potential.
    
            :param torch.Tensor x: Variable :math:`x` at which the prior is computed.
            :return: (torch.Tensor) prior :math:`g(x)`.
            """
    
            denoised = self.denoiser(x)
            

            residual = x - denoised
            

            return 0.5 * torch.norm(residual, p=2)**2
    
    prior = MyPnPPrior(denoiser=custom_denoiser)
    
    

    model = optim_builder(
        iteration="ADMM",
        prior=prior,
        g_first=True,
        data_fidelity=data_fidelity,
        params_algo=params_algo,
        early_stop=early_stop,
        max_iter=max_iter,
        crit_conv=crit_conv,
        thres_conv=thres_conv,
        backtracking=backtracking,
        verbose=True,
    )
    

    model.eval()
    
    save_folder = RESULTS_DIR / folder_name
    wandb_vis = False 
    plot_metrics = True 
    plot_images = True 
    
    with torch.no_grad():
        test(
            model=model,
            test_dataloader=dataloader,
            physics=p,
            device=device,
            plot_images=plot_images,
            save_folder=save_folder ,
            plot_metrics=plot_metrics,
            verbose=True,
            wandb_vis=wandb_vis,
            plot_only_first_batch=False,  
        )