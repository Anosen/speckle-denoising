from numpy import mean
import torch
from config import DEVICE
from dataset.download import download_data
from dataset.dataset import NoisyDataset, get_images_data, peek_dataset, train_test_split
from dataset.dataset import train_test_split
from models.autoencoder import DilatedConvAutoencoder
from torch import nn, optim
from models.unets.unets import DenoisingUNet, BatchRenormalizationUNet
from utils.utils import psnr, ssim

def main():
    # Download data
    download_data('./kaggle.json')
    
    # Peek at the dataset
    peek_dataset(5)
    
    # Get images
    train_images, test_images = get_images_data(trim=None)
    
    # Create a dict for the datasets parameters
    dataset_args = {
        'speckle_mean_add': 0.0,
        'speckle_std_add': 0.1,
        'speckle_mean_mul': 0.0,
        'speckle_std_mul': 0.1,
        'transform': None,
        'mult_noise_magnitude': 1.0,
        'addi_noise_magnitude': 1.0,
    }
    
    # Create two datasets with the loaded images
    train_val_dataset = NoisyDataset(train_images, kwargs=dataset_args)
    test_dataset = NoisyDataset(test_images, kwargs=dataset_args)
    
    # Split datasets  
    train_dataset, val_dataset, test_dataset = train_test_split(train_val_dataset, test_dataset, train_portion=0.8)

    # Define the models    
    model_dict = {'di_conv_ae': DilatedConvAutoencoder(), 
                  'd_u_net': DenoisingUNet(), 
                  'br_u_net': BatchRenormalizationUNet()
                  }

    # Train and test the networks
    for model_name, model in model_dict.items():
        print(f'Training model: {model_name}')
        _, _, _, _, _, _ = model.train_test(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=32,
            epochs=3,
            loss_func=nn.MSELoss(),
            optimizer=optim.Adam(model.parameters(), lr=0.001),
            verbose=True
            )

    # Evaluate the PSNR and SSIM of the model on the test dataset
    for model_name, model in model_dict.items():
        model.eval()
        psnr_list = []
        ssim_list = []
        with torch.no_grad():
            for noisy_img, clean_img in test_dataset:

                noisy_img = noisy_img.unsqueeze(0)
                clean_img = clean_img.unsqueeze(0)

                outputs = model(noisy_img.to(DEVICE))
                psnr_list.append(psnr(outputs.to(DEVICE), clean_img.to(DEVICE)))
                ssim_list.append(ssim(outputs.to(DEVICE), clean_img.to(DEVICE)))
                
        print(f"Model: {model_name}, PSNR: {mean(psnr_list)} | SSIM: {mean(ssim_list)}")

if __name__ == "__main__":
    main()