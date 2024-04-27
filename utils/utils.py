from matplotlib import pyplot as plt
import torch


def display_results(input_dataset, output_list, num_images=5, name='Train', epoch=-1):
    """
    Method to display the first num_images images from the dataset.
    :param output_list: The list of outputs from the model.
    :param num_images: The number of images to display.
    :param name: The name of the dataset.
    """

    if not isinstance(output_list, list):  # If the output_list is a single batch, convert it to a list
        output_list = [output_list]

    # Display the first 5 train clean, noisy, and denoised images
    fig, axs = plt.subplots(3, num_images, figsize=(15, 6))
    fig.suptitle(f'{name} Images')
    for i in range(min(len(input_dataset), num_images)):
        noisy_img, clean_img, denoised_img = output_list[epoch]
        noisy_img = noisy_img[i].squeeze()
        clean_img = clean_img[i].squeeze()
        denoised_img = denoised_img[i].squeeze()
        axs[0, i].imshow(noisy_img, cmap='gray')
        axs[0, i].axis('off')
        axs[0, i].set_title('Noisy Image')
        axs[1, i].imshow(clean_img, cmap='gray')
        axs[1, i].axis('off')
        axs[1, i].set_title('Clean Image')
        axs[2, i].imshow(denoised_img.detach().cpu(), cmap='gray')
        axs[2, i].axis('off')
        axs[2, i].set_title('Denoised Image')
    plt.show()

def display_loss(train_loss_list, val_loss_list):
    # Print the loss curve
    plt.figure()
    plt.plot(train_loss_list, label='Train')
    plt.plot(val_loss_list, label='Validation')
    plt.title('Loss Curve')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    
def ssim(noisy_img, clean_img):
    """Compute the structural similarity index (SSIM) between two images."""
    mu1 = torch.mean(noisy_img)
    mu2 = torch.mean(clean_img)
    sigma1 = torch.std(noisy_img)
    sigma2 = torch.std(clean_img)
    cov = torch.mean((noisy_img - mu1) * (clean_img - mu2))
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2
    ssim = (2 * mu1 * mu2 + c1) * (2 * cov + c2) / ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
    return ssim.detach().cpu().item()

def psnr(noisy_img, clean_img):
    """Compute the peak signal-to-noise ratio (PSNR) between two images."""
    mse = torch.mean((noisy_img - clean_img)**2)
    psnr = 10 * torch.log10(1 / mse)
    return psnr.detach().cpu().item()
