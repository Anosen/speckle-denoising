from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt  
from numpy import mean
import torch
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import DEFAULT_DATASET_ARGS

# Define the dataset class for handling input images
class NoisyDataset(Dataset):
    """
    A dataset class for handling noisy and clean images.
    """
    def __init__(self, clean_images, speckle_mean_add=0, speckle_std_add=0.1, speckle_mean_mul=0, speckle_std_mul=1, transform=None, mult_noise_magnitude=1, addi_noise_magnitude=1, **kwargs):
        """
        Initialize the dataset.
        :param clean_images: The clean images.
        :param speckle_mean_add: The mean of the additive speckle noise.
        :param speckle_std_add: The standard deviation of the additive speckle noise.
        :param speckle_mean_mul: The mean of the multiplicative speckle noise.
        :param speckle_std_mul: The standard deviation of the multiplicative speckle noise.
        :param transform: The transforms to apply to the images.
        :param mult_noise_magnitude: The magnitude of the multiplicative noise.
        :param addi_noise_magnitude: The magnitude of the additive noise.
        """
        
        # Get the parameters from kwargs or from the method arguments if not provided in kwargs
        # Noise parameters
        self.speckle_mean_add = kwargs.get('speckle_mean_add', speckle_mean_add)
        self.speckle_std_add = kwargs.get('speckle_std_add', speckle_std_add)
        self.speckle_mean_mul = kwargs.get('speckle_mean_mul', speckle_mean_mul)
        self.speckle_std_mul = kwargs.get('speckle_std_mul', speckle_std_mul)
        
        # Additional transforms
        self.transform = kwargs.get('transform', transform)
        
        # Noise magnitudes
        self.mult_noise_magnitude = kwargs.get('mult_noise_magnitude', mult_noise_magnitude)
        self.addi_noise_magnitude = kwargs.get('addi_noise_magnitude', addi_noise_magnitude)

        # Images preprocessing
        self.clean_images = self._preprocessing(clean_images)
        self.noisy_images = self._add_speckle_noise(self.clean_images)

    def __len__(self):
        """
        Return the number of noisy and clean images.
        """
        return len(self.clean_images)

    def __getitem__(self, idx):
        """
        Return the idx-th noisy and clean images.
        """
        clean_img = self.clean_images[idx]
        noisy_img = self.noisy_images[idx]
        if self.transform:
            clean_img = self.transform(clean_img)
            noisy_img = self.transform(noisy_img)
        return noisy_img, clean_img

    def _preprocessing(self, clean_images):
        """
        Transform the input images to tensors and normalize them.
        """
        # Reshape the images to size (128, 128)
        clean_images = [img.resize((128, 128)) for img in clean_images]

        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        # Apply transforms
        clean_images = [transform(img) for img in  clean_images]

        return clean_images

    def _add_speckle_noise(self, clean_images):
        """
        Create the noisy_images dataset from the clean_images dataset.
        """

        # Create a list to store the noisy images
        noisy_images = []

        # Iterate over the clean images
        for clean_image in clean_images:
            noisy_image = self.add_speckle_noise(clean_image)
            noisy_images.append(noisy_image)

        # Normalize the noisy images
        norm = transforms.Normalize(mean=[0.5], std=[0.5])
        noisy_images = [norm(img) for img in noisy_images]

        return noisy_images


    def add_speckle_noise(self, image):
        """
        Add speckle noise to an image.

        Args:
        - image (torch.Tensor): Input image data in grayscale.

        Returns:
        - torch.Tensor: Noisy image data.
        """
        _, row, col = image.shape
        additive_gauss = torch.normal(self.speckle_mean_add, self.speckle_std_add, (row, col)).reshape(row, col)
        multiplicative_gauss = torch.normal(self.speckle_mean_mul, self.speckle_std_mul, (row, col)).reshape(row, col)
        noisy = image +  self.mult_noise_magnitude * image * multiplicative_gauss + self.addi_noise_magnitude * additive_gauss
        return noisy
    
def get_images_data(train_images_dir = "/content/train", test_images_dir = "/content/test", trim=None):
    """
    Method to load the train and test images from the specified directories.
    :param train_images_dir: The directory containing the train images.
    :param test_images_dir: The directory containing the test images.
    :param trim: The number of images to load. If None, all images are loaded.
    :return: The train and test datasets.
    """
    # Get the list of image filenames
    train_image_filenames = [filename for filename in os.listdir(train_images_dir) if filename.endswith(".tif") and not filename.endswith("mask.tif")]
    test_image_filenames = [filename for filename in os.listdir(test_images_dir) if filename.endswith(".tif") and not filename.endswith("mask.tif")]

    # Trimming to avoid a bug in test data plotting I'm too lazy to fix
    # TODO: Properly resolve the plotting bug when no trim is applied
    if trim is None:
        trim = 5500

    if trim:
        train_image_filenames = train_image_filenames[:trim]  # Use the first 10 images for demonstration
        test_image_filenames = test_image_filenames[:trim]  # Use the first 10 images for demonstration

    print(f'Loading {len(train_image_filenames)} train images...')

    # Load the train images
    train_images = []
    for filename in tqdm(train_image_filenames):
        image = Image.open(os.path.join(train_images_dir, filename)).convert('L')
        train_images.append(image)

    print(f'Loading {len(test_image_filenames)} test images...')

    # Load the test images
    test_images = []
    for filename in tqdm(test_image_filenames):
        image = Image.open(os.path.join(test_images_dir, filename)).convert('L')
        test_images.append(image)

    print(f'Loaded {len(train_images)} train and {len(test_images)} test images.')
    
    return train_images, test_images


def train_test_split(train_val_dataset, test_dataset, train_portion=0.8):
    """
    Method to split the dataset into train, validation and test datasets.
    :param train_val_dataset: The train and validation dataset.
    :param test_dataset: The test dataset.
    :param train_portion: The portion of the train images to use for training.
    :return: The train, validation and test datasets.
    """

    # Print the average pixel range for the whole dataset
    print(f'The train_val images are of size {train_val_dataset.clean_images[0].shape} and have pixel values in the range [{mean([train_val_dataset.clean_images[i].min() for i in range(len(train_val_dataset.clean_images))])}, {mean([train_val_dataset.clean_images[i].max() for i in range(len(train_val_dataset.clean_images))])}].')
    print(f'The test images are of size {test_dataset.clean_images[0].shape} and have pixel values in the range [{mean([test_dataset.clean_images[i].min() for i in range(len(test_dataset.clean_images))])}, {mean([test_dataset.clean_images[i].max() for i in range(len(test_dataset.clean_images))])}].')

    # Split dataset into train and validation
    train_size = int(train_portion * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, val_size])

    return train_dataset, val_dataset, test_dataset

def get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    """
    Method to create the train, validation and test data loaders.
    :param train_dataset: The training dataset.
    :param val_dataset: The validation dataset.
    :param test_dataset: The test dataset.
    :param batch_size: The batch size.
    :return: The train, validation and test data loaders.
    """
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

def peek_dataset(num_images=5):
    """
    Display the first num_images images from the dataset.
    :param num_images: The number of images to display.
    """

    print(f'Displaying the first {num_images} images from the dataset...')
    
    train_images, test_images = get_images_data(trim=num_images)
    
    train_val_dataset = NoisyDataset(train_images, kwargs=DEFAULT_DATASET_ARGS)
    test_dataset = NoisyDataset(test_images, kwargs=DEFAULT_DATASET_ARGS)

    train_dataset, val_dataset, test_dataset = train_test_split(train_val_dataset, test_dataset, train_portion=0.8)

    # Display the first 10 train clean and noisy images
    figsize=(15, 14 - 2*num_images if num_images<=5 else 4)

    fig, axs = plt.subplots(2, num_images, figsize=figsize)
    fig.suptitle('Train Images')
    for i in range(num_images):
        if i<len(train_dataset):
            noisy_img, clean_img = train_dataset[i]
            axs[0, i].imshow(noisy_img[0], cmap='gray')
            axs[0, i].set_title('Noisy Image')
            axs[1, i].imshow(clean_img[0], cmap='gray')
            axs[1, i].set_title('Clean Image')
        axs[0, i].axis('off')
        axs[1, i].axis('off')
    plt.show()

    # Display the first 10 val clean and noisy images
    fig, axs = plt.subplots(2, num_images, figsize=figsize)
    fig.suptitle('Val Images')
    for i in range(num_images):
        if i<len(val_dataset):
            noisy_img, clean_img = val_dataset[i]
            axs[0, i].imshow(noisy_img[0], cmap='gray')
            axs[0, i].set_title('Noisy Image')
            axs[1, i].imshow(clean_img[0], cmap='gray')
            axs[1, i].set_title('Clean Image')
        axs[0, i].axis('off')
        axs[1, i].axis('off')
    plt.show()

    # Display the first 10 test clean and noisy images
    fig, axs = plt.subplots(2, num_images, figsize=figsize)
    fig.suptitle('Test Images')
    for i in range(num_images):
        if i<len(test_dataset):
            noisy_img, clean_img = test_dataset[i]
            axs[0, i].imshow(noisy_img[0], cmap='gray')
            axs[0, i].set_title('Noisy Image')
            axs[1, i].imshow(clean_img[0], cmap='gray')
            axs[1, i].set_title('Clean Image')
        axs[0, i].axis('off')
        axs[1, i].axis('off')
    plt.show()