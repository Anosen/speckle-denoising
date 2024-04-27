import torch
from torch import nn
from tqdm import tqdm
from config import DEVICE
import time
from datetime import timedelta

from dataset.dataset import get_data_loaders
from utils.utils import display_loss, display_results


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def test_model(self, dataloader, loss_func) -> tuple:
        """
        Method to test the model on a given dataloader.
        :param model: The model to test.
        :param dataloader: The DataLoader object containing the test data.
        :param loss_func: The loss function to use.
        :return: The outputs and the average loss.
        """
        total_loss = 0
        self.eval()
        with torch.no_grad():
            for noisy_imgs, clean_imgs in dataloader:
                outputs = self(noisy_imgs.to(DEVICE))
                loss = loss_func(outputs.to(DEVICE), clean_imgs.to(DEVICE))
                total_loss += loss.item()
        return (noisy_imgs, clean_imgs, outputs), loss / len(dataloader)
    
    # Set up the training loop
    def train_model(self, train_dataloader, val_dataloader, epochs, loss_func, optimizer):
        """
        Method to train the model on the given dataset.
        :param model: The model to train.
        :param train_val_dataset: The dataset to train on.
        :param epochs: The number of epochs to train for.
        :param loss_func: The loss function to use.
        :param optimizer: The optimizer to use.
        :param batch_size: The batch size to use.
        :return: The lists of train outputs, train losses, validation outputs, and validation losses.
        """

        self.to(DEVICE)

        # Print the model architecture
        print(self)

        # Print the number of parameters and the the memory footprint of the model
        num_params = sum(p.numel() for p in self.parameters())
        memory_footprint = sum(p.numel() * p.element_size() for p in self.parameters()) / 1024**2
        print(f"Number of parameters: {num_params:,}\n"
            f"Model memory usage: {memory_footprint:.3f}MB\n"
            f"Using: {DEVICE}")

        # Train the model
        train_output_list = []
        train_loss_list = []
        val_output_list = []
        val_loss_list = []
        self.train()
        for epoch in tqdm(range(epochs)):
            # Train the model on the training data
            total_train_loss = 0
            first_batch = True
            for noisy_imgs, clean_imgs in train_dataloader:
                if first_batch:
                    start = time.time()
                optimizer.zero_grad()
                outputs = self(noisy_imgs.to(DEVICE))
                loss = loss_func(outputs.to(DEVICE), clean_imgs.to(DEVICE))
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                end = time.time()
                if first_batch:
                    print(f"Estimated time for epoch: {timedelta(seconds=end - start)*len(train_dataloader)}.")
                    first_batch = False

            # Store the train outputs and losses
            train_output_list.append((noisy_imgs, clean_imgs, outputs))
            train_loss_list.append(total_train_loss/len(train_dataloader))

            # Infer the model on the validation data
            val_outputs, val_loss = self.test_model(val_dataloader, loss_func)

            # Store the validation outputs and losses
            val_output_list.append(val_outputs)
            val_loss_list.append(val_loss.item())
            print(f"Epoch {epoch}, Train Loss: {loss.item():.4f} - Val Loss: {val_loss.item():.4f}")

            # Early stop if val_loss has increased for 2 consecutive epochs
            if epoch > 1 and val_loss_list[-1] > val_loss_list[-2] and val_loss_list[-2] > val_loss_list[-3]:
                print(f"Early stopping at epoch {epoch}.")
                break

        return train_output_list, train_loss_list, val_output_list, val_loss_list

    def train_test(self, train_dataset, val_dataset, test_dataset, epochs, loss_func, optimizer, batch_size = 10, verbose=True):
        train_dataloader, val_dataloader, test_dataloader = get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=batch_size)

        train_output_list, train_loss_list, val_output_list, val_loss_list = self.train_model(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            epochs=epochs,
            loss_func=loss_func,
            optimizer=optimizer
        )

        test_outputs, test_loss = self.test_model(test_dataloader, loss_func)

        if verbose:  # Display the results
            display_loss(train_loss_list, val_loss_list)
            print(f'Final Test Loss: {test_loss.item()}')

            last_epoch_to_show = 1
            for i in range(max(0, epochs-last_epoch_to_show), epochs):  # Display the last 3 epochs outputs
                display_results(train_dataset, train_output_list, num_images=5, name=f'Train, epoch {i}')
                display_results(val_dataset, val_output_list, num_images=5, name=f'Validation, epoch {i}')
            display_results(test_dataset, test_outputs, num_images=5, name='Test')

        return train_output_list, train_loss_list, val_output_list, val_loss_list, test_outputs, test_loss