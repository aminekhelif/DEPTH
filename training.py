import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from albumentations import Compose, HorizontalFlip, Resize, Normalize, Rotate, RandomCrop
from albumentations.pytorch import ToTensorV2
from dataset import CustomDataset
from model import DepthEstimationNet
import copy
import logging

# Set the device with MPS support for Apple Silicon if available
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Set up logging
def setup_logging(save_dir):
    logfile = os.path.join(save_dir, 'training.log')
    logging.basicConfig(level=logging.INFO, filename=logfile, filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

class DBELoss(nn.Module):
    def __init__(self, a1=1.5, a2=-0.1, device=device):
        super(DBELoss, self).__init__()
        self.a1 = a1
        self.a2 = a2
        self.device = device

    def forward(self, estimated_depth, ground_truth_depth):
        estimated_depth = estimated_depth.to(self.device)
        ground_truth_depth = ground_truth_depth.to(self.device)
        diff = estimated_depth - ground_truth_depth
        balanced_diff = self.a1 * diff + self.a2 * torch.square(diff)
        return torch.mean(torch.square(balanced_diff))

# Define custom transform functions using Albumentations
def get_training_augmentation():
    return Compose([
        Rotate(limit=35, p=0.5),
        HorizontalFlip(p=0.5),
        RandomCrop(height=427, width=561, always_apply=True),
        Resize(240, 320),  # Resize the image to the required input size
        Normalize(),  # Normalize the image
        ToTensorV2(),  # Convert to tensor
    ])

def get_validation_augmentation():
    return Compose([
        RandomCrop(height=427, width=561, always_apply=True),
        Resize(240, 320),  # Resize the image to the required input size
        Normalize(),  # Normalize the image
        ToTensorV2(),  # Convert to tensor
    ])

# Wrapper class for dataset to apply transformations
class AlbumentationsWrapper(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        augmented = self.transform(image=image, depth=depth)
        sample['image'], sample['depth'] = augmented['image'], augmented['depth']
        return sample

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, save_dir, checkpoint_interval=5, early_stopping_patience=10):
    best_loss = float('inf')
    model.to(device)
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        logging.info(f'Epoch {epoch+1}/{num_epochs}')
        logging.info('-' * 10)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0

            for batch in tqdm(dataloaders[phase], desc=f'{phase} Progress'):
                inputs = batch['image'].to(device).float()
                labels = batch['depth'].to(device).float()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()

            if phase == 'val':
                scheduler.step(running_loss)
                epoch_loss = running_loss / len(dataloaders['val'])
                logging.info(f'Validation Loss: {epoch_loss:.4f}')
                if epoch_loss < best_loss:
                    logging.info(f'Saving best model with loss {epoch_loss:.4f}')
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, os.path.join(save_dir, 'best_model.pth'))
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= early_stopping_patience:
                        logging.info('Early stopping triggered')
                        break

        # Checkpoint saving
        if epoch % checkpoint_interval == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f'Checkpoint saved to {checkpoint_path}')

        if early_stopping_counter >= early_stopping_patience:
            break

        logging.info('')

    logging.info(f'Training complete. Best loss: {best_loss:.4f}')
    return model


def main():
    # Hyperparameters
    num_epochs = 50
    batch_size = 4
    learning_rate = 1e-4

    # Directories and paths are set relative to the dataset.py file structure
    project_dir = os.getcwd()  # Assumes training.py is run from the project root
    data_dir = os.path.join(project_dir, 'data', 'nyu_v2')
    save_dir = os.path.join(project_dir, 'models', 'training_runs')
    os.makedirs(save_dir, exist_ok=True)
    setup_logging(save_dir)

    # Transforms
    train_transforms = AlbumentationsWrapper(get_training_augmentation())
    val_transforms = AlbumentationsWrapper(get_validation_augmentation())

    # Datasets and Dataloaders
    train_dataset = CustomDataset(os.path.join(data_dir, 'images', 'train'), transform=train_transforms)
    val_dataset = CustomDataset(os.path.join(data_dir, 'images', 'val'), transform=val_transforms)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size)
    }

    # Model
    model = DepthEstimationNet(device=device)

    # Loss Function
    criterion = DBELoss(device=device)

    # Optimizer and Scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # Train the model
    trained_model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, save_dir)


if __name__ == '__main__':
    main()
