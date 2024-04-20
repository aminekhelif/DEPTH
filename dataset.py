import os
import numpy as np
import h5py
import pickle
import torch
import torch.utils.data as Data
import cv2

class CustomDataset(Data.Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.data_file = 'nyu_depth_v2_labeled.mat'
        self.images_folder = os.path.join(dataset_path, 'images')
        self.depths_folder = os.path.join(dataset_path, 'depths')
        self.converted_flag = os.path.join(dataset_path, 'converted.flag')
        self.train_size = 1200  # Define the number of images for training

    def convert_dataset(self):
        if os.path.exists(self.converted_flag):
            raise Exception("Dataset is already converted.")
        os.makedirs(self.images_folder, exist_ok=True)
        os.makedirs(self.depths_folder, exist_ok=True)
        os.makedirs(os.path.join(self.images_folder, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.images_folder, 'val'), exist_ok=True)
        os.makedirs(os.path.join(self.depths_folder, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.depths_folder, 'val'), exist_ok=True)

        with h5py.File(os.path.join(self.dataset_path, self.data_file), 'r') as f:
            N = len(f['images'])
            for n in range(N):
                if n % 200 == 0:
                    print(f'Processing image {n}...')
                group = 'train' if n < self.train_size else 'val'
                index = n if n < self.train_size else n - self.train_size
                img = self._reshape_and_convert_image(f['images'][n])
                img_path = os.path.join(self.images_folder, group, f'{index:05d}.p')
                self._save_data(img, img_path)
                depth = self._reshape_and_convert_depth(f['depths'][n])
                depth_path = os.path.join(self.depths_folder, group, f'{index:05d}.p')
                self._save_data(depth, depth_path)
        with open(self.converted_flag, 'w') as flag_file:
            flag_file.write('Dataset converted.')

    def __len__(self):
        return len(os.listdir(os.path.join(self.images_folder, 'train')))

    def __getitem__(self, index):
        img_path = os.path.join(self.images_folder, 'train', '{:05d}.p'.format(index))
        depth_path = os.path.join(self.depths_folder, 'train', '{:05d}.p'.format(index))
        with open(img_path, 'rb') as f_img:
            img = pickle.load(f_img)
        with open(depth_path, 'rb') as f_depth:
            depth = pickle.load(f_depth)
        sample = {'image': img, 'depth': depth}
        if self.transform:
            augmented = self.transform(image=img, depth=depth)
            img, depth = augmented['image'], augmented['depth']

        return {'image': img, 'depth': depth}

    def _reshape_and_convert_image(self, img):
        img_reshaped = np.empty((480, 640, 3), dtype='float32')
        for i in range(3):
            img_reshaped[:, :, i] = img[i, :, :].T
        return img_reshaped

    def _reshape_and_convert_depth(self, depth):
        depth_reshaped = depth.T.astype('float32')
        return depth_reshaped

    def _save_data(self, data, path):
        with open(path, 'wb') as file:
            pickle.dump(data, file)

    def is_converted(self):
        return os.path.exists(self.converted_flag)

def main():
    dataset_path = './data/nyu_v2'
    dataset = CustomDataset(dataset_path)
    if not dataset.is_converted():
        dataset.convert_dataset()
    sample = dataset[100]
    print(sample['image'].shape)
    print(sample['depth'].shape)

    image_display = (sample['image'] * 255).astype('uint8')  # Normalize and convert for display
    image_display = cv2.cvtColor(image_display, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    depth_display = cv2.normalize(sample['depth'], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')  # Normalize depth

    cv2.imshow('Image', image_display)
    cv2.imshow('Depth', depth_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
