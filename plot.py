import os
import numpy as np
import h5py
import pickle
import cv2

# data path
data_path = './data/nyu_v2/'
data_file = 'nyu_depth_v2_labeled.mat'

# Ensure data directories exist
os.makedirs(os.path.join(data_path, 'train', 'images'), exist_ok=True)
os.makedirs(os.path.join(data_path, 'train', 'depths'), exist_ok=True)
os.makedirs(os.path.join(data_path, 'val', 'images'), exist_ok=True)
os.makedirs(os.path.join(data_path, 'val', 'depths'), exist_ok=True)

# read mat file
f = h5py.File(os.path.join(data_path, data_file), 'r')

N = len(f['images'])
train_size = 1200
images_conv = []
depths_conv = []
for n in range(N):
    if n % 200 == 0:
        print(f'{n}...')

    group = 'train' if n < train_size else 'val'
    index = n if n < train_size else n - train_size
    
    # Processing image
    img = f['images'][n]
    img_ = np.empty([480, 640, 3])
    img_[:,:,0] = img[0,:,:].T
    img_[:,:,1] = img[1,:,:].T
    img_[:,:,2] = img[2,:,:].T
    img__ = img_.astype('float32')
    images_conv.append(img__)
    img_file_path = os.path.join(data_path, group, 'images', f'{index:05d}.p')
    with open(img_file_path, 'wb') as img_file:
        pickle.dump(img__, img_file)
    
    # Processing depth
    depth = f['depths'][n]
    depth_ = np.empty([480, 640])
    depth_[:,:] = depth[:,:].T
    depth__ = depth_.astype('float32')
    depths_conv.append(depth__)
    depth_file_path = os.path.join(data_path, group, 'depths', f'{index:05d}.p')
    with open(depth_file_path, 'wb') as depth_file:
        pickle.dump(depth__, depth_file)

# plot some images with cv2
for i in range(5):
    img = images_conv[i]
    depth = depths_conv[i]
    cv2.imshow('img', img)
    cv2.imshow('depth', depth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
