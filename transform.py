import numpy as np
from torch import from_numpy, stack
import skimage.transform as sk_transform
from torchvision import transforms

class Normalize(object):
    """Normalize pixel values for image and depth."""
    def __call__(self, data):
        image_data, depth_data = data['image'], data['depth']
        image_data = image_data.astype(np.float32) / 255
        depth_data = depth_data.astype(np.float32) / 10
        return {'image': image_data, 'depth': depth_data}

class RandomCrop(object):
    """Randomly crop image and depth to the specified size."""
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        img, dpt = data['image'], data['depth']
        img_height, img_width = img.shape[:2]
        crop_height, crop_width = self.size

        top = np.random.randint(0, img_height - crop_height)
        left = np.random.randint(0, img_width - crop_width)

        img = img[top: top + crop_height, left: left + crop_width]
        dpt = dpt[top: top + crop_height, left: left + crop_width]

        return {'image': img, 'depth': dpt}

class RandomRotate(object):
    """Randomly rotate the image and depth within a given angle range."""
    def __init__(self, angle=5):
        self.angle = angle

    def __call__(self, data):
        img, dpt = data['image'], data['depth']
        rotation_angle = np.random.uniform(-self.angle, self.angle)
        img = sk_transform.rotate(img, rotation_angle, mode='reflect')
        dpt = sk_transform.rotate(dpt, rotation_angle, mode='reflect')
        return {'image': img, 'depth': dpt}

class CenterCrop(object):
    """Crop the image and depth map from the center."""
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        img, dpt = data['image'], data['depth']
        h, w = img.shape[:2]
        new_h, new_w = self.size
        start_h, start_w = (h - new_h) // 2, (w - new_w) // 2

        img = img[start_h:start_h + new_h, start_w:start_w + new_w]
        dpt = dpt[start_h:start_h + new_h, start_w:start_w + new_w]

        return {'image': img, 'depth': dpt}

class RandomRescale(object):
    """Rescale the image and depth by a random scale factor."""
    def __init__(self, scale=0.1):
        self.scale = scale

    def __call__(self, data):
        img, dpt = data['image'], data['depth']
        h, w = img.shape[:2]
        scale_factor = np.random.uniform(1 - self.scale, 1 + self.scale)
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)

        img = sk_transform.resize(img, (new_h, new_w), mode='reflect', anti_aliasing=True)
        dpt = sk_transform.resize(dpt, (new_h, new_w), mode='reflect', anti_aliasing=True)

        return {'image': img, 'depth': dpt}

class RandomHorizontalFlip(object):
    """Randomly flip the image and depth horizontally."""
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, data):
        img, dpt = data['image'], data['depth']
        if np.random.random() < self.probability:
            img = np.fliplr(img).copy()
            dpt = np.fliplr(dpt).copy()
        return {'image': img, 'depth': dpt}

class ToTensor(object):
    """Convert numpy arrays in sample to Tensors."""
    def __call__(self, data):
        image, depth = data['image'], data['depth']
        image = image.transpose((2, 0, 1))
        depth = np.ravel(depth)
        return {'image': from_numpy(image), 'depth': from_numpy(depth)}

class ScaleDown(object):
    """Scale down the image and depth for neural network input."""
    def __call__(self, data):
        img, dpt = data['image'], data['depth']
        img = sk_transform.resize(img, (224, 224), mode='reflect', anti_aliasing=True)
        dpt = sk_transform.resize(dpt, (25, 32), mode='reflect', anti_aliasing=True)
        return {'image': img, 'depth': dpt}
