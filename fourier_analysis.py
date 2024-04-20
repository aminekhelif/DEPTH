import os
import torch
import pickle
from skimage import transform
from torch.utils.data import DataLoader
from model import DepthEstimationNet
from dataset import CustomDataset
import numpy as np
from albumentations.pytorch import ToTensorV2
from albumentations import Compose, Resize, Normalize

class FDC:
    def __init__(self, model, depth_size=(25, 32), crop_ratios=None, device=None):
        self.depth_size = depth_size
        self.ncoeff = self.depth_size[0] * (math.floor(self.depth_size[1] / 2) + 1) * 2
        self.crop_ratios = crop_ratios or [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
        self.device = device or (torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'))
        self.model = model.to(self.device)
        self.weights = None
        self.bias = None

    def process_batch(self, batch):
        with torch.no_grad():
            results = self.model(batch.view(-1, *batch.shape[-3:]))
        return results

    def predict(self, f_m_hat):
        return torch.sum(self.weights * (f_m_hat - self.bias), 0)

    def img2fourier(self, images):
        return torch.rfft(images, 2, onesided=False).view(images.size(0), -1)

    def fourier2img(self, images_fd):
        img_fd = images_fd.view(-1, *self.depth_size, 2)
        return torch.irfft(img_fd, 2, onesided=False)

    def merge_crops(self, crops):
        merged_crops = torch.empty((len(self.crop_ratios), *self.depth_size))
        for i, ratio in enumerate(self.crop_ratios):
            scale_factor = 1 / ratio
            crop_size = (round(self.depth_size[0] / ratio), round(self.depth_size[1] / ratio))
            merged = torch.zeros(crop_size)
            weights = torch.zeros(crop_size)

            for j in range(4):
                crop = transform.resize(crops[j].cpu().numpy() / scale_factor, crop_size, mode='reflect',
                                        anti_aliasing=True, preserve_range=True)
                y, x = np.divmod(j, 2)
                merged[y::2, x::2] = crop
                weights[y::2, x::2] += 1

            merged /= weights
            merged = transform.resize(merged, self.depth_size, mode='reflect',
                                      anti_aliasing=True, preserve_range=True)
            merged_crops[i] = torch.from_numpy(merged).to(self.device)

        return merged_crops

    def __call__(self, batch):
        predictions = []
        results = self.process_batch(batch.to(self.device))
        for i in range(0, results.size(0), len(self.crop_ratios)):
            candidates = self.merge_crops(results[i:i + len(self.crop_ratios)])
            f_m_hat = self.img2fourier(candidates)
            f_hat = self.predict(f_m_hat)
            predictions.append(self.fourier2img(f_hat.view(1, -1)))
        return predictions

    def fit(self, f_m_hat, f):
        """
        f_m_hat: (T, M, K)
        f: (T, K)
        Fit model weights based on Fourier transformed images and their corresponding depths.
        """
        print('Fitting FDC weights...')
        T, M, K = f_m_hat.shape
        self.weights = torch.zeros(M, K, dtype=torch.float32, device=f_m_hat.device)
        self.bias = torch.zeros(M, K, dtype=torch.float32, device=f_m_hat.device)

        for k in range(K):
            t_k = f[:, k].unsqueeze_(1).to(torch.float32)
            b_k = torch.mean(f_m_hat[:, :, k].to(torch.float32) - t_k, 0, True)
            T_k = f_m_hat[:, :, k].to(torch.float32) - b_k

            # Using SVD to compute the pseudoinverse manually
            U, S, V = torch.svd(T_k)
            # Create S_inv (inverse of S where we ignore zero singular values)
            S_inv = torch.zeros_like(S)
            tolerance = 1e-5
            non_zero_elements = S > tolerance
            S_inv[non_zero_elements] = 1.0 / S[non_zero_elements]
            S_inv = torch.diag(S_inv)

            # Compute pseudoinverse of T_k
            T_k_pinv = V.matmul(S_inv).matmul(U.transpose(-2, -1))
            w_k = torch.mm(T_k_pinv, t_k)

            self.weights[:, k] = w_k.squeeze()
            self.bias[:, k] = b_k.squeeze()

        print("Weights and biases fitted successfully.")
    def save_weights(self, path_to_dir):
        with open(os.path.join(path_to_dir, 'fdc_weights.p'), 'wb') as w:
            pickle.dump(self.weights.cpu(), w)
        with open(os.path.join(path_to_dir, 'fdc_bias.p'), 'wb') as b:
            pickle.dump(self.bias.cpu(), b)

    def load_weights(self, path_to_dir):
        with open(os.path.join(path_to_dir, 'fdc_weights.p'), 'rb') as w:
            self.weights = pickle.load(w).to(self.device)
        with open(os.path.join(path_to_dir, 'fdc_bias.p'), 'rb') as b:
            self.bias = pickle.load(b).to(self.device)

def main():
    seed = 2
    torch.manual_seed(seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Set paths
    data_path = './data/nyu_v2/'
    weights_path = './models/temp_v3/042_model.pt'
    save_dir = './models/FDC/den_dbe/'
    os.makedirs(save_dir, exist_ok=True)

    # Load model
    model = DepthEstimationNet(device=device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Define transforms
    transform_pipeline = Compose([
        Resize(240, 320),
        Normalize(),
        ToTensorV2()
    ])

    # Load dataset
    nyu_dataset = CustomDataset(data_path, transform=transform_pipeline)
    dataloader = DataLoader(nyu_dataset, batch_size=1, shuffle=True, num_workers=6)

    # Initialize and train FDC model
    fdc_model = FDC(model, device=device)
    # Note: Implement the forward and fit methods in FDC class as per your previous code and logic.
    fdc_model.forward(dataloader)
    fdc_model.fit()
    fdc_model.save_weights(save_dir)

    print(f'FDC weights saved in {save_dir}')

if __name__ == '__main__':
    main()