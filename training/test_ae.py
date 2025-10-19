import torch
import torch.nn as nn
import torch.nn.functional as F
from src.data import CTRATEInferenceDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import random
import numpy as np
import torch
from tqdm.auto import tqdm
from pathlib import Path
from src.modeling import ProjectionVIT
from collections import defaultdict
def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(42)

# ---------------------------
# Decoder: 512 → (1, 240, 480, 480)
# ---------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


# class CTDecoder(nn.Module):
#     def __init__(self, in_channels=512, out_channels=1):
#         super().__init__()
#         # Input: (B, 512, 24, 24, 24)
#         # Target: (B, 1, 60, 120, 120)
#         self.deconv_layers = nn.Sequential(
#             nn.ConvTranspose3d(in_channels, 256, kernel_size=4, stride=2, padding=1),  # 48x48x48
#             nn.BatchNorm3d(256),
#             nn.Dropout(0.1),
#             nn.ReLU(),

#             nn.ConvTranspose3d(256, 128, kernel_size=4, stride=(1, 2, 2), padding=1),  # 48x96x96
#             nn.BatchNorm3d(128),
#             nn.Dropout(0.1),
#             nn.ReLU(),

#             nn.ConvTranspose3d(128, 64, kernel_size=4, stride=(1, 2, 2), padding=1),   # 48x192x192
#             nn.BatchNorm3d(64),
#             nn.Dropout(0.1),
#             nn.ReLU(),

#             nn.Conv3d(64, out_channels, kernel_size=3, padding=1),  # (1, 48, 192, 192)
#             nn.Tanh()
#         )

#     def forward(self, z):
#         # Input z: (B, D=24, H=24, W=24, C=512)
#         # Rearrange to (B, C, D, H, W) for Conv3D
#         x = self.deconv_layers(z)
#         # Output ≈ (B, 1, 48, 192, 192)
#         # Interpolate to (60, 120, 120)
#         x = F.interpolate(x, size=(240, 480, 480), mode='trilinear', align_corners=False)
#         return x.squeeze(1)


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class CTDecoder(nn.Module):
    def __init__(self, in_channels=512, base_channels=256, target_size=(240,480,480)):
        super().__init__()
        self.target_size = target_size
        self.initial_conv = ConvBlock3D(in_channels, base_channels)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=False),
            ConvBlock3D(base_channels, base_channels//2)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=False),
            ConvBlock3D(base_channels//2, base_channels//4)
        )
        self.final_conv = nn.Conv3d(base_channels//4, 1, 3, padding=1)  # 2 channels: mu, logvar

    def forward(self, z):
        x = self.initial_conv(z)
        x = self.up1(x)
        x = self.up2(x)
        out = self.final_conv(x)
        out = F.interpolate(out, size=self.target_size, mode='trilinear', align_corners=False)
        return out.squeeze(1)


def nll_loss(x, mu, sigma, eps=1e-6):
    """Gaussian negative log-likelihood"""
    # print(x.shape, mu.shape, sigma.shape)
    loss = 0.5*((x - mu)**2 / (sigma**2 + eps) + torch.log(sigma**2 + eps))
    return loss.mean()

def ssim_loss(x, mu):
    """3D SSIM loss"""
    return 1.0 - ssim(x, mu, data_range=1.0, size_average=True)

def combined_prob_loss(x, mu, sigma, lambda_ssim=0.0):
    """Total loss = NLL + SSIM"""
    return nll_loss(x, mu, sigma) + lambda_ssim * ssim_loss(x, mu)


if __name__ == "__main__":
    decoders = []
    for i, name in enumerate(['0', '2', '4', '6', '8', '10']):
        decoder = CTDecoder()
        decoder.load_state_dict(torch.load(f'/home/borntowarn/projects/chest-diseases/training/{name}.pt'))
        decoder.eval()
        decoder.cuda()
        decoders.append(decoder)

    import pandas as pd
    ds = pd.read_csv('/home/borntowarn/projects/chest-diseases/training/data/CT-RATE/dataset/multi_abnormality_labels/valid_predicted_labels.csv')
    files = []
    for i, row in ds.iterrows():
        fname = row['VolumeName']
        if not any(row.iloc[1:]):
            files.append(fname)
            if len(files) > 50:
                break
    for j, row in ds.iterrows():
        fname = row['VolumeName']
        if any(row.iloc[1:]):
            files.append(fname)
            if len(files) > 100:
                break
    
    base_dir = Path('/home/borntowarn/projects/chest-diseases/training/data/CT-RATE/dataset/valid_fixed')
    pathes = [base_dir / fname.rsplit('_', 2)[0] / fname.rsplit('_', 1)[0] / fname for fname in files]
    print(len(pathes))

    dataset = CTRATEInferenceDataset(
        pathes=pathes
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=1)

    model_base = ProjectionVIT()
    model_base.load_state_dict(torch.load("/home/borntowarn/projects/chest-diseases/training/weights/CT-RATE/ProjectionVIT_Base_V2.pt"))
    model_base.eval()
    model_base.cuda()

    results_folder = defaultdict(list)

    with torch.no_grad():
        for batch in dataloader:
            ct_tensor, names = batch
            latent_tensor = model_base(ct_tensor.cuda())
            latent_tensor = latent_tensor.permute(0, 4, 1, 2, 3).contiguous()
            for decoder in decoders:
                recon = decoder(latent_tensor)

                recon_error = torch.mean((ct_tensor.squeeze(1) - recon.cpu()) ** 2, dim=(1, 2, 3)).cpu().tolist()
                print(recon_error)
                for name, error in zip(names, recon_error):
                    results_folder[name].append(error)
            if len(results_folder) > 100:
                break
            print(len(results_folder))
    
    import json
    with open('results.json', 'w') as f:
        json.dump(results_folder, f)