import torch
import torch.nn as nn
import torch.nn.functional as F
from src.data import CTWithTensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import random
import numpy as np
import torch
from tqdm.auto import tqdm

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(42)

import torch
import torch.nn as nn
import torch.nn.functional as F


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


# class Autoencoder(nn.Module):
#     def __init__(self, backbone):
#         super().__init__()
#         self.encoder = backbone
#         self.decoder = CTDecoder()

#     def forward(self, x):
#         z = self.encoder(x)  # (B, 24, 24, 24, 512)
#         recon_small = self.decoder(z)  # (B, 1, 240, 480, 480)
#         return recon_small

from pytorch_msssim import ssim

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

class Autoencoder(nn.Module):
    def __init__(self, backbone, decoder):
        super().__init__()
        self.encoder = backbone
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        if isinstance(z, (tuple,list)):
            # pick first tensor if multiple returned
            for item in z:
                if torch.is_tensor(item) and item.ndim>=4:
                    z = item
                    break
        mu, sigma = self.decoder(z)
        return {'mu': mu, 'sigma': sigma, 'latent': z}

def mse_loss(x, recon):
    """
    Standard voxel-wise mean squared error
    x: original CT tensor [B,1,D,H,W]
    recon: reconstructed CT tensor [B,1,D,H,W]
    """
    return F.mse_loss(recon, x)

def ssim_loss(x, recon, data_range=1.0):
    """
    3D SSIM loss
    data_range=2.0 for [-1,1], 1.0 for [0,1]
    """

    return 1.0 - ssim((recon + 1.0) / 2, (x + 1.0) / 2, data_range=data_range, size_average=True)

def combined_loss(x, recon, lambda_ssim=0.5, data_range=1.0):
    """
    Total loss = MSE + weighted SSIM
    """
    return mse_loss(x, recon) + lambda_ssim * ssim_loss(x, recon, data_range=data_range)


def train_decoder(
    decoder: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 1,
    loss_fn = None,
    scheduler=None,
):
    """
    Обучает только декодер по готовым эмбеддингам из датасета.

    Ожидается, что батч — это (embeddings, targets) или словарь с ключами
    'embedding' и 'target'. Размер `targets` должен совпадать с выходом декодера
    (B, 1, 240, 480, 480).
    """

    decoder.to(device, dtype=torch.bfloat16)

    torch.save(decoder.eval().state_dict(), f'{0}.pt')
    decoder.train()
    tqdm.write(f'Saved model at step {0}')

    loss_history = []
    writer = SummaryWriter(log_dir='log')
    global_step = 0

    epochs_bar = tqdm(range(num_epochs), desc='Epochs', leave=True, dynamic_ncols=True)
    for epoch in epochs_bar:
        decoder.train()
        epoch_loss_sum = 0.0
        num_batches = 0

        batch_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f'Epoch {epoch + 1}/{num_epochs}',
            leave=False,
            dynamic_ncols=True
        )

        for batch_idx, batch in enumerate(batch_bar):
            ct_tensor, latent_tensor = batch[0], batch[1]
            latent_tensor = latent_tensor.permute(0, 4, 1, 2, 3).contiguous()

            embeddings = latent_tensor.to(device, dtype=torch.bfloat16)
            targets = ct_tensor.to(device, dtype=torch.bfloat16)

            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                decoded = decoder(embeddings)
                loss = loss_fn(targets, decoded, lambda_ssim=0.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            loss_value = float(loss.detach().cpu())
            epoch_loss_sum += loss_value
            writer.add_scalar('step_loss', loss_value, global_step=global_step)
            # log current learning rate

            num_batches += 1
            global_step += 1

            if global_step % 1 == 0 and global_step > 0:
                torch.save(decoder.eval().state_dict(), f'{global_step}.pt')
                decoder.train()
                tqdm.write(f'Saved model at step {global_step}')

            avg_loss = epoch_loss_sum / max(1, num_batches)
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('lr', current_lr, global_step=global_step)
            batch_bar.set_postfix(
                step=global_step,
                batch=f"{batch_idx + 1}/{len(dataloader)}",
                loss=f"{loss_value:.5f}",
                avg=f"{avg_loss:.5f}",
                lr=f"{current_lr:.2e}"
            )

        epoch_avg_loss = epoch_loss_sum / max(1, num_batches)
        loss_history.append(epoch_avg_loss)
        writer.add_scalar('epoch_loss', epoch_avg_loss, global_step=epoch + 1)
        epochs_bar.set_postfix(epoch=f"{epoch + 1}/{num_epochs}", avg=f"{epoch_avg_loss:.5f}")
    writer.close()
    return loss_history

if __name__ == "__main__":
    decoder = CTDecoder()
    dataset = CTWithTensorDataset(
        csv_path='/home/borntowarn/projects/chest-diseases/training/df.csv',
        ct_base_dir='/home/borntowarn/projects/chest-diseases/training/data/CT-RATE/dataset/train_fixed_tensors',
        latent_base_dir='/home/borntowarn/projects/chest-diseases/training/data/CT-RATE/dataset/train_fixed_tensors_embeds_not_normalized_base_large'
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=3e-5, weight_decay=1e-2)
    device = torch.device("cuda")
    num_epochs = 10
    # loss_fn = nn.MSELoss()
    total_steps = len(dataloader) * num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=3e-6,
    )

    train_decoder(decoder, dataloader, optimizer, device, num_epochs, combined_loss, scheduler)