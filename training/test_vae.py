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
from src.data import CTRATEInferenceDataset

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(42)

from pathlib import Path
from src.modeling import ProjectionVIT
from collections import defaultdict

import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
try:
    from pytorch_msssim import ssim
    HAVE_SSIM = True
except Exception:
    HAVE_SSIM = False


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1, use_bn=True):
        super().__init__()
        layers = [nn.Conv3d(in_ch, out_ch, kernel_size=kernel, padding=padding)]
        if use_bn:
            layers.append(nn.BatchNorm3d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResidualBlock3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv3d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(ch)
        self.conv2 = nn.Conv3d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + res
        return self.act(out)


class SelfAttention3D(nn.Module):
    """
    Lightweight 3D self-attention at bottleneck.
    Use cautiously if memory is limited.
    """
    def __init__(self, channels, num_heads=4, proj_factor=1):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = (channels // proj_factor) // num_heads
        self.scale = self.head_dim ** -0.5
        # project into lower-dim qkv if proj_factor > 1
        out_ch = channels // proj_factor
        self.q = nn.Conv3d(channels, out_ch, kernel_size=1)
        self.k = nn.Conv3d(channels, out_ch, kernel_size=1)
        self.v = nn.Conv3d(channels, out_ch, kernel_size=1)
        self.out = nn.Conv3d(out_ch, channels, kernel_size=1)

    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        _, Cp, Dp, Hp, Wp = q.shape
        S = Dp * Hp * Wp
        # reshape: (B, heads, S, head_dim)
        q = q.view(B, self.num_heads, Cp // self.num_heads, S).permute(0,1,3,2)
        k = k.view(B, self.num_heads, Cp // self.num_heads, S).permute(0,1,3,2)
        v = v.view(B, self.num_heads, Cp // self.num_heads, S).permute(0,1,3,2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, heads, S, S)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, heads, S, head_dim)
        out = out.permute(0,1,3,2).contiguous().view(B, Cp, Dp, Hp, Wp)
        out = self.out(out)
        return out + x


class Encoder3D(nn.Module):
    def __init__(self,
                 in_ch: int = 1,
                 base_ch: int = 32,
                 latent_ch: int = 128,
                 num_down: int = 4,
                 use_attn: bool = True):
        """
        Produces mu and logvar maps of shape (B, latent_ch, D', H', W')
        Assumes input spatial dims divisible by 2**num_down.
        For input (240,480,480) and num_down=4 => (15,30,30).
        """
        super().__init__()
        self.num_down = num_down
        self.base_ch = base_ch

        # initial conv
        self.init = ConvBlock3D(in_ch, base_ch)

        # create downsample stages
        ch = base_ch
        downs = []
        for i in range(num_down):
            downs.append(nn.Sequential(
                ResidualBlock3D(ch),
                nn.Conv3d(ch, ch * 2, kernel_size=4, stride=2, padding=1),  # halving spatial dims
                nn.BatchNorm3d(ch * 2),
                nn.ReLU(inplace=True)
            ))
            ch *= 2
        self.downs = nn.ModuleList(downs)
        bottleneck_ch = ch  # base_ch * 2**num_down

        # optional attention at bottleneck
        self.use_attn = use_attn
        if use_attn:
            self.attn = SelfAttention3D(bottleneck_ch, num_heads=8, proj_factor=1)

        # project to mu/logvar maps
        self.mu_conv = nn.Conv3d(bottleneck_ch, latent_ch, kernel_size=1)
        self.logvar_conv = nn.Conv3d(bottleneck_ch, latent_ch, kernel_size=1)

    def forward(self, x):
        # x: (B,1,240,480,480)
        x = self.init(x)
        for down in self.downs:
            x = down(x)
        if self.use_attn:
            x = self.attn(x)
        mu = self.mu_conv(x)         # (B, latent_ch, D', H', W')
        logvar = self.logvar_conv(x) # same shape
        return mu, logvar


class Decoder3D(nn.Module):
    def __init__(self,
                 out_ch: int = 1,
                 base_ch: int = 32,
                 latent_ch: int = 128,
                 num_up: int = 4,
                 target_size: Tuple[int,int,int] = (240,480,480),
                 output_activation: str = 'tanh'):
        """
        Takes z_map: (B, latent_ch, D', H', W') and reconstructs full volume.
        """
        super().__init__()
        self.num_up = num_up
        self.base_ch = base_ch
        self.latent_ch = latent_ch
        self.target_size = target_size
        # first expand latent channels to bottleneck channels
        bottleneck_ch = base_ch * (2 ** num_up)
        self.expand = nn.Conv3d(latent_ch, bottleneck_ch, kernel_size=1)

        # upsample stages (mirror of encoder)
        ups = []
        ch = bottleneck_ch
        for i in range(num_up):
            ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                ResidualBlock3D(ch),
                nn.Conv3d(ch, ch // 2, kernel_size=3, padding=1),
                nn.BatchNorm3d(ch // 2),
                nn.ReLU(inplace=True)
            ))
            ch = ch // 2
        self.ups = nn.ModuleList(ups)

        self.final_conv = nn.Conv3d(ch, out_ch, kernel_size=3, padding=1)
        if output_activation == 'tanh':
            self.final_act = nn.Tanh()
        elif output_activation == 'sigmoid':
            self.final_act = nn.Sigmoid()
        else:
            self.final_act = nn.Identity()

    def forward(self, z_map):
        # z_map: (B, latent_ch, D', H', W')
        x = self.expand(z_map)  # (B, bottleneck_ch, D', H', W')
        for up in self.ups:
            x = up(x)
        x = self.final_conv(x)
        x = self.final_act(x)
        x = F.interpolate(x, size=self.target_size, mode='trilinear', align_corners=False)
        return x


class BetaVAE3D(nn.Module):
    def __init__(self,
                 in_ch: int = 1,
                 base_ch: int = 32,
                 latent_ch: int = 128,
                 num_down: int = 4,
                 use_attn: bool = True,
                 output_activation: str = 'tanh',
                 target_size: Tuple[int,int,int] = (240,480,480)):
        super().__init__()
        self.encoder = Encoder3D(in_ch, base_ch, latent_ch, num_down, use_attn)
        self.decoder = Decoder3D(in_ch, base_ch, latent_ch, num_up=num_down, target_size=target_size,
                                        output_activation=output_activation)

    def reparameterize(self, mu, logvar):
        # mu, logvar shape: (B, C, D', H', W')
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)

        return recon, mu, logvar

    def sample(self, n: int = 1, device: str = 'cpu'):
        """
        Sample from the prior z ~ N(0,1) with shape same as encoder output spatial dims.
        We need to know spatial dims; easiest way: run a dummy input through encoder to get shape.
        """
        # get shape by passing dummy
        with torch.no_grad():
            # small dummy with same device
            dummy = torch.zeros(1, 1, 240, 480, 480, device=device)
            mu_dummy, logvar_dummy = self.encoder(dummy)
            C, Dp, Hp, Wp = mu_dummy.shape[1:]
            z = torch.randn(n, C, Dp, Hp, Wp, device=device)
            samp = self.decoder(z)
        return samp


def spatial_kld(mu, logvar):
    """
    Element-wise KL between q(z|x)=N(mu, var) and p(z)=N(0,1).
    Sum over channel+spatial dims, mean over batch.
    """
    # KL per element: 0.5*(mu^2 + var - logvar - 1)
    var = torch.exp(logvar)
    kld_elem = 0.5 * (mu * mu + var - logvar - 1.0)
    # sum over channels & spatial dims, mean over batch
    kld = torch.mean(torch.sum(kld_elem, dim=[1,2,3,4]))
    return kld


def beta_vae_loss(recon, x, mu, logvar, beta: float = 4.0, use_ssim: bool = False, lambda_ssim: float = 0.5):
    """
    recon, x: (B, 1, D, H, W)
    mu, logvar: (B, C_z, D', H', W')
    """
    rec_mse = F.mse_loss(recon, x)
    kld = spatial_kld(mu, logvar)
    rec_ssim = 0.0
    if use_ssim:
        if not HAVE_SSIM:
            raise RuntimeError("pytorch-msssim not installed. pip install pytorch-msssim to use SSIM.")
        rec_ssim = 1.0 - ssim(recon, x, data_range=(2.0 if recon.min() < 0 else 1.0), size_average=True)

    # print(rec_mse.item(), kld.item())
    loss = rec_mse + lambda_ssim * rec_ssim + beta * kld
    metrics = {'mse': rec_mse.item(), 'kld': kld.item(), 'ssim': rec_ssim if isinstance(rec_ssim, float) else rec_ssim.item()}
    return loss, metrics

if __name__ == "__main__":
    
    decoders = []
    for i, name in enumerate(['800', '1100', '1300', '1500']):
        decoder = vae = BetaVAE3D(
            base_ch=4,
            latent_ch=16,
            num_down=5,
            use_attn=True,
            output_activation='tanh',
            target_size=(240,480,480)
        )
        decoder.load_state_dict(torch.load(f'/home/borntowarn/projects/chest-diseases/training/{name}.pt'))
        decoder.eval()
        decoder.to('cuda', dtype=torch.bfloat16)
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
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4)

    # model_base = ProjectionVIT()
    # model_base.load_state_dict(torch.load("/home/borntowarn/projects/chest-diseases/training/weights/CT-RATE/ProjectionVIT_Base_V2.pt"))
    # model_base.eval()
    # model_base.cuda()

    results_folder = defaultdict(list)

    with torch.no_grad():
        for batch in dataloader:
            ct_tensor, names = batch
            targets = ct_tensor.to('cuda', dtype=torch.bfloat16)
            for decoder in decoders:
                mu, logvar = decoder.encoder(targets)
                recon = decoder.decoder(mu)

                # print(ct_tensor.shape, recon.shape)
                recon_error = torch.mean((ct_tensor.squeeze(1) - recon.squeeze(1).cpu()) ** 2, dim=(1, 2, 3)).cpu().tolist()
                # print(recon_error)
                for name, error in zip(names, recon_error):
                    results_folder[name].append(error)
            if len(results_folder) > 100:
                break
            print(len(results_folder))
    
    import json
    with open('results.json', 'w') as f:
        json.dump(results_folder, f)