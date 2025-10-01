import torch
import torch.nn as nn
from transformer_maskgit import CTViT
from src.vista3d.modeling.segresnetds import SegResEncoder
import torch.nn.functional as F


class ProjectionVIT(nn.Module):
    def __init__(self):
        super(ProjectionVIT, self).__init__()
        self.VIT = CTViT(
            dim=512,
            codebook_size=8192,
            image_size=480,
            patch_size=20,
            temporal_patch_size=10,
            spatial_depth=4,
            temporal_depth=4,
            dim_head=32,
            heads=8,
        )
        self.projection_layer = nn.Linear(294912, 512, bias=False)

    def forward(self, x):
        x = self.VIT(x, return_encoded_tokens=True)

        x = torch.mean(x, dim=1)
        x = x.view(x.size(0), -1)

        x = self.projection_layer(x)
        return x


class OriginalClassifierHead(nn.Module):
    def __init__(self, latent_dim=512, num_classes=18):
        super(OriginalClassifierHead, self).__init__()
        self.classifier = nn.Linear(latent_dim, num_classes)


class VistaEncoder(nn.Module):
    def __init__(self, patch_size=64):
        super(VistaEncoder, self).__init__()
        self.encoder = SegResEncoder(
            spatial_dims=3,
            init_filters=48,
            in_channels=1,
            act=("relu", {"inplace": True}),
            norm=("instance", {"affine": True}),
            blocks_down=(1, 2, 2, 4, 4),
            anisotropic_scales=None,
        )
        self.patch_size = patch_size

    # Работает только при batch_size = 1, при большем batch_size на последних слоях выдает другой результат из-за InstanceNorm
    def forward(self, input_tensor):
        b, c, d, h, w = input_tensor.shape
        processed_patches = []
        for z in range(0, d, self.patch_size):
            for y in range(0, h, self.patch_size):
                for x in range(0, w, self.patch_size):
                    patch = input_tensor[
                        :, :, 
                        z:min(z + self.patch_size, d),
                        y:min(y + self.patch_size, h),
                        x:min(x + self.patch_size, w)
                    ]

                    dz, dy, dx = patch.shape[2:]

                    # если все измерения < 32 → паддим первую ось (d) до 32
                    if dz < 32 and dy < 32 and dx < 32:
                        pad_d = 32 - dz
                        patch = torch.nn.functional.pad(patch, (0, 0, 0, 0, 0, pad_d))  # паддим только по глубине

                    x_encoded = self.encoder(patch)
                    processed_patches.append(
                        torch.cat([torch.mean(j, dim=(2, 3, 4)).cpu() for j in x_encoded], dim=1)
                    )

        return torch.mean(torch.cat(processed_patches, dim=0), dim=0, keepdim=True)

class MLP(nn.Module):
    def __init__(
            self,
            input_size=512,
            num_classes=18,
            activation='relu',
            hidden_sizes=[1024, 2048, 1024, 256, 128],
            dropout=0.1
        ):
        super().__init__()
        
        # Pick activation
        if activation == "relu":
            activation_cls = nn.ReLU
        elif activation == "leaky_relu":
            activation_cls = nn.LeakyReLU
        elif activation == "gelu":
            activation_cls = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = []
        in_dim = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))  # helps stabilize
            layers.append(activation_cls())
            layers.append(nn.Dropout(dropout))
            in_dim = h

        # Final classification layer
        layers.append(nn.Linear(in_dim, num_classes))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)