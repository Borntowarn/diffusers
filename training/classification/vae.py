import torch
import torch.nn as nn
import torch.nn.functional as F

class BetaVAE(nn.Module):
    def __init__(self, input_dim=512, latent_dim=32, hidden_dims=(256, 128), beta=4.0):
        """
        Beta-VAE for vector inputs.

        Args:
            input_dim (int): Dimension of input vector.
            latent_dim (int): Dimension of latent space.
            hidden_dims (tuple): Hidden layer sizes for encoder and decoder.
            beta (float): Weight for KL divergence term (β > 1 encourages disentanglement).
        """
        super(BetaVAE, self).__init__()
        self.beta = beta

        # --- Encoder ---
        encoder_layers = []
        last_dim = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(last_dim, h))
            encoder_layers.append(nn.BatchNorm1d(h))
            encoder_layers.append(nn.ReLU())
            # encoder_layers.append(nn.Dropout(0.2))
            last_dim = h
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(last_dim, latent_dim)
        self.fc_logvar = nn.Linear(last_dim, latent_dim)

        # --- Decoder ---
        decoder_layers = []
        last_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(last_dim, h))
            decoder_layers.append(nn.BatchNorm1d(h))
            decoder_layers.append(nn.ReLU())
            # decoder_layers.append(nn.Dropout(0.2))
            last_dim = h
        decoder_layers.append(nn.Linear(last_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        x = torch.nn.functional.normalize(x, dim=-1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        """
        Computes Beta-VAE loss = reconstruction + beta * KL divergence.
        """
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_div, recon_loss, kl_div

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# def get_activation(name: str):
#     name = name.lower()
#     if name == "relu":
#         return nn.ReLU()
#     elif name == "elu":
#         return nn.ELU()
#     elif name == "leaky_relu":
#         return nn.LeakyReLU(0.2)
#     elif name == "gelu":
#         return nn.GELU()
#     else:
#         raise ValueError(f"Unknown activation {name}")


# class BetaVAE(nn.Module):
#     def __init__(self, input_dim=512, latent_dim=32, hidden_dims=(256, 128),
#                  beta=4.0, use_bn=False, activation="relu", dropout=0.2, recon_loss="mse"):
#         super().__init__()
#         self.beta = beta
#         self.use_bn = use_bn
#         self.recon_loss = recon_loss

#         # ----- Encoder -----
#         enc_layers = []
#         last = input_dim
#         for h in hidden_dims:
#             enc_layers.append(nn.Linear(last, h))
#             if use_bn:
#                 enc_layers.append(nn.BatchNorm1d(h))
#             enc_layers.append(get_activation(activation))
#             enc_layers.append(nn.Dropout(dropout))
#             last = h
#         self.encoder = nn.Sequential(*enc_layers)

#         self.fc_mu = nn.Linear(last, latent_dim)
#         self.fc_logvar = nn.Linear(last, latent_dim)

#         # ----- Decoder -----
#         dec_layers = []
#         last = latent_dim
#         for h in reversed(hidden_dims):
#             dec_layers.append(nn.Linear(last, h))
#             if use_bn:
#                 dec_layers.append(nn.BatchNorm1d(h))
#             dec_layers.append(get_activation(activation))
#             dec_layers.append(nn.Dropout(dropout))
#             last = h
#         dec_layers.append(nn.Linear(last, input_dim))
#         self.decoder = nn.Sequential(*dec_layers)

#     # -----------------
#     def encode(self, x):
#         h = self.encoder(x)
#         mu, logvar = self.fc_mu(h), self.fc_logvar(h)
#         return mu, logvar

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def decode(self, z):
#         return self.decoder(z)

#     def forward(self, x):
#         x = F.normalize(x, dim=-1)
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         recon = self.decode(z)
#         return recon, mu, logvar

#     def loss_function(self, recon_x, x, mu, logvar):
#         if self.recon_loss == "mse":
#             recon_loss = F.mse_loss(recon_x, x, reduction="mean")
#         elif self.recon_loss == "l1":
#             recon_loss = F.l1_loss(recon_x, x, reduction="mean")
#         kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#         return recon_loss + self.beta * kl, recon_loss, kl