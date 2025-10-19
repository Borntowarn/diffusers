import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from torch.utils.tensorboard import SummaryWriter
import math
import os

# ---- import your BetaVAE ----
from src.vae import BetaVAE  # save the earlier model code in vae_model.py


# -------------------------------
# ⚙️ CONFIG
# -------------------------------
config = {
    "input_dim": 512,
    "latent_dim": 4,
    "beta": 16,
    "batch_size": 128,
    "epochs": 100,
    "lr": 3e-5,
    "warmup_steps": 50,
    "log_dir": "./runs/logs/logs_vae115",
    "output_dir": "./runs/checkpoints/vae_checkpoints15",
    "mixed_precision": None
}

def _load_concat_dataset(embed_paths):
    X_list = [torch.load(p) for p in embed_paths]
    X = torch.cat(X_list, dim=0)
    return TensorDataset(X)

# -------------------------------
# 🧠 DATASET (replace with your embeddings)
# -------------------------------


# train_dataset = TensorDataset(
#     torch.load('data/vae_x.pt')
# )
# val_dataset = TensorDataset(
#     torch.load('data/validation_data.pt')
# )

train_dataset = _load_concat_dataset(
    ['data/vae_x.pt','data/vae_x_mosmed.pt'],
)

val_dataset = _load_concat_dataset(
    ['data/validation_data.pt','data/test_data.pt'],
)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)


# -------------------------------
# 🚀 INIT ACCELERATOR
# -------------------------------
accelerator = Accelerator(mixed_precision=config['mixed_precision'])  # can be "fp16" or "no" as needed
device = accelerator.device


# -------------------------------
# 🧩 MODEL, OPTIMIZER, SCHEDULER
# -------------------------------
vae = BetaVAE(
    input_dim=config["input_dim"],
    latent_dim=config["latent_dim"],
    hidden_dims=[256,128],
    beta=config["beta"]
)

optimizer = torch.optim.AdamW(vae.parameters(), lr=config["lr"])
lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=config["warmup_steps"],
    num_training_steps=config["epochs"] * math.ceil(len(train_loader))
)

vae, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
    vae, optimizer, train_loader, val_loader, lr_scheduler
)


# -------------------------------
# 🧾 TENSORBOARD LOGGER
# -------------------------------
if accelerator.is_main_process:
    os.makedirs(config["log_dir"], exist_ok=True)
    writer = SummaryWriter(log_dir=config["log_dir"])
else:
    writer = None


# -------------------------------
# 🧮 TRAINING LOOP
# -------------------------------
global_step = 0

for epoch in range(config["epochs"]):
    vae.train()
    train_loss, train_recon, train_kl = 0.0, 0.0, 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}", disable=not accelerator.is_local_main_process)

    for batch in pbar:
        (x,) = batch
        recon, mu, logvar = vae(x)
        loss, recon_loss, kl_loss = vae.loss_function(recon, x, mu, logvar)

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()

        train_loss += loss.item()
        train_recon += recon_loss.item()
        train_kl += kl_loss.item()

        if writer and global_step % 50 == 0:
            writer.add_scalar("train/total_loss", loss.item(), global_step)
            writer.add_scalar("train/recon_loss", recon_loss.item(), global_step)
            writer.add_scalar("train/kl_loss", kl_loss.item(), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "recon": f"{recon_loss.item():.4f}",
            "kl": f"{kl_loss.item():.4f}"
        })
        global_step += 1

    # -------------------------------
    # 🔍 VALIDATION
    # -------------------------------
    vae.eval()
    val_loss, val_recon, val_kl = 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch in val_loader:
            (x,) = batch
            recon, mu, logvar = vae(x)
            loss, recon_loss, kl_loss = vae.loss_function(recon, x, mu, logvar)
            val_loss += loss.item()
            val_recon += recon_loss.item()
            val_kl += kl_loss.item()

    val_loss /= len(val_loader)
    val_recon /= len(val_loader)
    val_kl /= len(val_loader)

    accelerator.print(
        f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f} | Val Loss={val_loss:.4f}"
    )

    if writer:
        writer.add_scalar("val/total_loss", val_loss, epoch)
        writer.add_scalar("val/recon_loss", val_recon, epoch)
        writer.add_scalar("val/kl_loss", val_kl, epoch)

    # -------------------------------
    # 💾 SAVE CHECKPOINT
    # -------------------------------
    if accelerator.is_main_process:
        os.makedirs(config["output_dir"], exist_ok=True)
        torch.save(vae.state_dict(), f"{config['output_dir']}/vae_epoch_{epoch+1}.pth")

if writer:
    writer.close()
