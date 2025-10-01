from src.ctrate_multilabel_model import LightningMLP
import torch
import hydra

# CHECKPOINT_PATH = '/home/free4ky/projects/chest-diseases/training/classification/runs/output/outputs_multibel/2025-09-26/13-30-30/4/logs/lightning_logs/version_0/checkpoints/vit_mlp-epoch=195-val_f1_macro=0.8323163986206055.ckpt'
# CHECKPOINT_PATH = '/home/free4ky/projects/chest-diseases/training/classification/runs/output/outputs_multibel/2025-09-26/13-07-45/0/logs/lightning_logs/version_0/checkpoints/vit_mlp-epoch=12-val_f1_macro=0.43588829040527344.ckpt'
CHECKPOINT_PATH = '/home/free4ky/projects/chest-diseases/training/classification/runs/output/outputs_multibel/2025-09-26/19-16-52/1/logs/lightning_logs/version_0/checkpoints/vit_mlp-epoch=49-val_f1_macro=0.45349034667015076.ckpt'


@hydra.main(version_base=None, config_path='configs', config_name='config_multilabel_test.yaml')
def main(config):
    model = LightningMLP.load_from_checkpoint(
        CHECKPOINT_PATH,
        config=config,
        num_classes=20
    )
    torch.save(model.model.state_dict(), 'model_multilabel20_pr034_rec0777.pth')
if __name__ == '__main__':
    main()
