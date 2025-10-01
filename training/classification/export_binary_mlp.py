from src.ctrate_binary_model import LightningMLP
import torch
import hydra

@hydra.main(version_base=None, config_path='configs', config_name='config_binary_test.yaml')
def main(config):
    model = LightningMLP.load_from_checkpoint(
        '/home/free4ky/projects/chest-diseases/training/classification/runs/output/outputs_binary_ctrate_mosmed/2025-09-29/11-09-22/11/logs/lightning_logs/version_0/checkpoints/vit_mlp-epoch=13-val_f1=0.7607603073120117.ckpt',
        config=config,
        num_classes=2
    )
    torch.save(model.model.state_dict(), 'model_binary_2025-09-29_11-09-22_11.pth')
if __name__ == '__main__':
    main()
