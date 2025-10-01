import os
import traceback
from pathlib import Path

import torch
from src import CT_RATE_DIR, CT_RATE_WEIGHTS_DIR, VISTA3D_WEIGHTS_DIR, logger
from src.data import CTRATECachingDataset
from src.modeling import ProjectionVIT, VistaEncoder
from src.utils.downloading import (
    download_anatomy_segmentation_labels,
    download_metadata,
    download_multi_abnormality_labels,
    download_radiology_text_reports,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

logger.add("precompute_embeds.log")


@torch.no_grad()
def precompute_embeds(folder2model, dataloader, downloaded_folder):
    for data in tqdm(dataloader):
        try:
            tensor, tensor_vista3d, local_file, name = data
            if isinstance(tensor, list):
                logger.success(f"tensor for {str(local_file[0])} is exists")
                continue

            local_file = Path(local_file[0])
            rel_path = local_file.relative_to(
                CT_RATE_DIR / "dataset" / downloaded_folder
            )
            for folder, model in folder2model.items():
                if isinstance(model, VistaEncoder):
                    embed = model(tensor_vista3d.cuda())
                else:
                    embed = model(tensor.cuda())

                save_path = CT_RATE_DIR / "dataset" / folder / rel_path
                save_path.parent.mkdir(parents=True, exist_ok=True)

                logger.info(f"Saving {name[0]} to {save_path}")
                pt_path = save_path.with_suffix(".pt")
                torch.save(embed.cpu().detach()[0], pt_path)

            # os.remove(local_file)
            logger.info(f"Deleted {local_file}")
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(e)
            logger.error(local_file)
            continue


if __name__ == "__main__":
    download_metadata()
    download_anatomy_segmentation_labels()
    download_multi_abnormality_labels()
    download_radiology_text_reports()

    model_lipro = ProjectionVIT()
    model_lipro.load_state_dict(
        torch.load(CT_RATE_WEIGHTS_DIR / "ProjectionVIT_LiPro_V2.pt")
    )
    model_lipro.eval()
    model_lipro.cuda()
    logger.info("Model 1 loaded")

    model_base = ProjectionVIT()
    model_base.load_state_dict(torch.load(CT_RATE_WEIGHTS_DIR / "ProjectionVIT_Base_V2.pt"))
    model_base.eval()
    model_base.cuda()
    logger.info("Model 2 loaded")

    model_vista3d = VistaEncoder()
    model_vista3d.load_state_dict(torch.load(VISTA3D_WEIGHTS_DIR / "image_encoder.pt"))
    model_vista3d.eval()
    model_vista3d.cuda()
    logger.info("Model 3 loaded")

    downloaded_folder = "valid_fixed"
    folder2model = {
        f"{downloaded_folder}_embeds_not_normalized_lipro_new": model_lipro,
        f"{downloaded_folder}_embeds_not_normalized_base_new": model_base,
        f"{downloaded_folder}_embeds_not_normalized_vista3d_new": model_vista3d,
    }

    dataset = CTRATECachingDataset(
        list(range(20000)), downloaded_folder, list(folder2model.keys())
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    precompute_embeds(folder2model, dataloader, downloaded_folder)
