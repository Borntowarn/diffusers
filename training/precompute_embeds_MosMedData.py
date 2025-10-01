import os
import traceback
from pathlib import Path

import torch
from src import CT_RATE_WEIGHTS_DIR, logger
from src.data import MosMedDataCachingDataset
from src.modeling import ProjectionVIT
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data import examination_to_tensor_updated

logger.add("precompute_embeds_MosMedData.log")


@torch.no_grad()
def precompute_embeds(data_root, folder2model, dataloader):
    for data in tqdm(dataloader):
        try:
            tensor, local_file, name = data
            if isinstance(tensor, list):
                logger.success(f"tensor for {str(local_file[0])} is exists")
                continue

            local_file = Path(local_file[0])
            for folder, model in folder2model.items():
                embed = model(tensor.cuda())

                rel_path = local_file.relative_to(data_root)
                save_path = data_root / "embeddings" / folder / rel_path.with_suffix(rel_path.suffix + ".pt")
                save_path.parent.mkdir(parents=True, exist_ok=True)

                logger.info(f"Saving {name[0]} to {save_path}")
                torch.save(embed.cpu().detach()[0], save_path)

        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(e)
            logger.error(local_file)
            continue


if __name__ == "__main__":
    model1 = ProjectionVIT()
    model1.load_state_dict(
        torch.load(CT_RATE_WEIGHTS_DIR / "ProjectionVIT_LiPro_V2.pt")
    )
    model1.eval()
    model1.cuda()
    logger.info("Model 1 loaded")

    model2 = ProjectionVIT()
    model2.load_state_dict(torch.load(CT_RATE_WEIGHTS_DIR / "ProjectionVIT_Base_V2.pt"))
    model2.eval()
    model2.cuda()
    logger.info("Model 2 loaded")

    folder2model = {
        "embeds_not_normalized_lipro": model1,
        "embeds_not_normalized_base": model2,
    }

    data_root = Path("/home/borntowarn/projects/chest-diseases/training/data/MosMedData")
    dataset = MosMedDataCachingDataset(
        root_folder=data_root,
        folders=[
            "COVID19_1110",
            'CT_LUNGCANCER_500',
            'MosMedData-CT-COVID19-type I-v 4',
            'MosMedData-CT-COVID19-type VII-v 1',
            'MosMedData-LDCT-LUNGCR-type I-v 1'
        ],

        saving_folders=list(folder2model.keys())
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)
    precompute_embeds(data_root, folder2model, dataloader)
