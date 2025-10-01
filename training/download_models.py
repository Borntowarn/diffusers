from src.utils.extract_vit import (
    extract_vit_base,
    extract_vit_vocabfine,
    extract_vit_classfine,
)
from src.utils.extract_vista_encoder import extract_vista_encoder, download_vista_encoder
from src import CT_RATE_WEIGHTS_DIR, VISTA3D_WEIGHTS_DIR
from src.utils.downloading import download_CLIP

if __name__ == "__main__":
    models_to_download = [
        "CT-CLIP_v2.pt",
        "CT_LiPro_v2.pt",
        "CT_VocabFine_v2.pt",
    ]
    weight_pathes = download_CLIP(models_to_download)
    extract_vit_base(
        weight_pathes[0],
        CT_RATE_WEIGHTS_DIR / "ProjectionVIT_Base_V2.pt",
    )
    extract_vit_classfine(
        weight_pathes[1],
        CT_RATE_WEIGHTS_DIR / "ProjectionVIT_LiPro_V2.pt",
        CT_RATE_WEIGHTS_DIR / "ClassifierHead_LiPro_V2.pt",
    )
    extract_vit_vocabfine(
        weight_pathes[2],
        CT_RATE_WEIGHTS_DIR / "ProjectionVIT_VocabFine_V2.pt",
    )

    download_vista_encoder(VISTA3D_WEIGHTS_DIR / "model.pt")
    extract_vista_encoder(
        VISTA3D_WEIGHTS_DIR / "model.pt",
        VISTA3D_WEIGHTS_DIR / "image_encoder.pt",
    )
