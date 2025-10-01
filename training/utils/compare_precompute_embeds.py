import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os

import pandas as pd
import torch
from src import CT_RATE_DIR, CT_RATE_WEIGHTS_DIR
from src.data import examination_to_tensor
from src.modeling import ProjectionVIT
from src.utils.downloading import download_data
from transformers import BertTokenizer, BertModel
from transformer_maskgit import CTViT
from ct_clip import CTCLIP
import torch.nn as nn

class ImageLatentsClassifier(nn.Module):
    def __init__(self, trained_model, latent_dim, num_classes, dropout_prob=0.3):
        super(ImageLatentsClassifier, self).__init__()
        self.trained_model = trained_model
        self.dropout = nn.Dropout(dropout_prob)  # Add dropout layer
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(latent_dim, num_classes)  # Assuming trained_model.image_latents_dim gives the size of the image_latents

    def forward(self, latents=False, *args, **kwargs):
        kwargs['return_latents'] = True
        _, image_latents, _ = self.trained_model(*args, **kwargs)
        image_latents = self.relu(image_latents)
        if latents:
            return image_latents
        image_latents = self.dropout(image_latents)  # Apply dropout on the latents

        return self.classifier(image_latents)
    
    def get_image_latents(self, *args, **kwargs):
        kwargs['return_latents'] = True
        _, image_latents, _ = self.trained_model(*args, **kwargs)
        return image_latents
    
    def get_image_encodings(self, *args, **kwargs):
        kwargs['return_encodings'] = True
        _, image_encodings = self.trained_model(*args, **kwargs)
        return image_encodings

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)
    def load(self, file_path):
        loaded_state_dict = torch.load(file_path)
        self.load_state_dict(loaded_state_dict)

if __name__ == "__main__":
    model_path = CT_RATE_WEIGHTS_DIR / "ProjectionVIT_LiPro_V2.pt"
    meta_path = CT_RATE_DIR / "dataset" / "metadata" / "train_metadata.csv"

    model1 = ProjectionVIT()
    model1.load_state_dict(torch.load(model_path))
    model1.eval()
    model1.cuda()

    tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

    text_encoder.resize_token_embeddings(len(tokenizer))

    image_encoder = CTViT(
        dim = 512,
        codebook_size = 8192,
        image_size = 480,
        patch_size = 20,
        temporal_patch_size = 10,
        spatial_depth = 4,
        temporal_depth = 4,
        dim_head = 32,
        heads = 8
    )

    clip = CTCLIP(
        image_encoder = image_encoder,
        text_encoder = text_encoder,
        dim_image = 294912,
        dim_text = 768,
        dim_latent = 512,
        extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
        use_mlm=False,
        downsample_image_embeds = False,
        use_all_token_embeds = False

    )
    classifier = ImageLatentsClassifier(clip, 512, 18)
    classifier.load('/home/borntowarn/projects/chest-diseases/training/weights/CT-RATE/models/CT-CLIP-Related/CT_LiPro_v2.pt')
    classifier.eval()
    classifier.cuda()
    device = torch.device("cuda")
    text_tokens = tokenizer("", return_tensors="pt", padding="max_length", truncation=True, max_length=200).to(device)

    df = pd.read_csv(meta_path)

    test_filenames = [
        "train_1_a_1.nii.gz",
        # "train_1000_a_1.nii.gz",
        # "train_2000_a_1.nii.gz",
        # "train_3000_a_1.nii.gz",
        # "train_4000_a_1.nii.gz",
        # "train_5000_a_1.nii.gz",
        # "train_6000_a_1.nii.gz",
        # "train_7000_a_1.nii.gz",
        # "train_8000_a_1.nii.gz",
        # "train_9000_a_1.nii.gz",
        # "train_10000_a_1.nii.gz",
        # "train_11000_a_1.nii.gz",
        # "train_12000_a_1.nii.gz",
    ]
    all_close = []
    for filename in test_filenames:
        data_path = download_data("train_fixed", filenames=[filename])[0]
        tensor = examination_to_tensor(data_path)
        result1 = classifier.get_image_encodings(text_tokens, tensor.cuda().unsqueeze(0), device=device, return_latents=True).cpu()

        rel_path = data_path.relative_to(CT_RATE_DIR / "dataset" / "train_fixed")
        result2 = torch.load(
            CT_RATE_DIR
            / "dataset"
            / "train_fixed_embeds_not_normalized_lipro_new"
            / rel_path.with_suffix(".pt")
        ).view(1, -1)
        all_close.append(torch.allclose(result1, result2))
        os.remove(data_path)

    print(all_close)
    print(len(all_close))
    print(all_close.count(True))
    print(all_close.count(False))