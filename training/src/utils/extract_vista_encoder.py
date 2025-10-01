import torch
from src.modeling import VistaEncoder
import gdown


def load_original_vista_encoder(file_path):
    loaded_state_dict = torch.load(file_path)
    state_dict = {
        k.replace("image_encoder.", ""): v
        for k, v in loaded_state_dict.items()
        if "image_encoder.encoder" in k
    }
    return state_dict


def download_vista_encoder(output_model):
    # https://drive.google.com/file/d/1DRYA2-AI-UJ23W1VbjqHsnHENGi0ShUl/view
    gdown.download(id="1DRYA2-AI-UJ23W1VbjqHsnHENGi0ShUl", output=str(output_model), quiet=False)


def extract_vista_encoder(input_model, output_model):
    model = VistaEncoder()
    model.load_state_dict(load_original_vista_encoder(input_model))
    torch.save(model.state_dict(), output_model)