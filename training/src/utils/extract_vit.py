import torch
from src.modeling import ProjectionVIT, OriginalClassifierHead


def load_original_classifier_head(file_path):
    loaded_state_dict = torch.load(file_path)
    loaded_state_dict = {
        k: v for k, v in loaded_state_dict.items() if "trained_model" not in k
    }
    return loaded_state_dict


def load_original_projection_vit(file_path):
    loaded_state_dict = torch.load(file_path)
    state_dict = {
        k.replace("visual_transformer.", "VIT."): v
        for k, v in loaded_state_dict.items()
        if "visual_transformer" in k
    }
    state_dict["projection_layer.weight"] = loaded_state_dict["to_visual_latent.weight"]
    return state_dict


def load_original_projection_vit_classfine(file_path):
    loaded_state_dict = torch.load(file_path)
    state_dict = {
        k.replace("trained_model.visual_transformer.", "VIT."): v
        for k, v in loaded_state_dict.items()
        if "trained_model.visual_transformer" in k
    }
    state_dict["projection_layer.weight"] = loaded_state_dict[
        "trained_model.to_visual_latent.weight"
    ]
    return state_dict


def extract_vit_base(input_model, output_model):
    model = ProjectionVIT()
    model.load_state_dict(load_original_projection_vit(input_model))
    torch.save(model.state_dict(), output_model)


def extract_vit_vocabfine(input_model, output_model):
    model = ProjectionVIT()
    model.load_state_dict(load_original_projection_vit(input_model))
    torch.save(model.state_dict(), output_model)


def extract_vit_classfine(input_model, output_vit_model, output_head_model):
    model = ProjectionVIT()
    model.load_state_dict(load_original_projection_vit_classfine(input_model))
    torch.save(model.state_dict(), output_vit_model)

    classifier_head = OriginalClassifierHead()
    classifier_head.load_state_dict(load_original_classifier_head(input_model))
    torch.save(classifier_head.state_dict(), output_head_model)
