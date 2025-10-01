import torch
import torch.nn as nn
from pathlib import Path
from transformer_maskgit.ctvit import CTViT

class ProjectionVIT(nn.Module):
    def __init__(self):
        super(ProjectionVIT, self).__init__()
        self.VIT = CTViT(
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
        self.projection_layer = nn.Linear(294912, 512, bias = False)

    def forward(self, x):
        x = self.VIT(x, return_encoded_tokens=True)

        x = torch.mean(x, dim=1)
        x = x.view(x.size(0), -1)

        x = self.projection_layer(x)
        return x

CT_RATE_WEIGHTS_DIR = Path("/home/borntowarn/projects/chest-diseases/training/weights/CT-RATE")

if __name__ == "__main__":
    with torch.no_grad():
        model = ProjectionVIT()
        model.load_state_dict(torch.load(CT_RATE_WEIGHTS_DIR / "ProjectionVIT_LiPro_V2.pt"), strict=False)
        model.eval().cuda()

        x = torch.randn(2, 1, 240, 480, 480)

        dim = torch.export.Dim('batch_size')

        model(x.cuda())
        onnx_model = torch.onnx.export(
            model,
            x.cuda(),
            input_names=["input"],
            output_names=["output"],
            dynamic_shapes={"input": {0: dim}, "output": {0: dim}},
            verbose=True,
            dynamo=True,
            external_data=False
        )
        onnx_model.optimize()
        onnx_model.save("vit_lipro.onnx")