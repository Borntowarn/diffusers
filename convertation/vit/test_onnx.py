from transformer_maskgit.ctvit import CTViT
import torch
import onnxruntime
import torch.nn as nn
import numpy as np
from pathlib import Path

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
    model = ProjectionVIT()
    model.load_state_dict(torch.load(CT_RATE_WEIGHTS_DIR / "ProjectionVIT_LiPro_V2.pt"), strict=False)
    model.eval().cuda()

    onnx_model = onnxruntime.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])

    with torch.no_grad():
        import time
        start_time = time.time()
        for i in range(10):
            x = torch.randn(1, 1, 240, 480, 480)
            x = x.cuda()
            torch_result = model(x)
            torch_result = torch_result.cpu()
        end_time = time.time()
        print((end_time - start_time)/10)

        start_time = time.time()
        for i in range(10):
            onnx_result = onnx_model.run(None, {"input": torch.randn(1, 1, 240, 480, 480).numpy()})
        end_time = time.time()
        print((end_time - start_time)/10)

        print(torch_result.shape)
        print(torch.from_numpy(onnx_result[0]).shape)

        print(torch_result)
        print(torch.from_numpy(onnx_result[0]))

        print(torch.allclose(torch_result, torch.from_numpy(onnx_result[0])))
