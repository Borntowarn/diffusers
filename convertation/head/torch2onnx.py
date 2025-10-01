import torch
import torch.nn as nn
from pathlib import Path

class MLP(nn.Module):
    def __init__(
            self,
            input_size=512,
            num_classes=18,
            activation='relu',
            hidden_sizes=[1024, 2048, 1024, 256, 128],
            dropout=0.1
        ):
        super().__init__()
        
        # Pick activation
        if activation == "relu":
            activation_cls = nn.ReLU
        elif activation == "leaky_relu":
            activation_cls = nn.LeakyReLU
        elif activation == "gelu":
            activation_cls = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = []
        in_dim = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))  # helps stabilize
            layers.append(activation_cls())
            layers.append(nn.Dropout(dropout))
            in_dim = h

        # Final classification layer
        layers.append(nn.Linear(in_dim, num_classes))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

@torch.no_grad()
def export_onnx(model, x, name):
    dim = torch.export.Dim('batch_size')
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
    onnx_model.save(name)

CT_RATE_WEIGHTS_DIR = Path("../../training/weights/CT-RATE")

if __name__ == "__main__":
    with torch.no_grad():
        multilabel_model = MLP(
            input_size=512,
            activation='leaky_relu',
            dropout=0.2,
            num_classes=20,
            hidden_sizes=[512, 256, 128],
        )
        multilabel_model.load_state_dict(torch.load(CT_RATE_WEIGHTS_DIR / "model_multilabel.pth"))
        multilabel_model.eval().cuda()
        export_onnx(multilabel_model, torch.randn(2, 512), "multilabel_model.onnx")


        binary_model = MLP(
            input_size=512,
            activation='gelu',
            dropout=0.2,
            hidden_sizes=[256, 128],
            num_classes=2
        )
        binary_model.load_state_dict(torch.load(CT_RATE_WEIGHTS_DIR / "model_binary.pth"))
        binary_model.eval().cuda()
        export_onnx(binary_model, torch.randn(2, 512), "binary_model.onnx")