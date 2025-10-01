from torch import nn

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
        if len(hidden_sizes) == 0:
            layers.append(activation_cls())
            layers.append(nn.Dropout(dropout))
            
        layers.append(nn.Linear(in_dim, num_classes))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)