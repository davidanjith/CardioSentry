import torch.nn as nn
import torch



class ECGTransformerModel(nn.Module):
    def __init__(self, input_dim=3600, embed_dim=128, num_heads=4, num_layers=3, num_classes=3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 12, embed_dim) * 0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=256, dropout=0.3)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes))
        self._init_weights()

    def _init_weights(self):
        for param in self.transformer.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def extract_features(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        return x

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding
        x = self.transformer(x).mean(dim=1)
        return self.fc(x)

    # def get_gradients(self):
    #     return self.gradients
    #
    # def register_hooks(self):
    #     def hook_fn(module, grad_in, grad_out):
    #         self.gradients = grad_out[0]
    #
    #     self.transformer_encoder.layers[-1].register_backward_hook(hook_fn)