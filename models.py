import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Gated 融合模块
class GatedFusion(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.gates = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, 1),
                nn.Sigmoid()
            )
            for name, dim in input_dims.items()
        })

    def forward(self, feature_dict):
        gated_features = []
        for name, feat in feature_dict.items():
            gate = self.gates[name](feat)
            gated = gate * feat
            gated_features.append(gated)
        return torch.cat(gated_features, dim=1)

# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        out = avg_out + max_out
        return self.sigmoid(out).unsqueeze(-1) * x

# 残差卷积块
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=15):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.ca = ChannelAttention(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        x = self.bn2(self.conv2(x))
        x = self.ca(x)
        return F.leaky_relu(x + residual, 0.1)

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, 1, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].permute(1, 0, 2)
        return self.dropout(x)

# 各模态子网络
class SparseModalityNet(nn.Module):
    def __init__(self, in_channels, feature_dim=32, is_xgb=False):
        super().__init__()
        self.is_xgb = is_xgb

        if is_xgb:
            self.projection = nn.Sequential(
                nn.Conv1d(in_channels, 16, kernel_size=1),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(0.1),
                nn.AdaptiveAvgPool1d(1)
            )
            self.fc = nn.Linear(16, feature_dim)
        else:
            self.feature_extractor = nn.Sequential(
                ResBlock(in_channels, 16),
                nn.MaxPool1d(2),
                ResBlock(16, 32)
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=32,
                nhead=2,
                dim_feedforward=128,
                dropout=0.5,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.pos_encoder = PositionalEncoding(32, dropout=0.5)

    def forward(self, x):
        if self.is_xgb:
            x = self.projection(x).squeeze(-1)
            return self.fc(x)
        else:
            x = self.feature_extractor(x)
            x = x.permute(0, 2, 1) # 形状变为 (batch, seq_len, 32)
            x = self.pos_encoder(x)
            x = self.transformer(x)
            return x.mean(dim=1) 

# 主模型
class DeepMultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_net = nn.Sequential(
            ResBlock(4, 32, kernel_size=15),
            nn.MaxPool1d(3),
            ResBlock(32, 64),
            nn.MaxPool1d(2),
            ResBlock(64, 128),
            nn.AdaptiveMaxPool1d(1)
        )

        self.snp_net = SparseModalityNet(1)
        self.variant_net = SparseModalityNet(2)
        self.m6a_net = SparseModalityNet(32, is_xgb=True)        
        self.DNA6mA_net = SparseModalityNet(32, is_xgb=True)
        self.methylation_net = SparseModalityNet(32, is_xgb=True)
        self.expression_net = SparseModalityNet(32, is_xgb=True)

        self.gated_fusion = GatedFusion({
            'seq': 128,
            'snp': 32,
            'variant': 32,
            'm6a': 32,
            'methylation': 32,
            'expression': 32,
            'DNA6mA': 32
        })

        self.classifier = nn.Sequential(
            nn.Linear(320, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.6),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        features = {
            'seq': self.seq_net(x['data']['seq'] * x['mask']['seq']).squeeze(-1),
            'snp': self.snp_net(x['data']['snp'] * x['mask']['snp']),
            'variant': self.variant_net(x['data']['variant'] * x['mask']['variant']),
            'm6a': self.m6a_net(x['data']['m6a'] * x['mask']['m6a']),
            'methylation': self.methylation_net(x['data']['methylation'] * x['mask']['methylation']),
            'expression': self.expression_net(x['data']['expression']),
            'DNA6mA': self.DNA6mA_net(x['data']['DNA6mA'] * x['mask']['DNA6mA'])
        }
        merged = self.gated_fusion(features)
        return self.classifier(merged)
