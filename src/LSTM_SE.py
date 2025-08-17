import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSE(nn.Module):
    """
    对时间步进行 SE 加权：对每个时间步的特征向量 h_t 产生一个标量 gate a_t ∈ (0,1)
    输入:  H, shape = (B, T, C)
    输出:  H_weighted = H * a, 其中 a 的 shape = (B, T, 1)
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, 1)

        # 初始化（可选）
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        # H: (B, T, C)
        # 对每个时间步独立地产生一个 gate
        g = F.relu(self.fc1(H))          # (B, T, C//r)
        g = torch.sigmoid(self.fc2(g))   # (B, T, 1)
        return H * g                      # (B, T, C)


class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 output_size: int, dropout: float = 0.5, use_sigmoid: bool = True,
                 se_reduction: int = 8):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_sigmoid = use_sigmoid

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 时间维度的 SE 模块：为每个时间步打分并加权
        self.temporal_se = TemporalSE(channels=hidden_size, reduction=se_reduction)

        # 对加权后的序列做聚合（这里用加权求和；也可以改为 mean 或者 learnable pooling）
        self.bn = nn.BatchNorm1d(hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, output_size)
        )

        self.activation = nn.Sigmoid() if use_sigmoid else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                # classifier 的线性层权重已覆盖
                pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, input_size)
        """
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

        H, _ = self.lstm(x, (h0, c0))     # (B, T, C)

        # 对时间步做 SE 加权
        Hw = self.temporal_se(H)          # (B, T, C)

        # 聚合：加权求和（若想尺度更稳可用平均：context = Hw.mean(dim=1)）
        context = Hw.sum(dim=1)           # (B, C)

        # 归一化 + 分类头
        out = self.bn(context)
        out = self.classifier(out)
        return self.activation(out)
