import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.5,
        use_sigmoid: bool = True,
    ):
        super(BiLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True
        self.use_sigmoid = use_sigmoid
        self.num_directions = 2

        # ===== 双向 LSTM =====
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=self.bidirectional
        )

        # ===== 层归一化，更适合 RNN 时间序列 =====
        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)

        # ===== 全连接分类器 =====
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
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
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        h0 = torch.zeros(self.num_layers * self.num_directions, B, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, B, self.hidden_size).to(x.device)

        # LSTM 输出所有时间步
        out, _ = self.lstm(x, (h0, c0))  # out.shape: (B, T, 2H)
        out = out[:, -1, :]              # 取最后时间步

        # 层归一化
        out = self.layer_norm(out)

        # 全连接分类器 + 激活函数
        out = self.classifier(out)
        return self.activation(out)
