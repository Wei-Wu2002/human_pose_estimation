import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            output_size: int,
            dropout: float = 0.5,
            use_sigmoid: bool = True  # 允许后续切换为 logits 输出
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_sigmoid = use_sigmoid

        # GRU 层：支持多层堆叠
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 归一化：作用在最后时间步的输出向量上
        self.bn = nn.BatchNorm1d(hidden_size)

        # 三层全连接网络
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, output_size)
        )

        # 激活函数（仅用于二分类时）
        self.activation = nn.Sigmoid() if use_sigmoid else nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        """对 GRU 和 Linear 层的权重进行初始化"""
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：x.shape = (batch_size, seq_len, input_size)
        输出：y.shape = (batch_size, output_size)，通常为概率值
        """
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

        out, _ = self.gru(x, h0)  # GRU 输出：所有时间步
        out = out[:, -1, :]  # 仅取最后时间步的特征
        out = self.bn(out)  # 批归一化
        out = self.classifier(out)  # 全连接网络
        out = self.activation(out)  # Sigmoid 或 Identity（Logits 输出）
        return out
