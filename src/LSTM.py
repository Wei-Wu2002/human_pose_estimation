import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 output_size: int, dropout: float = 0.5, use_sigmoid: bool = True):
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

        # 对 pooled feature 做归一化
        self.bn = nn.BatchNorm1d(hidden_size)

        # Frame-level classifier (B, T, H) → (B, T, output_size)
        self.frame_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

        # Clip-level classifier (mean pooled) (B, H) → (B, output_size)
        self.clip_classifier = nn.Sequential(
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
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.01)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: shape (B, T, D)
            return_all: 是否输出每帧分类结果
        Returns:
            clip_out: shape (B, output_size)
            frame_outs (optional): shape (B, T, output_size)
        """
        B = x.size(0)
        h0 = torch.zeros(self.num_layers, B, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, B, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))  # out: (B, T, H)

        # ----- Frame-level classification -----
        frame_outs = self.frame_classifier(out)             # (B, T, output_size)
        frame_outs = self.activation(frame_outs)

        # ----- Clip-level classification (mean over T) -----
        last_step = out[:, -1, :]  # (B, H)
        last_step = self.bn(last_step)
        clip_out = self.clip_classifier(last_step)             # (B, output_size)
        clip_out = self.activation(clip_out)

        return clip_out, frame_outs