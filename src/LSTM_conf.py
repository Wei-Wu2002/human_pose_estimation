import torch
import torch.nn as nn

class ConfidenceGatedLSTM(nn.Module):
    """
    仅做逐时间步 (x,y) 门控:
      - 输入 (B,T,51): 每帧 17×[x,y,conf] 展平 -> 用 conf^gamma 对 (x,y) 逐点门控
      - 输入 (B,T,34): 仅 (x,y)，若 expect_xyconf=False 则不做门控
    不使用时间注意力；时间维度汇聚仍取最后一步。
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.5,
        use_sigmoid: bool = True,
        expect_xyconf: bool = True,   # True 表示输入含 conf (51 维)
        conf_gamma: float = 1.0,      # 门控强度: 使用 conf**gamma
        conf_threshold: float = 0.0,  # 低于该阈值的 conf 会被截断到阈值
        keep_conf_in_features: bool = True,  # 门控后是否把 conf 拼回送入 LSTM
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_sigmoid = use_sigmoid
        self.expect_xyconf = expect_xyconf
        self.conf_gamma = conf_gamma
        self.conf_threshold = conf_threshold
        self.keep_conf_in_features = keep_conf_in_features

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
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
        for name, p in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def _gate_xy_with_conf(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T,51) 或 (B,T,34)
        返回: 与输入同形状 (仅对含 conf 的情况做 (x,y) 门控)
        """
        B, T, D = x.shape
        if self.expect_xyconf:
            # 需要 51 维
            if D != 51:
                raise ValueError(f"expect_xyconf=True 要求输入维度 51, 但得到 {D}")
            seq = x.view(B, T, 17, 3)     # [x,y,conf]
            xy   = seq[..., :2]           # (B,T,17,2)
            conf = seq[..., 2]            # (B,T,17)

            # 置信度截断 + 幂次门控
            conf_eff = conf.clamp(min=self.conf_threshold) if self.conf_threshold > 0 else conf
            if self.conf_gamma != 1.0:
                conf_eff = conf_eff.pow(self.conf_gamma)

            xy_gated = xy * conf_eff.unsqueeze(-1)  # (B,T,17,2)

            if self.keep_conf_in_features:
                seq_gated = torch.cat([xy_gated, conf_eff.unsqueeze(-1)], dim=-1)  # (B,T,17,3)
                return seq_gated.view(B, T, 51)
            else:
                return xy_gated.view(B, T, 34)
        else:
            # 仅 (x,y)，不做门控，直接返回
            if D != 34:
                raise ValueError(f"expect_xyconf=False 要求输入维度 34, 但得到 {D}")
            return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T,51) 若 expect_xyconf=True；或 (B,T,34) 若 expect_xyconf=False
        """
        B = x.size(0)

        feats = self._gate_xy_with_conf(x)  # 仅逐时间步门控

        # 若门控后维度与 LSTM 声明不一致，做一次线性投影兜底
        if feats.size(-1) != self.lstm.input_size:
            proj = getattr(self, "_proj_in", None)
            if proj is None or proj.in_features != feats.size(-1) or proj.out_features != self.lstm.input_size:
                self._proj_in = nn.Linear(feats.size(-1), self.lstm.input_size).to(feats.device)
                nn.init.xavier_uniform_(self._proj_in.weight); nn.init.constant_(self._proj_in.bias, 0.0)
            feats = self._proj_in(feats)

        h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, B, self.hidden_size, device=x.device)
        out_seq, _ = self.lstm(feats, (h0, c0))   # (B,T,H)

        out = out_seq[:, -1, :]                   # 仍取最后时间步
        out = self.bn(out)
        out = self.classifier(out)
        return self.activation(out)

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
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))      # LSTM 输出：所有时间步
        out = out[:, -1, :]                  # 取最后时间步的输出
        out = self.bn(out)
        out = self.classifier(out)
        return self.activation(out)