import torch
import torch.nn as nn

class PSDHead(nn.Module):
    """
    Predict normalized log-PSD from h_dmg only.
    Output shape: (B, C, K) where K = nfft//2 + 1
    """
    def __init__(self, in_dim: int, out_ch: int, nfft: int, hidden: int = 512):
        super().__init__()
        self.out_ch = out_ch
        self.nfft = nfft
        self.K = nfft // 2 + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_ch * self.K),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        y = self.net(h)  # (B, C*K)
        return y.view(h.size(0), self.out_ch, self.K)