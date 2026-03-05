import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1dEncoder(nn.Module):
    """
    Input:  (B, C, L) 
    Output: z: (B, z_ch, Lz) with 4 downsamples by 2
    """
    def __init__(self, in_ch: int = 12, z_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=3, stride=2, padding=1),   
            nn.BatchNorm1d(64),
            nn.SiLU(),

            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),    
            nn.BatchNorm1d(128),
            nn.SiLU(),

            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm1d(256),
            nn.SiLU(),

            nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm1d(256),
            nn.SiLU(),

            nn.Conv1d(256, z_ch, kernel_size=3, stride=1, padding=1), 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Conv1dDecoder(nn.Module):
    """
    Input:  z: (B, z_ch, Lz)
    Output: x_hat: (B, out_ch, L)
    """
    def __init__(self, out_ch: int = 12, z_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(z_ch, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.SiLU(),

            nn.ConvTranspose1d(256, 256, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm1d(256),
            nn.SiLU(),

            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm1d(128),
            nn.SiLU(),

            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm1d(64),
            nn.SiLU(),

            nn.ConvTranspose1d(64, out_ch, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

# two head 1D AE ######################################################################################
class TwoHeadAutoEncoder1D(nn.Module):
    """
    Two-head AE:
      encoder trunk -> z_total
      split -> z_dmg, z_ndmg
      decoder reconstructs from concat([z_dmg, z_ndmg])
    """
    def __init__(self, in_ch: int = 12, z_dmg_ch: int = 64, z_ndmg_ch: int = 64):
        super().__init__()
        self.in_ch = in_ch
        self.z_dmg_ch = z_dmg_ch
        self.z_ndmg_ch = z_ndmg_ch
        self.z_total_ch = z_dmg_ch + z_ndmg_ch

        # simplest: one encoder producing concatenated channels
        self.encoder = Conv1dEncoder(in_ch=in_ch, z_ch=self.z_total_ch)

        # decoder expects total channels (concat of both heads)
        self.decoder = Conv1dDecoder(out_ch=in_ch, z_ch=self.z_total_ch)

    @staticmethod
    def crop_to(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        L = x.shape[-1]
        return x_hat[..., :L]

    def split_latent(self, z_total: torch.Tensor):
        # z_total: (B, z_total_ch, Lz)
        z_dmg = z_total[:, : self.z_dmg_ch, :]
        z_ndmg = z_total[:, self.z_dmg_ch : self.z_total_ch, :]
        return z_dmg, z_ndmg

    def forward(self, x: torch.Tensor):
        z_total = self.encoder(x)                       # (B, z_dmg+z_ndmg, Lz)
        z_dmg, z_ndmg = self.split_latent(z_total)      # two heads
        x_hat = self.decoder(torch.cat([z_dmg, z_ndmg], dim=1))
        x_hat = self.crop_to(x_hat, x)
        return x_hat, z_ndmg, z_dmg