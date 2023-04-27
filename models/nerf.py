import torch
from torch import nn
import torch.nn.functional as F

class PosEmbedding(nn.Module):
    def __init__(self, max_logscale, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        super().__init__()
        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freqs = 2**torch.linspace(0, max_logscale, N_freqs)
        else:
            self.freqs = torch.linspace(1, 2**max_logscale, N_freqs)

    def forward(self, x):
        """
        Inputs:
            x: (B, 3)

        Outputs:
            out: (B, 6*N_freqs+3)
        """
        out = [x]
        for freq in self.freqs:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)


class NeRF(nn.Module):
    def __init__(self, typ,
                 D=8, W=256, skips=[4],
                 in_channels_xyz=63, in_channels_dir=27):
        """
        ---Parameters for the original NeRF---
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        skips: add skip connection in the Dth layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        in_channels_t: number of input channels for t

        ---Parameters for NseRF-W (used in fine model only as per section 4.3)---
        ---cf. Figure 3 of the paper---
        encode_appearance: whether to add appearance encoding as input (NeRF-A)
        in_channels_a: appearance embedding dimension. n^(a) in the paper
        encode_transient: whether to add transient encoding as input (NeRF-U)
        in_channels_t: transient embedding dimension. n^(tau) in the paper
        beta_min: minimum pixel color variance
        """
 
        super().__init__()
        self.typ = typ
        self.D = D
        self.W = W
        self.skips = skips
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir

        self.pts_linears = nn.ModuleList(
            [nn.Linear(in_channels_xyz, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + in_channels_xyz, W) for i in range(D-1)])
        
        
        # self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                        nn.Linear(W+in_channels_dir, W//2), nn.ReLU(True))

        # static output layers
        self.static_rgb = nn.Linear(W//2, 3)
        self.alpha_linear = nn.Linear(W, 1)
        self.feature_linear = nn.Linear(W,W)

    def forward(self, x):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: the embedded vector of position (+ direction + appearance + transient)
            sigma_only: whether to infer sigma only.
            has_transient: whether to infer the transient component.

        Outputs (concatenated):
            if sigma_ony:
                static_sigma
            elif output_transient:
                static_rgb, static_sigma, transient_rgb, transient_sigma, transient_beta
            else:
                static_rgb, static_sigma
        """
  
      
        input_xyz, input_dir_a = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir], dim=-1)
            

        xyz_ = input_xyz
        for i in range(self.D):
            xyz_ = self.pts_linears[i](xyz_)
            xyz_ = F.relu(xyz_)
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], 1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        static_sigma = self.alpha_linear(xyz_) # (B, 1)
        feature  = self.feature_linear(xyz_)
        xyz_ = torch.cat([feature,static_sigma],-1)

        for i, l in enumerate(self.view_linear):
            xyz_ = self.view_linear[i](xyz_)

        static_rgb = self.static_rgb(xyz_) # (B, 3)
        static = torch.cat([static_rgb, static_sigma], 1) # (B, 4)

        #self.dir_encoding = views_linear
       
        #self.static_rgb
        #self.alpha_linear 
        #self.feature_linear

        return static # (B, 9)