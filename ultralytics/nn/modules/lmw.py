# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""LMW modules."""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import Conv

__all__ = ("DWRBottleneck", "LKABottleneck", "LKCA", "MSDP")


class LKABottleneck(nn.Module):
    """Large-kernel attention bottleneck."""

    def __init__(self, c: int, shortcut: bool = True):
        """Initialize the bottleneck."""
        super().__init__()
        self.dw5 = Conv(c, c, k=5, g=c, act=False)
        self.dw7 = Conv(c, c, k=7, g=c, d=3, act=False)
        self.cv = Conv(c, c, k=1, act=False)
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the bottleneck to the input tensor."""
        y = self.cv(self.dw7(self.dw5(x)))
        return x + y if self.add else y


class LKCA(nn.Module):
    """Cross-stage block with large-kernel attention bottlenecks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, e: float = 0.5):
        """Initialize the LKCA block."""
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*(LKABottleneck(c_, shortcut=shortcut) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LKCA to the input tensor."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class DWRBottleneck(nn.Module):
    """Dilated depthwise residual bottleneck."""

    def __init__(self, c: int, shortcut: bool = True):
        """Initialize the bottleneck."""
        super().__init__()
        self.cv1 = Conv(c, c, k=3)
        self.m = nn.ModuleList(Conv(c, c, k=3, g=c, d=d, act=False) for d in (1, 3, 5))
        self.cv2 = Conv(3 * c, c, k=1, act=False)
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the bottleneck to the input tensor."""
        y = self.cv1(x)
        y = self.cv2(torch.cat([m(y) for m in self.m], 1))
        return x + y if self.add else y


class MSDP(nn.Module):
    """Cross-stage block with multi-scale dilated bottlenecks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, e: float = 0.5):
        """Initialize the MSDP block."""
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*(DWRBottleneck(c_, shortcut=shortcut) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MSDP to the input tensor."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
