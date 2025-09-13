import torch
import torch.nn as nn
from typing import Optional, Tuple

# rotate 90 derajat
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Fungsi untuk melakukan rotasi 90 derajat pada vektor input
    Rotasi pasangan dimensi (2D): (x_genap, x_ganjil) -> (-x_ganjil, x_genap)
    Args:
        x (tensor) : input dengan ukuran [..., d] d harus genap -> [16, 20] [4, 5, 6, 12]
    Returns:
        tensor
    """
    x_genap = x[..., ::2] # ambil elemen genap
    x_ganjil = x[..., 1::2] # ambil elemen ganjil
    # rubah ganjil menjadi negatif
    # gabungkan pada dimensi terakhir
    # reshape, pastikan kembali ke bentuk semula
    return torch.stack((-x_ganjil, x_genap), dim=-1).reshape_as(x)


class RotaryEmbedding(nn.Module):
    """
    RoPE dengan cos sin
    Args:
        dim (int): dimensi yang akan dilakukann RoPE (harus genap agar bisa dibagi 2)
        base (float) : nilai theta default/base = 10000 
    """
    def __init__(self, dim: int, base: float = 10000):
        super().__init__()
        # pastikan dim genap
        if dim % 2 != 0:
            raise ValueError("Dimensi harus genap")
        
        self.dim = dim
        # frekuensi invers
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        # register inv_freq agar tidak trainable
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # caching seq_len
        self._seq_len_cached = 0

        # register buffer cos sin
        self.register_buffer("cos_cached", torch.empty(1, 0, 1, dim), persistent=False)
        self.register_buffer("sin_cached", torch.empty(1, 0, 1, dim), persistent=False)