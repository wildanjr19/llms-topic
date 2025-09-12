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