# This file is part of tad-mctc.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Batch Utility: Masks
====================

Functions for creating masks that discern between padding and actual values.
"""
from __future__ import annotations

import torch
import re
from ...typing import Tensor
from .atoms import real_atoms

__all__ = ["real_pairs"]


# scripting or tracing does not improve performance
def real_pairs(numbers: Tensor, mask_diagonal: bool = True, mon_A_indices=None, mon_B_indices=None) -> Tensor:
    """
    Create a mask for pairs of atoms from atomic numbers, discerning padding
    and actual atoms. Padding value is zero.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms.
    mask_diagonal : bool, optional
        Flag for also masking the diagonal, i.e., all pairs with the same
        indices. Defaults to `True`, i.e., writing False to the diagonal.

    Returns
    -------
    Tensor
        Mask for atom pairs that discerns padding and real atoms.
    """
    if mask_diagonal is True:
        return real_pairs_maskdiag(numbers, mon_A_indices, mon_B_indices)
    return real_pairs_no_maskdiag(numbers)


def real_pairs_maskdiag(numbers: Tensor, mon_A_indices=None, mon_B_indices=None) -> Tensor:
    """
    Create a mask for pairs of atoms from atomic numbers, discerning padding
    and actual atoms. Padding value is zero.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms.

    Returns
    -------
    Tensor
        Mask for atom pairs that discerns padding and real atoms.
    """
    real = real_atoms(numbers)

    if mon_A_indices is not None and mon_B_indices is not None:
        print(f"{numbers= }")
        maskA = torch.zeros_like(real)
        maskB = torch.zeros_like(real)

        batch_size = maskA.shape[0]
        rows = torch.arange(batch_size).unsqueeze(1)

        maskA[rows, mon_A_indices] = True
        print(f"{maskA= }")
        maskB[rows, mon_B_indices] = True
        #I am setting maskB zero index to False, because molecule B will never have atom 0 
        #if I use the qcelemental, and then they pad with zeros, so the code is accidentally assigning 
        #atom 0 to A and B
        maskB[rows, 0] = False
        print(f"{maskB= }")


        #Get atom pairs between mon A and mon B, double count on purpose cuz they * by 0.5
        AB = maskA.unsqueeze(-2) * maskB.unsqueeze(-1)
        BA = maskB.unsqueeze(-2) * maskA.unsqueeze(-1)
        mask = AB | BA
        print("Mask sum per batch:", mask.sum(dim=(-1,-2)))


    else:
        mask = real.unsqueeze(-2) * real.unsqueeze(-1)
        mask *= ~torch.diag_embed(torch.ones_like(real))
    return mask

def real_pairs_no_maskdiag(numbers: Tensor) -> Tensor:
    """
    Create a mask for pairs of atoms from atomic numbers, discerning padding
    and actual atoms. Padding value is zero.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms.

    Returns
    -------
    Tensor
        Mask for atom pairs that discerns padding and real atoms.
    """
    real = real_atoms(numbers)
    return real.unsqueeze(-2) * real.unsqueeze(-1)
