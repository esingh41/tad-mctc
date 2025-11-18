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
Coordination number: DFT-D3
===========================

Calculation of coordination number for DFT-D3.
"""
from __future__ import annotations

import torch

from .. import storch
from ..batch import real_pairs
from ..data import radii
from ..typing import DD, Any, CountingFunction, Tensor
from . import defaults
from .count import dexp_count, exp_count

#Needed for summing across the coordination numbers
from apnet_pt.util import scatter_sum_compile


__all__ = ["cn_d3", "cn_d3_gradient"]


def cn_d3(
    numbers: Tensor,
    positions: Tensor,
    *,
    counting_function: CountingFunction | None = None,
    rcov: Tensor | None = None,
    cutoff: Tensor | None = None,
    **kwargs: Any,
) -> Tensor:
    """
    Compute the D3 fractional coordination (exponential counting function).

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape ``(..., nat)``.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    counting_function : CountingFunction, optional
        Calculate weight for pairs. Defaults to
        :func:`tad_mctc.ncoord.count.exp_count`.
    rcov : Tensor | None, optional
        Covalent radii for each species. Defaults to ``None``.
    cutoff : Tensor | None, optional
        Real-space cutoff. Defaults to ``None``.
    kwargs : dict[str, Any]
        Pass-through arguments for counting function. For example, ``kcn``,
        the steepness of the counting function, which defaults to
        :data:`tad_mctc.ncoord.defaults.KCN_D3`.

    Returns
    -------
    Tensor
        Coordination numbers for all atoms (shape: ``(..., nat)``).

    Raises
    ------
    ValueError
        If shape mismatch between ``numbers``, ``positions`` and
        ``rcov`` is detected.
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    
    if cutoff is None:
        cutoff = torch.tensor(defaults.CUTOFF_D3, **dd)

    if rcov is None:
        rcov = radii.COV_D3(**dd)[numbers]
    else:
        rcov = rcov.to(**dd)

    if counting_function is None:
        counting_function = exp_count

    if numbers.shape != rcov.shape:
        raise ValueError(
            f"Shape of covalent radii {rcov.shape} is not consistent with "
            f"({numbers.shape})."
        )
    if numbers.shape != positions.shape[:-1]:
        raise ValueError(
            f"Shape of positions ({positions.shape[:-1]}) is not consistent "
            f"with atomic numbers ({numbers.shape})."
        )

    eps = torch.tensor(torch.finfo(positions.dtype).eps, **dd)

    mask = real_pairs(numbers, mask_diagonal=True)
    distances = torch.where(mask, storch.cdist(positions, positions, p=2), eps)

    rc = rcov.unsqueeze(-2) + rcov.unsqueeze(-1)
    cf = torch.where(
        mask * (distances <= cutoff),
        counting_function(distances, rc, **kwargs),
        torch.tensor(0.0, **dd),
    )

    return torch.sum(cf, dim=-1)

#Borrowed from APNET
def get_distances(RA, RB, e_source, e_target):
        RA_source = RA.index_select(0, e_source)
        RB_target = RB.index_select(0, e_target)
        dR_xyz = RB_target - RA_source

        # Compute distances with safe operation for square root
        # dR = torch.sqrt(nn.functional.relu(torch.sum(dR_xyz**2, dim=-1)))
        dR = torch.sqrt(torch.sum(dR_xyz * dR_xyz, dim=-1).clamp_min(1e-10))
        return dR, dR_xyz

def cn_d3_apnet(
    batch,
    *,
    counting_function: CountingFunction | None = None,
    rcov: Tensor | None = None,
    cutoff: Tensor | None = None,
    **kwargs: Any,
) -> Tensor:
    """
    Compute the D3 fractional coordination (exponential counting function).

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape ``(..., nat)``.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    counting_function : CountingFunction, optional
        Calculate weight for pairs. Defaults to
        :func:`tad_mctc.ncoord.count.exp_count`.
    rcov : Tensor | None, optional
        Covalent radii for each species. Defaults to ``None``.
    cutoff : Tensor | None, optional
        Real-space cutoff. Defaults to ``None``.
    kwargs : dict[str, Any]
        Pass-through arguments for counting function. For example, ``kcn``,
        the steepness of the counting function, which defaults to
        :data:`tad_mctc.ncoord.defaults.KCN_D3`.

    Returns
    -------
    Tensor
        Coordination numbers for all atoms (shape: ``(..., nat)``).

    Raises
    ------
    ValueError
        If shape mismatch between ``numbers``, ``positions`` and
        ``rcov`` is detected.
    """
    RA = batch.RA

    #dictionary of defaults; RA is reference tensor, extracting device and precision
    #so can be used for other tensors
    dd: DD = {"device": RA.device, "dtype": RA.dtype}

    #What is this cutoff
    #I don't really have to care for a cutoff right, 
    #because I want to use all of the intramonomer edges anyways right?

    if cutoff is None:
        cutoff = torch.tensor(defaults.CUTOFF_D3, **dd)

    if counting_function is None:
        counting_function = exp_count
        
    ############################################
    ##Getting the covalent radii for monomer A##
    ############################################
    ZA = batch.ZA
    #ZA =tensor([8, 1, 1])
    print(f"{ZA =}")
    rcov_A = radii.COV_D3(**dd)[ZA] 
    print(f"{rcov_A = }")
    #rcov_A = tensor([1.5874, 0.8063, 0.8063])
    e_AA_source = batch.e_AA_source
    e_AA_target = batch.e_AA_target
    rc_A = rcov_A.index_select(0, e_AA_source) + rcov_A.index_select(0, e_AA_target)
    print(f"{rc_A = }")
    #rc_A = tensor([2.3937, 2.3937, 2.3937, 1.6126, 2.3937, 1.6126])
    #rc_A contains the covalent radii sums


    ############################################
    ##Getting the covalent radii for monomer B##
    ############################################
    ZB = batch.ZB
    rcov_B = radii.COV_D3(**dd)[ZB] 
    print(f"{rcov_B = }")
    #rcov_B = tensor([1.5874, 0.8063, 0.8063])
    e_BB_source = batch.e_BB_source
    e_BB_target = batch.e_BB_target
    rc_B = rcov_B.index_select(0, e_BB_source) + rcov_B.index_select(0, e_BB_target)
    print(f"{rc_B = }")
    #rc_B = tensor([2.3937, 2.3937, 2.3937, 1.6126, 2.3937, 1.6126])
    


    ############################################
    #Getting the coordination #s for monomer A##
    ############################################
    RA = batch.RA
    e_AA_source = batch.e_AA_source
    e_AA_target = batch.e_AA_target
    dRA, _ = get_distances(RA, RA, e_AA_source, e_AA_target)
    print(f"{dRA = }")
    #dRA = tensor([0.9581, 0.9647, 0.9581, 1.5118, 0.9647, 1.5118]) dRA is 1D, so covalent radii also need to be one D
    
    cf_A = torch.where(
        (dRA <= cutoff),
        counting_function(dRA, rc_A),
        torch.tensor(0.0, **dd)
    )
    print(f"{cf_A = }")
    #cf_A = tensor([1.0000, 1.0000, 1.0000, 0.7440, 1.0000, 0.7440])
    #Hmmm, what does a coordination number of 0.74 mean? 3/4s of a bond?
    #Oxygen has the same coordination number with respect to both Hs makes sense
    #cf_A = scatter_sum_compile(cf_A, e_AA_source, 1,)
    cn_A_size = e_AA_source.max().item() + 1
    cn_A = torch.zeros(cn_A_size, dtype=cf_A.dtype)
    cn_A.scatter_reduce_(0, e_AA_source, cf_A, reduce="sum", include_self=False)
    print(f"{cn_A = }")
    #Makes sense oxygen has two bonding partners, and then hydrogen also has close to two? Weird
    #cn_A = tensor([2.0000, 1.7440, 1.7440])
    
    #Computing the coordination numbers for Monomer B
    RB=batch.RB
    e_BB_source = batch.e_BB_source
    e_BB_target = batch.e_BB_target
    dRB, _ = get_distances(RB, RB, e_BB_source, e_BB_target)
    cf_B = torch.where(
        (dRB <= cutoff),
        counting_function(dRB, rc_B),
        torch.tensor(0.0, **dd)
    )

    cn_B_size = e_BB_source.max().item() + 1
    cn_B = torch.zeros(cn_B_size, dtype=cf_B.dtype)
    cn_B.scatter_reduce_(0, e_BB_source, cf_B, reduce="sum", include_self=False)
    print(f"{cn_B = }")
    #cn_B = tensor([2.0000, 1.7435, 1.7435])
    return cn_A, cn_B

def cn_d3_gradient(
    numbers: Tensor,
    positions: Tensor,
    *,
    dcounting_function: CountingFunction = dexp_count,
    rcov: Tensor | None = None,
    cutoff: Tensor | None = None,
    **kwargs: Any,
) -> Tensor:
    """
    Compute the derivative of the fractional coordination number with respect
    to atomic positions.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape ``(..., nat)``.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    dcounting_function : CountingFunction, optional
        Derivative of the counting function. Defaults to
        :func:`tad_mctc.ncoord.count.dexp_count`.
    rcov : Tensor | None, optional
        Covalent radii for each species. Defaults to ``None``.
    cutoff : Tensor | None, optional
        Real-space cutoff. Defaults to ``None``.
    kwargs : dict[str, Any]
        Pass-through arguments for counting function. For example, ``kcn``,
        the steepness of the counting function, which defaults to
        :data:`tad_mctc.ncoord.defaults.KCN_D3`.

    Returns
    -------
    Tensor
        Coordination numbers for all atoms (shape: ``(..., nat, nat, 3)``).

    Raises
    ------
    ValueError
        If shape mismatch between ``numbers``, ``positions`` and
        ``rcov`` is detected.
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    if cutoff is None:
        cutoff = torch.tensor(defaults.CUTOFF_D3, **dd)

    if rcov is None:
        rcov = radii.COV_D3(**dd)[numbers]
    else:
        rcov = rcov.to(**dd)

    if numbers.shape != rcov.shape:
        raise ValueError(
            f"Shape of covalent radii {rcov.shape} is not consistent with "
            f"({numbers.shape})."
        )
    if numbers.shape != positions.shape[:-1]:
        raise ValueError(
            f"Shape of positions ({positions.shape[:-1]}) is not consistent "
            f"with atomic numbers ({numbers.shape})."
        )

    eps = torch.tensor(torch.finfo(positions.dtype).eps, **dd)

    mask = real_pairs(numbers, mask_diagonal=True)
    distances = torch.where(mask, storch.cdist(positions, positions, p=2), eps)

    rc = rcov.unsqueeze(-2) + rcov.unsqueeze(-1)
    dcf = torch.where(
        mask * (distances <= cutoff),
        dcounting_function(distances, rc, **kwargs),
        torch.tensor(0.0, **dd),
    )

    # (..., nat, nat, 3)
    rij = positions.unsqueeze(-3) - positions.unsqueeze(-2)

    # (..., nat, nat, 1) * (..., nat, nat, 3)
    return (dcf / distances).unsqueeze(-1) * rij  # "...ij,...ijx->...ijx"
