# -*- coding: utf-8 -*-
# Copyright 2024 - 2025, Gorka Zamora-López and Matthieu Gilson.
# Contact: gorka@zamora-lopez.xyz
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
Miscelaneous tools and helpers
==============================

Module for extra helper functionality that doesn't fit as response estimation,
simulations or metric categories.

NNt2tNN
    Transposes a 3D array from shape (N,N,nt) to (nt,N,N) shape.
tNN2NNt
    Transposes a 3D array from shape (nt,N,N) to (N,N,nt) shape.
"""
# Standard library imports

# Third party packages
import numpy as np
import numpy.random


## MISCELLANEOUS FUNCTIONS #####################################################
def NNt2tNN(tensor):
    """Transposes a 3D array from shape (N,N,nt) to (nt,N,N),

    where N is the number of nodes in a network (features) and nt is the
    number of time-points (samples).

    Parameters
    ----------
    tensor : ndarray of rank-3.
        Temporal evolution of the N x N elements in a matrix, arranged with
        shape (N,N,nt).

    Returns
    -------
    newtensor : ndarray of rank-3.
        Same as input `tensor` but in shape (nt,N,N). Matrix rows an columns at
        each slice in `tensor` are conserved row and columns in `newtensor`.

    Notes
    -----
    Please remind that np.transpose() function returns a view of the array,
    not a copy! If you want a copy with the entries properly sorted in memory,
    call the function as follows:

    >>> arr2 = np.copy( NNt2tNN(arr1), order='C' )
    """
    # Security checks
    assert len(np.shape(tensor)) == 3, "3D array required."
    n0, n1, n2 = np.shape(tensor)
    if n0 != n1:
        raise TypeError("3D array of shape (N,N,nt) required.")

    # Transpose the array
    newtensor = np.transpose(tensor, axes=(2,0,1))
    return newtensor

def tNN2NNt(tensor):
    """Transposes a 3D array from shape (nt,N,N) to (N,N,nt),

    where N is the number of nodes in a network (features) and nt is the
    number of time-points (samples).

    Parameters
    ----------
    tensor : ndarray of rank-3.
        Temporal evolution of the N x N elements in a matrix, arranged with
        shape (nt,N,N).

    Returns
    -------
    newtensor : ndarray of rank-3.
        Same as input `tensor` but in shape (N,N,nt). Matrix rows an columns at
        each slice in `tensor` are conserved row and columns in `newtensor`.

    Notes
    -----
    Please remind that np.transpose() function returns a view of the array,
    not a copy! If you want a copy with the entries properly sorted in memory,
    call the function as follows:

    >>> arr2 = np.copy( tNN2NNt(arr1), order='C' )

    """
    # Security checks
    assert len(np.shape(tensor)) == 3, "3D array required."
    n0, n1, n2 = np.shape(tensor)
    if n1 != n2:
        raise TypeError("3D array of shape (nt,N,N) required.")

    newtensor = np.transpose(tensor, axes=(1,2,0))
    return newtensor





##
