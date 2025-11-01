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

# TODO: Add aliases to functions in GAlib, once GAlib is properly updated in PyPI

"""
Random network and surrogate generation module
==============================================

Functions to construct synthetic networks and generate surrogates out of given
binary or weighted networks.

Generation and randomization of binary graphs
---------------------------------------------
These functions are imported from or inspired by the GAlib library
(https://github.com/gorkazl/pyGAlib)
Please see doctsring of module "galib.models" for a list of functions.  ::

    >>> import galib
    >>> help(galib.models)

Generation of random weighted networks
--------------------------------------
GenRandomWeightedCon
    Generates a randomly directed network for specified link probability and
    weight distribution.
SeedRandomWeights
    Assigns random weights to the links of a given connectivity matrix.

Generation of surrogates from given connectivity
------------------------------------------------
 GenRandomWeightedCon_Like
    Generates a random connectivity matrix with same number of links and same
    weight values as the input `con`, but randomly re-assigned.

ShuffleLinkWeights
    Randomly re-allocates the weights of the links without changing the links.
"""

# Standard library imports
import warnings
# Third party packages
import numpy as np
# Local imports from sireneta
from . import io_helpers


## RANDOM NETWORK MODELS #######################################################
def GenRandomWeightedCon(N, con_prob, w_distr, directed=False, **arg_w_distr):
    """
    Generates a random connectivity matrix, of given connection probability
    between each pair of nodes and connection weights following a desired
    distribution, e.g., numpy.random.uniform or scipy.stats.uniform, scipy.stats.norm).
    The resulting connectivity is directed (

    Parameters
    ----------
    N : integer
        Number of nodes.
    con_prob : float
        The probability of connection for every link (pair of nodes).
    w_distr : function
        The distribution function for drawing weight samples, it must have a
        `size` argument for the number of generated samples.
    directed : boolean (optional)
        `False` (default) to generate an undirected graph with symmetric weights.
        `True` for a directed connectivity with asymmetric weights.
    arg_w_distr : dictionary or named arguments
        The other arguments necessary to define `w_distr`.

    Returns
    -------
    con : ndarray (2d) of shape (N,N).
        Connectivity matrix for a network with randomly seeded links and weights.

    Examples
    --------
    GenRandomWeightedNet(3, 0.7, np.random.uniform, low=0.0, high=1.0)
    GenRandomWeightedNet(3, 0.7, np.random.normal, loc=0.0, scale=1.0)
    GenRandomWeightedNet(3, 0.7, w_smpl)
        with def w_smpl(size): return np.random.uniform(low=0.0, high=1.0, size=size)
    """
    # 0) SECURITY CHECKS
    if not type(N) == int:
        raise TypeError( "Please enter the number of nodes 'N' as an integer." )
    if (not type(con_prob) == float) or con_prob < 0.0 or con_prob > 1.0:
        raise TypeError( "Please enter the probability of connection 'con_prob' as float." )

    # 1) GENERATE THE BINARY CONNECTIVITY MATRIX
    adjmatrix = np.random.rand(N, N)
    # Convert to boolean, with 'True' for thresholded values
    adjmatrix = adjmatrix <= con_prob
    # Remove potential self-loops
    np.fill_diagonal(adjmatrix, False)

    # 2) SEED THE WEIGHTS
    con = np.zeros_like(adjmatrix, dtype=np.float64)
    con[adjmatrix] = w_distr(**arg_w_distr, size=adjmatrix.sum())

    # 3) In case of undirected connectivity desired
    if not directed:
        con[np.triu_indices(N,k=1)] = 0
        con += con.T

    return con

def SeedRandomWeights(con, w_distr, **arg_w_distr):
    # TODO: This function could/should identify whether 'con' is (un)directed and
    # return (a)symmetric weights accordingly.
    """
    Assigns random weights to the links of a given connectivity matrix. The weights
    are sampled from a given distribution, e.g., numpy.random.uniform,
    scipy.stats.uniform or scipy.stats.norm.

    NOTE: The function does not change the input `con` in place, it returns a
    copy of `con` with the new weights.

    Parameters
    ----------
    con : ndarray (2d) of shape (N,N).
        The matrix of a network whose weights want to be randomly assigned.
        `con` can be either a connectivity matrix (already weighted or not), or
        the mask (boolean matrix) of an existing `con`.
    w_distr : function.
        The distribution function for drawing weight samples, it must have a
        `size` argument for the number of generated samples.
    arg_w_distr : dictionary or named arguments.
        The other arguments necessary to define `w_distr`.

    Returns
    -------
    con : ndarray (2d) of shape (N,N).
        A connectivity matrix with same links as input `con` but link weights
        reassigned, drawn from distribution `w_distr`.
    """
    # 0) SECURITY CHECKS
    io_helpers.validate_con(con)

    # Extract the mask (if needed)
    if con.dtype == np.bool:
        mask = con.copy()
    else:
        mask = con.astype(bool)

    # Remove self-loops (if needed)
    if np.any(mask.diagonal()):
        warnings.warn( "Diagonal elements in 'con' are being ignored.",
                        category=RuntimeWarning )
        np.fill_diagonal(mask, False)

    # 1) Create a copy of the matrix and seed the weights to the links
    newcon = np.zeros_like(mask, dtype=np.float64)
    newcon[mask] = w_distr(**arg_w_distr, size=mask.sum())

    return newcon

def RndNonNormalNet(con):
    raise ValueError( "Not implemented yet" )



## NETWORK RANDOMIZATION FUNCTIONS (SURROGATE GENERATION) ######################
def GenRandomWeightedCon_Like(con, impose_directed=False, tol=1e-15):
    """
    Generates a random connectivity matrix with same number of links and same
    weight values as the input `con`, but randomly re-assigned.

    Parameters
    ----------
    con : ndarray (2d) of shape (N,N)
        The connectivity matrix of the network.
    impose_directed : boolean
        The function detects whether `con` is symmetric or asymmetric, and returns
        accordingly a symmetric (undirected) or an asymmetric (directed) surrogate.
        But if `impose_directed = True` is given, it will always return an
        asymmetric connectivity matrix.
    tol : float
        Adjust `tol` to avoid considering a symmetric `con` as if it were asymmetric
        due to small rounding errors in floating numbers.

    Returns
    -------
    newcon : ndarray (2d) of shape (N,N)
        A connectivity matrix with same number of links and same weight distribution
        as `con`, but fully randomised.
    """
    # NOTE: The function might be organised differently, but it is explicit and clear.
    # 0) SECURITY CHECKS
    io_helpers.validate_con(con)

    # 1) IDENTIFY WHICH ALGORITHM TO USE
    if impose_directed:
        algo = 'directed'
    else:
        asymmetry = abs(con - con.T).mean()
        if asymmetry > tol:
            algo = 'directed'
        else:
            algo = 'undirected'

    # 2) GENERATE THE RANDOM CONNECTIVITY MATRIX
    N = len(con)
    ## The directed case
    if algo=='directed':
        # Get all weights as 1D array, including the zero weights
        idx = (np.eye(N)-1).nonzero()
        weights = con[idx]
        # Initialise the matrix
        newcon = np.zeros_like(con, dtype=np.float64)
        # Re-allocate the shuffled list of weights
        np.random.shuffle(weights)
        newcon[idx] = weights

    ## The undirected case
    elif algo=='undirected':
        # Get all upper-triangular weights as 1D array, including zero weights
        idx = np.triu_indices(N, k=1)
        weights = con[idx]
        # Initialise the matrix
        newcon = np.zeros_like(con, dtype=np.float64)
        # Re-allocate the shuffled list of weights (upper triangular)
        np.random.shuffle(weights)
        newcon[idx] = weights
        # Add the corresponding symmetric links (lower triangular)
        newcon = newcon + newcon.T

    return newcon

def ShuffleLinkWeights(con):
    # TODO: this function could/should identify whether 'con' is (un)directed
    # and thus return (a)symmetric matrix accordingly.
    """
    Randomly re-allocates the link weights of an input network.

    The function does not alter the position of the links, it only shuffles
    the weights associated to the links. Therefore, the binarised version
    is preserved.

    NOTE: The function does not change the input `con` in place, it returns a
    copy of `con` with the weights shuffled.

    Parameters
    ----------
    con : ndarray (2d) of shape (N,N).
        The connectivity matrix of the network.

    Returns
    -------
    newcon : ndarray (2d) of shape (N,N).
        A connectivity matrix with links between same nodes as `con` but the
        link weights shuffled.
    """
    # 0) SECURITY CHECKS
    io_helpers.validate_con(con)

    # 1) Extract the weights from 'con'
    nzidx = con.nonzero()
    weights = con[nzidx]

    # 2) Generate the new network with the weights shuffled
    np.random.shuffle(weights)
    newcon = np.zeros_like(con, dtype=np.float64)
    newcon[nzidx] = weights

    return newcon


###
