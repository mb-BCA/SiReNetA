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


# TODO: REVISE ALL THESE FUNCTIONS. DO WE WANT THEM HERE OR ...
# TODO: Add aliases to functions in GAlib, once GAlib is properly updated in PyPI

# NOTE: THIS MODULE IS NOT NEEDED IN THE FIRST RELEASE. WORK ON IT AFTERWARDS !!

"""
Network and surrogate generation module
=======================================

Functions to construct synthetic networks and generate surrogates out of given
binary or weighted networks.

Generation and randomization of binary graphs
---------------------------------------------
These functions are imported from or inspired by the GAlib library
(https://github.com/gorkazl/pyGAlib)
Please see doctsring of module "galib.models" for a list of functions.  ::

    >>> import galib
    >>> help(galib.models)


Generation of directed and weighted networks
--------------------------------------------
GenRandomWeightedNet
    Generates a randomly directed network (Erdős–Rényi) with given probability
    of connection and weight distribution.
GenRandomMaskNet
    Generates a connectivity matrix with a given topology and random weights
    according to a given distribution.

Surrogate methods for directed weighted networks
------------------------------------------------
ShuffleWeightsFixedLinks
    Randomly re-allocates the weights of the links without changing the links.
ShuffleLinks
    Randomises a connectivity matrix (links with weights).
"""

# Standard library imports

# Third party packages
import numpy as np
# import galib.models
#from numba import jit



## RANDOM NETWORK MODELS #######################################################
def GenRandomWeightedCon(N, con_prob, w_distr, directed=True, **arg_w_distr):
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
        'size' argument for the number of generated samples.
    directed : boolean (optional)
        True if a directed graph is desired. False, for an undirected graph.
    arg_w_distr : dictionary or named arguments
        The other arguments necessary to define 'w_distr'.

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

def GenRandomMaskNet(mask_con, w_distr, **arg_w_distr):
    """
    Generates a squared connectivity matrix for a mask that determines the
    connectivity topology for the network. Weights are sampled from a given
    distribution (like numpy.random.uniform or scipy.stats.uniform,
    scipy.stats.norm).

    Parameters
    ----------
    mask_con : ndarray of shape (N x N).
        The mask of existing connections.
    w_distr : function.
        The distribution function for drawing weight samples, it must have a
        'size' argument for the number of generated samples.
    arg_w_distr : dictionary or named arguments.
        The other arguments necessary to define 'w_distr'.

    Returns
    -------
    con : ndarray of rank-2 and shape (N x N).
        A connectivity matrix with topology determined by 'mask_con' and random
        weights drawn from the distribution 'w_distr'.
    """
    # 0) SECURITY CHECKS
    if (not type(mask_con) == np.ndarray) and len(mask_con.shape) == 2:
        raise TypeError( "Please enter the matrix shape as integer." )
    mask_con = np.array(mask_con, dtype=bool)
    if np.any(mask_con.diagonal()):
        print( "Warning: Diagonal elements in mask_con ignored." )
        np.fill_diagonal(mask_con, False)

    # 1) POPULATE WITH WEIGHTS
    con = np.zeros_like(mask_con, dtype=float)
    con[mask_con] = w_distr(**arg_w_distr, size=mask_con.sum())

    return con

def RndNonNormalNet(con):
    raise ValueError( "Not implemented yet" )


## RANDOMIZED SURROGATE NETWORKS ############################################
# NOTE: See GAlib.models package

def ShuffleWeightsFixedLinks(con):
    """
    Randomly re-allocates the link weights of an input network.

    The function does not alter the position of the links, it only shuffles
    the weights associated to the links. Therefore, the binarised version
    is preserved.

    Parameters
    ----------
    con : ndarray (2d) of shape (N,N).
        The connectivity matrix of the network.

    Returns
    -------
    newcon : ndarray of rank-2 and shape (N x N).
        A connectivity matrix with links between same nodes as `con` but the
        link weights shuffled.

    """
    # 0) SECURITY CHECKS
    if not type(con) == np.ndarray:
        raise TypeError( "Please enter the connectivity matrix as a numpy array." )
    con_shape = np.shape(con)
    if (len(con_shape) != 2) or (con_shape[0] != con_shape[1]):
        raise ValueError( "Input not aligned. 'con' should be a 2D array of shape (N x N)." )

    # 1) EXTRACT THE CONSTRAINTS FROM THE con MATRIX
    nzidx = con.nonzero()
    weights = con[nzidx]

    # 2) GENERATE THE NEW NETWORK WITH THE WEIGHTS SHUFFLED
    np.random.shuffle(weights)
    newcon = np.zeros_like(con, dtype=con.dtype)
    newcon[nzidx] = weights

    return newcon

#@jit
def ShuffleLinks(con):
    """
    Randomises a connectivity matrix and its weights.

    Returns a random connectivity matrix (Erdos-Renyi-type) with the same number
    of links and same link weights as the input matrix `con`. Therefore, both
    the total weight (sum of link weights) and the distribution of link weights
    are conserved, but the input/output degrees of the nodes, or their individual
    strengths, are not conserved.

    IMPORTANT: As compared to GAlib, we only consider directed network matrices without
    self-loops.

    Parameters
    ----------
    con : ndarray (2d) of shape (N,N).
        The connectivity matrix of the network.

    Returns
    -------
    newcon : ndarray of rank-2 and shape (N x N)
        A connectivity matrix with links between same nodes as `con` but the
        link weights shuffled.

    """
    # 0) SECURITY CHECKS
    if not type(con) == np.ndarray:
        raise TypeError( "Please enter the connectivity matrix as a numpy array." )
    con_shape = np.shape(con)
    if (len(con_shape) != 2) or (con_shape[0] != con_shape[1]):
        raise ValueError( "Input not aligned. 'con' should be a 2D array of shape (N x N)." )

    # 1) EXTRACT INFORMATION NEEDED FROM THE con MATRIX
    N = con_shape[0]

    # Get all weights as 1D array, including zero weights
    idx = (np.eye(N)-1).nonzero()
    weights = con[idx]

    # 2) GENERATE THE NEW NETWORK WITH THE WEIGHTS SHUFFLED
    # Initialise the matrix. Give same dtype as `con`
    newcon = np.zeros_like(con, dtype=con.dtype)

    # Shuffle the list of weights and allocate
    np.random.shuffle(weights)
    newcon[idx] = weights

    return newcon
