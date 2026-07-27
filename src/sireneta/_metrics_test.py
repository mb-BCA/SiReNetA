# -*- coding: utf-8 -*-
# Copyright (c) 2024, Gorka Zamora-López and Matthieu Gilson.
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
Analysis of dynamic communicability and flow
============================================
Functions in testing version, before they are ported to their corresponding
module for 'official' release into the package.

"""
# Standard library imports

# Third party packages
import numpy as np
from numba import jit


## METRICS EXTRACTED FROM THE RESPONSE TENSORS #################################
def Time2Decay(arr, dt, fraction=0.99):
    """
    NOTE: Probably, this function will be deprecated. Replaced by others to
    identify the moment a network, node or pair-wise response reach a given
    value or reach convergence.


    The time that links, nodes or the network need to decay to zero.

    Strictly speaking, this function measures the time that the cumulative
    flow (area under the curve) needs to reach x% of the total (cumulative)
    value. Here 'x%' is controled by the optional parameter 'fraction'.
    For example, 'fraction = 0.99' means the time needed to reach 99%
    of the area under the curve, given a response curve.

    The function calculates the time-to-decay either for all pair-wise
    interactions, for the nodes or for the whole network, depending on the
    input array given.
    - If 'arr' is a (nt,N,N) flow tensor, the output 'ttd_arr' will be an
    (N,N) matrix with the ttd between every pair of nodes.
    - If 'arr' is a (nt,N) temporal flow of the N nodes, the output 'ttd_arr'
    will be an array of length N, containing the ttd of all N nodes.
    - If 'arr' is an array of length nt (total network flow over time), 'ttd_arr'
    will be a scalar, indicating the time at which the whole-network flow decays.

    Parameters
    ----------
    arr : ndarray of adaptive shape, according to the case.
        Temporal evolution of the flow. An array of optional shapes. Either
        (nt,N,N) for the pair-wise flows, shape (nt,N,N) for the in- or output
        flows of nodes, or a 1D array of length nt for the network flow.
    timestep : real valued number.
        Sampling time-step. This has to be the time-step employed to simulate
        the temporal evolution encoded in 'arr'.
    fraction : scalar, optional
        The fraction of the total area-under-the-curve to be reached.
        For example, 'fraction = 0.99' means the time the flow needs to
        reach 99% of the area under the curve.

    Returns
    -------
    ttd_arr : ndarray of variable rank
        The time(s) taken for the flows through links, nodes or the network to
        decay. Output shape depends on input.
    """

    # 0) SECURITY CHECKS
    ## TODO: Write a check to verify the curve(s) has (have) really decayed back
    ## to zero. At this moment, it is the user's responsability to guarantee
    ## that all the curves have decayed reasonably well.
    ## The check should rise a warning to simulate for longer time.

    # Check correct shape, in case input is the 3D array for the pair-wise flow
    arr_shape = np.shape(arr)
    if arr_shape==3:
        if arr_shape[1] != arr_shape[2]:
            raise ValueError("Input array not aligned. For 3D arrays shape (nt x N x N) is expected.")

    # 1) Set the level of cummulative flow to be reached over time
    targetcflow = fraction * arr.sum(axis=0)

    # 2) Calculate the time the flow(s) need to decay
    # Initialise the output array, to return the final time-point
    ## TODO: This version iterates over all the times. This is not necessary.
    ## We could start from the end and save plenty of iterations.
    ttd_shape = arr_shape[1:]
    nsteps = arr_shape[0]
    ttd_arr = nsteps * np.ones(ttd_shape, np.int64)

    # Iterate over time, calculating the cumulative flow(s)
    cflow = arr[0].copy()
    for t in range(1,nsteps):
        cflow += arr[t]
        ttd_arr = np.where(cflow < targetcflow, t, ttd_arr)

    # Finally, convert the indices into integration time
    ttd_arr = ttd_arr.astype(np.float64) * dt

    return ttd_arr



## RANDOMIZATION OF (WEIGHTED) NETWORKS ########################################
@jit(nopython=True)
def RandomiseWeightedNetwork1(con):
    # GORKA: This version seems to be faster, with and without Numba.
    # At least, it is never slower
    """
    Randomises a (weighted) connectivity matrix.
    The function returns a random connectivity matrix with the same number of
    links as the input matrix. The resulting connectivity has the same link
    weights of the input matrix (thus total weight is also conserved) but the
    input / output strengths of the nodes are not conserved. If 'con' is an
    unweighted adjacency matrix, the function returns an Erdos-Renyi-like
    random graph, of same size and number of links as 'con'.
    If the binarisation of 'con' is a symmetric matrix, the result will also be
    symmetric. Otherwise, if 'con' represents a directed network, the result
    will be directed.
    !!!!!!!
    GORKA: In the current version, if the underlying graph is undirected but the
    weights are asymmetric, the function won't work. The result will be a
    symmetric matrix and the total weight will likely not be conserved !!!
    !!!!!!!
    Parameters
    ----------
    con : ndarray
        Adjacency matrix of the (weighted) network.
    Returns
    -------
    rewcon : ndarray
        A connectivity matrix with links between same nodes as 'con' but the
        link weights shuffled.
    """
    # 0) SECURITY CHECKS
    if not type(con) == numpy.ndarray:
        raise TypeError('Please enter the connectivity matrix as a numpy array.')

    # 1) EXTRACT THE NEEDED INFORMATION FROM THE con MATRIX
    N = len(con)

    # Get whether 'con' is directed and calculate the number of links
    if Reciprocity(con) == 1.0:
        directed = False
        L = int( round(0.5*con.astype(bool).sum()) )
    else:
        directed = True
        L = con.astype(bool).sum()

    # Get the list of weights
    if directed:
        nzidx = con.nonzero()
        weights = con[nzidx]
    else:
        nzidx = np.triu(con, k=1).nonzero()
        weights = con[nzidx]

    # Get whether 'con' allows self-loops (non-zero diagonal elements)
    if con.trace() == 0:
        selfloops = False
    else:
        selfloops = True

    # 2) GENERATE THE NEW NETWORK WITH THE WEIGHTS SHUFFLED
    # Initialise the matrix. Give same dtype as 'con'
    rewcon = np.zeros((N,N), dtype=con.dtype)

    # Shuffle the list of weights
    numpy.random.shuffle(weights)

    # Finally, add the links at random
    counter = 0
    while counter < L:
        # 2.1) Pick up two nodes at random
        source = int(N * numpy.random.rand())
        target = int(N * numpy.random.rand())

        # 2.2) Check if they can be linked, otherwise look for another pair
        if rewcon[source,target]: continue
        if source == target and not selfloops: continue

        rewcon[source,target] = weights[counter]
        if not directed:
            rewcon[target,source] = weights[counter]

        counter += 1

    return rewcon

@jit
def RandomiseWeightedNetwork2(con):
    ## Deprecated function. Not desirable output. See the docstring.
    """
    Randomises a (weighted) connectivity matrix.

    The function returns a random connectivity matrix with the same number of
    links as the input matrix.
    ACHTUNG!!! However, the weights are always randomised such
    that they are asymmetric. Therefore, if 'con' is symmetric weighted matrix,
    the function returns an undirected underlying graph, but with the weights
    asymmetric. So, I don't like it.

    Parameters
    ----------
    con : ndarray
        Adjacency matrix of the (weighted) network.

    Returns
    -------
    rewcon : ndarray
        A connectivity matrix with links between same nodes as 'con' but the
        link weights shuffled.
    """

    # 0) SECURITY CHECKS
    if not type(con) == numpy.ndarray:
        raise TypeError('Please enter the connectivity matrix as a numpy array.')

    # 1) EXTRACT THE NEEDED INFORMATION FROM THE con MATRIX
    N = len(con)
    nzidx = con.nonzero()
    weights = con[nzidx]

    # Get whether 'con' is directed and calculate the number of links
    if Reciprocity(con) == 1.0:
        directed = False
        L = int( round(0.5*con.astype(bool).sum()) )
    else:
        directed = True
        L = con.astype(bool).sum()

    # Get whether 'con' allows self-loops (non-zero diagonal elements)
    if con.trace() == 0:
        selfloops = False
    else:
        selfloops = True

    # 2) GENERATE THE NEW NETWORK WITH THE WEIGHTS SHUFFLED
    # Initialise the matrix. Give same dtype as 'con'
    rewcon = np.zeros((N,N), dtype=con.dtype)

    # Finally, add the links at random
    counter = 0
    while counter < L:
        # 2.1) Pick up two nodes at random
        source = int(N * numpy.random.rand())
        target = int(N * numpy.random.rand())

        # 2.2) Check if they can be linked, otherwise look for another pair
        if rewcon[source,target]: continue
        if source == target and not selfloops: continue

        # 2.3) If the nodes are linkable, place the link
        rewcon[source,target] = 1
        if not directed:
            rewcon[target,source] = 1

        counter += 1

    # Shuffle the list of weights
    numpy.random.shuffle(weights)
    newnzidx = rewcon.nonzero()
    rewcon[newnzidx] = weights

    return rewcon


##
