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



####
