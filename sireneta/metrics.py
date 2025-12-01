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
Analysis of spatio-temporal network responses
=============================================

Metrics to extract information about the network, from the pair-wise responses
$R_{ij}(t)$. See the *responses.py* module for the estimation of $R_{ij}(t)$
for different canonical models.

Metrics derived from the response tensors
-----------------------------------------
GlobalResponse
    Calculates temporal evolution of network response, sum of all pair-wise responses.
Diversity
    Inhomogeneity of the pair-wise responses patterns, caluclated over time.
NodeResponses
    Temporal evolution of the input and output responses for each node.
SelfResponses
    Temporal evolution of the responses of nodes due to stimulus on themselves.
AreaUnderCurve
    Total amount of response accumulated over time.
TimeToPeak
    The time that links, nodes or the network need to reach maximal response.


**Reference and Citation**

1) G. Zamora-Lopez and M. Gilson "An integrative dynamical perspective for graph
theory and the analysis of complex networks" Chaos 34, 041501 (2024).

2) M. Gilson, N. Kouvaris, et al. "Network analysis of whole-brain fMRI
dynamics: A new framework based on dynamic communicability" NeuroImage 201,
116007 (2019).

3) M. Gilson, N. Kouvaris, G. Deco & G.Zamora-Lopez "Framework based on communi-
cability and flow to analyze complex networks" Phys. Rev. E 97, 052301 (2018).
"""

# Standard library imports
# Third party packages
import numpy as np
# Local imports from sireneta
from . import io_helpers


# TODO: MAKE SURE FUNCTIONS RUN AFTER INTRODUCTION OF validate_tensor() CHECKS
# TODO: REVISE AND ADAPT ALL THE DOCSTRING DESCRIPTIONS

## METRICS EXTRACTED FROM THE PAIR-WISE RESPONSE TENSORS #######################
def GlobalResponse(tensor, selfresp=True):
    """
    Calculates network response over time, summed over all pair-wise responses.

    Parameters
    ----------
    tensor : ndarray (3d) of shape (nt,N,N)
        Temporal evolution of the pair-wise responses, as calculated by one of
        the functions of module *responses.py*.
    selfresp : boolean
        If `True` (default), returns the global response summing also the
        self-responses: the response of a node to the initial stimulus applied
        on itself. That is, adds the diagonal $R_{ii}(t)$ entries to the row and
        column sums.
        If `False`, excludes the response of a node to the stimulus applied on
        itself. Excludes the diagonal entries $R_{ii}(t)$ in the row and
        column sums.

    Returns
    -------
    global_response : ndarray (1d) of length nt
        The total networks response over time, summed over all pair-wise
        responses at each time-point.
    """
    # 0) CHECK THE USER INPUT
    io_helpers.validate_tensor(tensor)

    # 1) Compute the global network responses over time (nt)
    global_response = tensor.sum(axis=(1,2))

    if not selfresp:
        nt = len(tensor)
        for t in range(nt):
            global_response[t] -= tensor[t].trace()

    return global_response

def Diversity(tensor):
    """
    Inhomogeneity of the pair-wise responses patterns, calucalted over time.

    Parameters
    ----------
    tensor : ndarray (3d) of shape (nt,N,N)
        Temporal evolution of the pair-wise responses, as calculated by one of
        the functions of module *responses.py*.

    Returns
    -------
    diversity : ndarray (1d) of length nt.
        The diversity (coefficient of variation) of the pattern of network
        responses at each time-point.

    """
    # 0) CHECK THE USER INPUT
    io_helpers.validate_tensor(tensor)

    # 1) Do the calculations
    nt = tensor.shape[0]
    diversity = np.zeros(nt, np.float64)
    diversity[0] = np.nan
    for i_t in range(1,nt):
        temp = tensor[i_t]
        diversity[i_t] = temp.std() / temp.mean()

    return diversity

def NodeResponses(tensor, selfresp=True):
    """
    Temporal evolution of the input and output responses for each node.

    Parameters
    ----------
    tensor : ndarray (3d) of shape (nt,N,N)
        Temporal evolution of the pair-wise responses, as calculated by one of
        the functions of module *responses.py*.
    selfresp : boolean
        If `True` (default), returns the in-/out-responses of the nodes,
        summing also the self-responses: the response of a node to the
        initial stimulus applied on itself. That is, adds the diagonal $R_{ii}(t)$
        entries to the row and column sums.
        If `False`, excludes the response of a node to the stimulus applied on
        itself. Excludes the diagonal entries $R_{ii}(t)$ in the row and
        column sums.

    Returns
    -------
    node_resps : tuple contaning two ndarrays (2d) of shape (nt,N).
        The temporal evolution of the input and output responses for all nodes.
        `node_resps[0]` is the input responses into the nodes node and
        `node_resps[1]` the output node responses.

    See Also
    --------
    SelfResponses : Temporal evolution of the responses of nodes due to stimulus on themselves.
    """
    # 0) CHECK THE USER INPUT
    io_helpers.validate_tensor(tensor)

    # 1) Calculate the input and output node properties
    # When self-responses shall be included to the temporal node responses
    if selfresp:
        inflows = tensor.sum(axis=2)
        outflows = tensor.sum(axis=1)

    # Excluding the self-responses a node due to inital perturbation on itself.
    else:
        nt, N,N = tensor.shape
        inflows = np.zeros((nt,N), np.float64)
        outflows = np.zeros((nt,N), np.float64)
        for i in range(N):
            tempdiags = tensor[:,i,i]
            inflows[:,i]  = tensor[:,i,:].sum(axis=1) - tempdiags
            outflows[:,i] = tensor[:,:,i].sum(axis=1) - tempdiags

    node_resps = ( inflows, outflows )
    return node_resps

def SelfResponses(tensor):
    """
    Temporal evolution of the responses of nodes due to stimulus on themselves.

    Parameters
    ----------
    tensor : ndarray (3d) of shape (nt,N,N)
        Temporal evolution of the pair-wise responses, as calculated by one of
        the functions of module *responses.py*.

    Returns
    -------
    self_resps : ndarray (2d) of shape (nt,N).
        The temporal evolution of the node responses to stimulus on themselves.

    See Also
    --------
    NodeResponses : Temporal evolution of the input and output responses for each node.
    """
    # 0) CHECK THE USER INPUT
    io_helpers.validate_tensor(tensor)

    # 1) Calculate the self reponses
    nt, N,N = tensor.shape
    self_resps = np.zeros((nt,N), np.float64)
    for i in range(N):
        self_resps[:,i] = tensor[:,i,i]

    return self_resps

def Time2Peak(arr, timestep):
    """
    The time that links, nodes or the network need to reach maximal response.

    The function calculates the time-to-peak for either links, nodes or the
    whole network, depending on the input array given.
    - If 'arr' is a (nt,N,N) flow tensor, the output 'ttp_arr' will be an
    (N,N) matrix with the ttp between every pair of nodes.
    - If 'arr' is a (nt,N) temporal flow of the N nodes, the output 'ttp_arr'
    will be an array of length N, containing the ttp of all N nodes.
    - If 'arr' is an array of length nt (total network flow over time), 'ttp_arr'
    will be a scalar, indicating the time at which the whole-network flow peaks.

    Parameters
    ----------
    arr : ndarray of adaptive shape, according to the case.
        Temporal evolution of the flow. An array of optional shapes. Either
        (nt,N,N) for the pair-wise flows, shape (nt,N,N) for the in- or output
        flows of nodes, or a 1D array of length nt for the network flow.
    timestep : real valued number.
        Sampling time-step. This has to be the time-step employed to simulate
        the temporal evolution encoded in 'arr'.

    Returns
    -------
    ttp_arr : ndarray of variable rank
        The time(s) taken for links, nodes or the network to reach peak flow.
        Output shape depends on input.
    """

    # 0) CHECK THE USER INPUT
    ## TODO: Write a check to verify the curve has a real peak and decays after
    ## the peak. Raise a warning that maybe longer simulation is needed.
    if arr.shape == 3:
        io_helpers.validate_tensor(arr)

    # 1) Get the indices at which every element peaks
    ttp_arr = arr.argmax(axis=0)
    # 2) Identify disconnected pairs
    ttp_arr =np.where(ttp_arr==0, np.inf, ttp_arr)
    # 3) Convert into simulation time
    ttp_arr = timestep * ttp_arr

    return ttp_arr

def AreaUnderCurve(arr, timestep, timespan='alltime'):
    """
    The amount of response accumulated over time.

    The function calculates the area-under-the-curve for the response curves over
    time. It does so for all pair-wise interactions, for the nodes or for
    the whole network, depending on the input array given.

    - If 'arr' is a (nt,N,N) flow tensor, the output 'integral' will be an
    (N,N) matrix with the accumulated flow passed between every pair of nodes.
    - If 'arr' is a (nt,N) temporal flow of the N nodes, the output 'integral'
    will be an array of length N, containing the accumulated flow passed through
    all the nodes.
    - If 'arr' is an array of length nt (total network flow over time), 'integral'
    will be a scalar, indicating the total amount of flow that went through the
    whole network.

    Parameters
    ----------
    arr : ndarray of adaptive shape, according to the case.
        Temporal evolution of the flow. An array of shape nt x N x N for the
        flow of the links, an array of shape N X nt for the flow of the nodes,
        or a 1-dimensional array of length nt for the network flow.
    timestep : real valued number.
        Sampling time-step. This has to be the time-step employed to simulate
        the temporal evolution encoded in 'arr'.
    timespan : string, optional
        If timespan = 'alltime', the function calculates the area under the
        curve(s) along the whole time span (nt) that 'arr' contains, from t0 = 0
        to tfinal.
        If timespan = 'raise', the function calculates the area-under-the-
        curve from t0 = 0, to the time the flow(s) reach a peak value.
        If timespan = 'decay', it returns the area-under-the-curve for the
        time spanning from the time the flow peaks, until the end of the signal.

    Returns
    -------
    integral : ndarray of variable rank
        The accumulated response (area-under-the-curve) between pairs of nodes,
        by nodes or by the whole network, over a period of time.
    """
    # 0) CHECK THE USER INPUT
    ## TODO: Write a check to verify the curve has a real peak and decays after
    ## the peak. Raise a warning that maybe longer simulation is needed.
    if arr.shape == 3:
        io_helpers.validate_tensor(arr)

    # Validate options for optional variable 'timespan'
    caselist = ['alltime', 'raise', 'decay']
    if timespan not in caselist :
        raise ValueError( f"Optional parameter 'timespan' requires one of the following values: {str(caselist)}" )

    # 1) DO THE CALCULATIONS
    # 1.1) Easy case. Integrate area-under-the-curve along whole time interval
    if timespan == 'alltime':
        integral = timestep * arr.sum(axis=0)

    # 1.2) Integrate area-under-the-curve until or from the peak time
    else:
        # Get the temporal indices at which the flow(s) peak
        tpidx = arr.argmax(axis=0)

        # Initialise the final array
        tf_shape = arr.shape[1:]
        integral = np.zeros(tf_shape, np.float64)

        # Sum the flow(s) over time, only in the desired time interval
        nsteps = arr.shape[0]
        for t in range(1,nsteps):
            # Check if the flow at time t should be accounted for or ignored
            if timespan == 'raise':
                counts = np.where(t < tpidx, True, False)
            elif timespan == 'decay':
                counts = np.where(t < tpidx, False, True)
            # Sum the flow at the given iteration, if accepted
            integral += (counts * arr[t])

        # Finally, normalise the integral by the time-step
        integral *= timestep

    return integral





##
