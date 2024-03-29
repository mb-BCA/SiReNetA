# -*- coding: utf-8 -*-
# Copyright 2024, Gorka Zamora-López and Matthieu Gilson.
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
Calculation of pair-wise, temporal responses for different propagation models
=============================================================================

This module contains functions to calculate the temporal evolution of the
pair-wise reponses $R_{ij}(t)$ between nodes, due to initial stimuli.
The location and intensity of the perturbations can be defined by the users.
Default are set to all nodes receive stimulus of unit amplitude. Results for
$R_{ij}(t)$ are returned as a numpy array of rank-3 of shape (nt,N,N) where nt
is the number of time points and N the number of nodes.

Calculation of Jacobian matrices
--------------------------------
TransitionMatrix
    Returns the transition probability matrix for random walks.
JacobianMOU
    Calculates the Jacobian matrix for the MOU dynamic system.
LaplacianMatrix
    Calculates the graph Laplacian.

Generation of main tensors
--------------------------
Resp_DiscreteCascade
    Computes the pair-wise responses over time for the discrete cascade model.
Resp_RandomWalk
    Computes the pair-wise responses over time for the simple random walk model.
Resp_ContCascade
    Computes the pair-wise responses over time for the continuous cascade model.
Resp_LeakyCascade
    Computes the pair-wise responses over time for the leaky-cascade model.
Resp_ContDiffusion
    Computes the pair-wise responses over time for the linear diffusive model.


**Reference and Citation**

1) G. Zamora-Lopez and M. Gilson "An integrative dynamical perspective for graph
theory and the analysis of complex networks" arXiv:2307.02449 (2023).
DOI: `https://doi.org/10.48550/arXiv.2307.02449
<https://doi.org/10.48550/arXiv.2307.02449>`_

2) M. Gilson, N. Kouvaris, et al. "Network analysis of whole-brain fMRI
dynamics: A new framework based on dynamic communicability" NeuroImage 201,
116007 (2019).

3) M. Gilson, N. Kouvaris, G. Deco & G.Zamora-Lopez "Framework based on communi-
cability and flow to analyze complex networks" Phys. Rev. E 97, 052301 (2018).
"""

# Standard library imports

# Third party packages
import numpy as np
# import numpy.linalg
import scipy.linalg
# Local imports from sireneta
from . import io_helpers



## JACOBIAN MATRICES ###########################################################
def TransitionMatrix(con, rwcase='simple'):
    """Returns the transition probability matrix for random walks.

    TODO: ADD RANDOM WALK WITH TELEPORTATION

    - If rwcase='simple'
    Given a connectivity matrix A, where Aij represents the (weighted)
    connection from i to j, the transition probability matrix for a simple
    random walk is computed as Tij = Aij / deg(i), where deg(i) is the output
    (weighted) degree of node i.

    Parameters
    ----------
    con : ndarray (2d) of shape (N,N).
        The connectivity matrix of the network.
    rwcase : string, optional
        Default 'simple' returns the transition probability matrix for the
        simple random walk.

    Returns
    -------
    tp_matrix : ndarray of rank-2 and shape (N,N).
        The transition probability matrix.

    NOTE
    ----
    For now only the simple random walk is supported. Optional parameter
    available to cover different classes of random walks in future
    releases.
    """
    # 0) HANDLE AND CHECK THE INPUTS
    io_helpers.validate_con(con)
    if con.dtype != np.float64:
        con = con.astype(np.float64)
    N = len(con)

    caselist = ['simple']
    if rwcase not in caselist:
        raise ValueError( "Please enter one of accepted cases: %s" %str(caselist) )

    # 1) COMPUTE THE TRANSITION PROBABILITY MATRIX
    tp_matrix = con.copy().astype(np.float64)
    if rwcase=='simple':
        outdegree = con.sum(axis=1)
        for i in range(N):
            # Avoids NaN values in tp_matrix if node is disconnected
            if outdegree[i]:
                tp_matrix[i] = con[i] / outdegree[i]

    return tp_matrix

def Jacobian_LeakyCascade(con, tau):
    """Calculates the Jacobian matrix for the leaky-cascade dynamical system.

    Note: this is the same as the Ornstein-Uhlenbeck process on a network.

    TODO: RETHINK THE NAME OF THIS FUNCTION. MERGE DIFFERENT JACOBIAN GENERATOR
    FUNCTIONS INTO A SINGLE FUNCTION !?

    Parameters
    ----------
    con : ndarray (2d) of shape (N,N).
        The connectivity matrix of the network.
    tau : real value or ndarray (1d) of length N.
        The decay time-constants of the nodes. If a scalar value is entered,
        `tau = c`, then all nodes will be assigned the same value `tau[i] = 2`
        (identical nodes). If an 1d-array is entered, each node i is assigned
        decay time-constant `tau[i]`.

    Returns
    -------
    jac : ndarray (2d) of shape (N,N)
        The Jacobian matrix for the MOU dynamical system.
    """
    # 0) HANDLE AND CHECK THE INPUTS
    io_helpers.validate_con(con)
    N = len(con)
    tau = io_helpers.validate_tau(tau, N)

    # Ensure all arrays are of same dtype (np.float64)
    if con.dtype != np.float64:    con = con.astype(np.float64)
    if tau.dtype != np.float64:    tau = tau.astype(np.float64)

    # 1) CALCULATE THE JACOBIAN MATRIX
    jac = -1.0/tau * np.identity(N, dtype=np.float64) + con

    return jac

def LaplacianMatrix(con, normed=False):
    """Calculates the graph Laplacian.

    TODO: WRITE THE DESCRIPTION HERE

    Parameters
    ----------
    con : ndarray (2d) of shape (N,N).
        The connectivity matrix of the network.
    normed : boolean, optional
        If True, it returns the normalised graph Laplacian.

    Returns
    -------
    lap : ndarray (2d) of shape(N,N)
        The graph Laplacian matrix.
    """
    # 0) HANDLE AND CHECK THE INPUTS
    io_helpers.validate_con(con)
    if con.dtype != np.float64:
        con = con.astype(np.float64)
    N = len(con)

    # 1) CALCULATE THE GRAPH LAPLACIAN MATRIX
    outdegree = con.sum(axis=1)
    lap = - outdegree * np.identity(N, dtype=np.float64) + con

    if normed:
        for i in range(N):
            # Avoids NaN values in tp_matrix if node is disconnected
            if outdegree[i]:
                lap[i] /= outdegree[i]

    return lap


## GENERATION OF THE MAIN TENSORS #############################################
## DISCRETE-TIME CANONICAL MODELS _____________________________________________

# TODO: MAYBE, RETHING THE NAMING OF THESE FUNCTIONS ?

def Resp_DiscreteCascade(con, S0=1.0, tmax=10):
    """Computes the pair-wise responses over time for the discrete cascade model.

    Given a connectivity matrix A, where Aij represents the (weighted)
    connection from i to j, the response matrices Rij(t) encode the temporal
    response observed at node j due to a short stimulus applied on node i at
    time t=0.
    The discrete cascade is the simplest linear propagation model for
    DISCRETE VARIABLE and DISCRETE TIME in a network. It is represented by
    the following iterative equation:

            x(t+1) = A x(t)  .

    This system is NON-CONSERVATIVE and leads to DIVERGENT dynamics. If all
    entries of A are positive, e.g, A is a binary graph, the both the solutions
    x_i(t) and the responses Rij(t) rapidly explode.

    Parameters
    ----------
    con : ndarray (2d) of shape (N,N).
        The connectivity matrix of the network.
    S0 : scalar or ndarray (1d) of length N, optional
        Amplitude of the stimuli applied to nodes at time t = 0.
        If scalar value given, `S0 = c`, all nodes are initialised as `S0[i] = c`
        Default, `S0 = 1.0` represents a unit perturbation to all nodes.
        If a 1d-array is given, node i receives initial stimulus `S0[i]`.
    tmax : integer, optional
        The duration of the simulation, number of discrete time steps.

    Returns
    -------
    resp_matrices : ndarray (3d) of shape (tmax+1,N,N)
        Temporal evolution of the pair-wise responses. The first time point
        contains the matrix of inputs. Entries `resp_matrices[t,i,j]` represent
        the response of node j at time t, due to an initial perturbation on i.
     """
    # 0) HANDLE AND CHECK THE INPUTS
    io_helpers.validate_con(con)
    N = len(con)
    S0 = io_helpers.validate_S0(S0,N)

    if tmax <= 0.0: raise ValueError("'tmax' must be positive")

    # Ensure all arrays are of same dtype (np.float64)
    if con.dtype != np.float64:     con = con.astype(np.float64)
    if S0.dtype != np.float64:      S0 = S0.astype(np.float64)

    # 1) PREPARE FOR THE CALCULATIONS
    # Initialise the output array and enter the initial conditions
    nt = int(tmax) + 1
    resp_matrices = np.zeros((nt,N,N), dtype=np.float64)
    # Enter the initial conditions
    resp_matrices[0][np.diag_indices(N)] = S0

    # 2) COMPUTE THE PAIR-WISE RESPONSE MATRICES OVER TIME
    for t in range(1,nt):
        resp_matrices[t] = np.matmul(resp_matrices[t-1], con)

    return resp_matrices

def Resp_RandomWalk(con, S0=1, tmax=10):
    """Computes the pair-wise responses over time for the simple random walk model.

    Given a connectivity matrix A, where Aij represents the (weighted)
    connection from i to j, the transition probability matrix is computed as
    Tij = Aij / deg(i), where deg(i) is the output (weighted) degree of node i.
    The response matrices Rij(t) encode the temporal response observed at
    node j due to a short stimulus applied on node i at time t=0.
    The random walk is the simplest linear propagation model for DISCRETE
    VARIABLE and DISCRETE TIME in a network. It is represented by the following
    iterative equation:

            x(t+1) = T x(t) .

    This system is CONSERVATIVE and leads to CONVERGENT dynamics. At any time
    t > 0 the number of walkers (or agents) found in the network is the same
    as the number of walkers initially seeded.

    Parameters
    ----------
    con : ndarray (2d) of shape (N,N).
        The connectivity matrix of the network.
    S0 : scalar or ndarray (1d) of length N, optional
        Number of walkers seed at every node, at time t = 0. If scalar value
        given, `S0 = c`, all nodes are initialised as `S0[i] = c`.
        Default, `S0 = 1.0` represents one agent per node. If a 1d-array is
        given, node i starts with `S0[i]` walkers.
    tmax : integer, optional
        The duration of the simulation, number of discrete time steps.

    Returns
    -------
    resp_matrices : ndarray (3d) of shape (tmax+1,N,N)
        Temporal evolution of the pair-wise responses. The first time point
        contains the matrix of inputs. Entry `resp_matrices[t,i,j]` is the
        response of node j at time t, due to the initial perturbation on i.
    """
    # 0) HANDLE AND CHECK THE INPUTS
    io_helpers.validate_con(con)
    N = len(con)
    S0 = io_helpers.validate_S0(S0,N)

    if tmax <= 0.0: raise ValueError("'tmax' must be positive")

    # Ensure all arrays are of same dtype (np.float64)
    if con.dtype != np.float64:     con = con.astype(np.float64)
    if S0.dtype != np.float64:      S0 = S0.astype(np.float64)

    # 1) CALCULATE THE RESPONSE MATRICES
    # Define the transition probability matrix
    tpmatrix = TransitionMatrix(con, rwcase='simple')
    # Initialise the output array and enter the initial conditions
    nt = int(tmax) + 1
    resp_matrices = np.zeros((nt,N,N), dtype=np.float64)
    # Enter the initial conditions
    resp_matrices[0][np.diag_indices(N)] = S0

    # 2) COMPUTE THE PAIR-WISE RESPONSE MATRICES OVER TIME
    for t in range(1,nt):
        resp_matrices[t] = np.matmul(resp_matrices[t-1], tpmatrix)

    return resp_matrices



## CONTINUOUS-TIME CANONICAL MODELS ____________________________________________
def Resp_ContCascade(con, S0=1.0, tmax=10, timestep=0.1):
    """Computes the pair-wise responses over time for the continuous cascade model.

    TODO: SHALL WE ALLOW 'S0' TO BE A MATRIX OF (POSSIBLY CORRELATED) GAUSSIAN
    WHITE NOISE, AS ORIGINALLY FOR THE MOU ?

    Given a connectivity matrix A, where Aij represents the (weighted)
    connection from i to j, the response matrices Rij(t) encode the temporal
    response observed at node j due to a short stimulus applied on node i at
    time t=0.
    The continuous-cascade is the simplest linear propagation model for
    CONTINUOUS VARIABLE and CONTINUOUS TIME in a network. It is represented by
    the following differential equation:

            xdot(t) = A x(t) .

    This system is NON-CONSERVATIVE and leads to DIVERGENT dynamics. If all
    entries of A are positive, e.g, A is a binary graph, the both the solutions
    x_i(t) and the responses Rij(t) rapidly explode.

    Parameters
    ----------
    con : ndarray (2d) of shape (N,N).
        The connectivity matrix of the network.
    S0 : scalar or ndarray (1d) of length N, optional
        Amplitude of the stimuli applied to nodes at time t = 0.
        If scalar value given, `S0 = c`, all nodes are initialised as `S0[i] = c`
        Default, `S0 = 1.0` represents a unit perturbation to all nodes.
        If a 1d-array is given, node i receives initial stimulus `S0[i]`.
    tmax : scalar, optional
        Duration of the simulation, arbitrary time units.
    timestep : scalar, optional
        Temporal step (resolution) between consecutive calculations of responses.

    Returns
    -------
    resp_matrices : ndarray (3d) of shape (tmax+1,N,N)
        Temporal evolution of the pair-wise responses. Entries
        `resp_matrices[t,i,j]` represent the response of node j at time t, due
        to an initial perturbation on i.

    NOTE
    ----
    Simulation runs from t=0 to t=tmax, in sampled `timestep` apart. Thus,
    simulation steps go from it=0 to it=nt, where `nt = int(tmax*timestep) + 1`
    is the total number of time samples (number of response matrices calculated).
    Get the sampled time points as `tpoints = np.arange(0,tmax+timestep,timestep)`
    """
    # 0) HANDLE AND CHECK THE INPUTS
    io_helpers.validate_con(con)
    N = len(con)
    S0 = io_helpers.validate_S0(S0,N)

    if tmax <= 0.0: raise ValueError("'tmax' must be positive")
    if timestep <= 0.0: raise ValueError( "'timestep' must be positive")
    if timestep >= tmax: raise ValueError("'timestep' must be smaller than 'tmax'")

    # Ensure all arrays are of same dtype (float64)
    if con.dtype != np.float64:     con = con.astype(np.float64)
    if S0.dtype != np.float64:      S0 = S0.astype(np.float64)

    # 1) PREPARE FOR THE CALCULATIONS
    # Initialise the output array and enter the initial conditions
    nt = int(tmax / timestep) + 1
    resp_matrices = np.zeros((nt,N,N), dtype=np.float64 )
    # Convert the stimuli into a matrix
    if S0.ndim in [0,1]:
        S0mat = S0 * np.identity(N, dtype=np.float64)
        # S0mat = scipy.linalg.sqrtm(S0mat)

    # 2) COMPUTE THE PAIR-WISE RESPONSE MATRICES OVER TIME
    for it in range(nt):
        t = it * timestep
        # Calculate the Green's function at time t.
        greenf_t = scipy.linalg.expm(con * t)
        # Calculate the pair-wise responses at time t.
        resp_matrices[it] = np.matmul(S0mat, greenf_t)

    return resp_matrices

def Resp_LeakyCascade(con, S0=1.0, tau=1.0, tmax=10, timestep=0.1,
                                                case='regressed', normed=False):
    """Computes the pair-wise responses over time for the leaky-cascade model.

    TODO: DECIDE ABOUT THE 'normed' PARAMETER.
    TODO: SHALL WE ALLOW 'S0' TO BE A MATRIX OF (POSSIBLY CORRELATED) GAUSSIAN
    WHITE NOISE, AS ORIGINALLY FOR THE MOU ?

    Given a connectivity matrix A, where Aij represents the (weighted)
    connection from i to j, the response matrices Rij(t) encode the temporal
    response observed at node j due to a short stimulus applied on node i at
    time t=0.
    The leaky-cascade is the time-continuous and variable-continuous linear
    propagation model represented by the following differential equation:

            xdot(t) = - x(t) / tau + A x(t).

    where tau is a leakage time-constant for a dissipation of the flows through
    the nodes. This model is reminiscent of the multivariate Ornstein-Uhlenbeck
    process, when additive Gaussian white noise is included.
    Given λmax is the largest eigenvalue of the (positive definite) matrix A, then
    - if tau < tau_max = 1 / λmax, then the leakage term dominates in the long
    time and the solutions for all nodes converge to zero.
    - If tau = tau_max, all nodes converge to x_i(t) = 1.
    - And, if tau < tau_max, then time-courses xdot(t) grow exponentially fast.

    Parameters
    ----------
    con : ndarray (2d) of shape (N,N).
        The connectivity matrix of the network.
    S0 : scalar or ndarray (1d) of length N, optional
        Amplitude of the stimuli applied to nodes at time t = 0.
        If scalar value given, `S0 = c`, all nodes are initialised as `S0[i] = c`
        Default, `S0 = 1.0` represents a unit perturbation to all nodes.
        If a 1d-array is given, node i receives initial stimulus `S0[i]`.
    tau : real value or ndarray (1d) of length N, optional
        The decay time-constants of the nodes. If a scalar value is entered,
        `tau = c`, then all nodes will be assigned the same value `tau[i] = 2`
        (identical nodes). If an 1d-array is entered, each node i is assigned
        decay time-constant `tau[i]`. Default `tau = 1.0` is probably too large
        for most real networks and will diverge. If so, enter a `tau` smaller
        than the spectral diameter (λ_max) of `con`.
    tmax : scalar, optional
        Duration of the simulation, arbitrary time units.
    timestep : scalar, optional
        Temporal step (resolution) between consecutive calculations of responses.
    case : string (optional)
        - 'full' Computes the responses a given by the Green's function of the
        Jacobian of the system: e^{Jt} with J = - I / tau + A.
        - 'intrinsic' Computes the trivial responses due to the leakage through
        the nodes: e^{J0t} with J0 = I / tau. This represents a 'null' case where
        the network is empty (has no links) and the initial inputs passively
        leak through the nodes without propagating.
        - 'regressed' Computes the network responses due to the presence of the
        links: e^{Jt} - e^{J0t}. That is, the 'full' response minus the passive,
        'intrinsic' leakage.
    normed : boolean (optional)
        DEPRECATED. If True, normalises the tensor by a scaling factor, to make networks
        of different size comparable.

    Returns
    -------
    resp_matrices : ndarray (3d) of shape (tmax+1,N,N)
        Temporal evolution of the pair-wise responses. Entries
        `resp_matrices[t,i,j]` represent the response of node j at time t, due
        to an initial perturbation on i.

    NOTE
    ----
    Simulation runs from t=0 to t=tmax, in sampled `timestep` apart. Thus,
    simulation steps go from it=0 to it=nt, where `nt = int(tmax*timestep) + 1`
    is the total number of time samples (number of response matrices calculated).
    Get the sampled time points as `tpoints = np.arange(0,tmax+timestep,timestep)`.
    """
    # 0) HANDLE AND CHECK THE INPUTS
    io_helpers.validate_con(con)
    N = len(con)
    S0 = io_helpers.validate_S0(S0,N)
    tau = io_helpers.validate_tau(tau, N)

    if tmax <= 0.0: raise ValueError("'tmax' must be positive")
    if timestep <= 0.0: raise ValueError( "'timestep' must be positive")
    if timestep >= tmax: raise ValueError("'timestep' must be smaller than 'tmax'")

    # Ensure all arrays are of same dtype (np.float64)
    if con.dtype != np.float64:     con = con.astype(np.float64)
    if S0.dtype != np.float64:      S0 = S0.astype(np.float64)
    if tau.dtype != np.float64:     tau = tau.astype(np.float64)

    caselist = ['regressed', 'full', 'intrinsic']
    if case not in caselist:
        raise ValueError( "Please enter one of accepted cases: %s" %str(caselist) )

    # 1) PREPARE FOR THE CALCULATIONS
    # Initialise the output array and enter the initial conditions
    nt = int(tmax / timestep) + 1
    resp_matrices = np.zeros((nt,N,N), dtype=np.float64 )
    # Compute the Jacobian matrices
    jac = Jacobian_LeakyCascade(con, tau)
    jacdiag = np.diagonal(jac)
    # Convert the stimuli into a matrix
    if S0.ndim in [0,1]:
        S0mat = S0 * np.identity(N, dtype=np.float64)
    # S0mat = scipy.linalg.sqrtm(S0mat)

    if case == 'full':
        for it in range(nt):
            t = it * timestep
            # Calculate the Green's function at time t
            green_t = scipy.linalg.expm(jac * t)
            # Calculate the pair-wise responses at time t
            resp_matrices[it] = np.matmul( S0mat, green_t )

    elif case == 'intrinsic':
        for it in range(nt):
            t = it * timestep
            # Calculate the Green's function (of an empty graph) at time t
            greendiag_t = np.diag( np.exp(jacdiag * t) )
            # Calculate the pair-wise responses at time t
            resp_matrices[it] = np.matmul( S0mat, greendiag_t)

    elif case == 'regressed':
        for it in range(nt):
            t = it * timestep
            # Calculate the Green's function (of the full system) at time t
            green_t = scipy.linalg.expm(jac * t)
            # Calculate the Green's function (of an empty graph) at time t
            greendiag_t = np.diag( np.exp(jacdiag * t) )
            # Calculate the pair-wise responses at time t
            resp_matrices[it] = np.matmul( S0mat, green_t - greendiag_t )

    # 2.2) Normalise by the scaling factor
    if normed:
        scaling_factor = np.abs(1./jacdiag).sum()
        resp_matrices /= scaling_factor

    return resp_matrices

def Resp_ContDiffusion(con, S0=1.0, tmax=10, timestep=0.1,
                                                case='regressed', normed=False):
    """Computes the pair-wise responses over time for the linear diffusive model.

    TODO: SHALL WE ALLOW 'S0' TO BE A MATRIX OF (POSSIBLY CORRELATED) GAUSSIAN
    WHITE NOISE, AS ORIGINALLY FOR THE MOU ?

    Given a connectivity matrix A, where Aij represents the (weighted)
    connection from i to j, the response matrices Rij(t) encode the temporal
    response observed at node j due to a short stimulus applied on node i at
    time t=0.
    The continuous diffusion is the simplest time-continuous and variable-
    continuous linear propagation model with diffusive coupling. It is
    represented by the following differential equation:

            xdot(t) = -D x(t) + A x(t) = L x(t).

    where D is a diagonal matrix containing the (output) degrees of the nodes
    in the diagonal, and L = -D + A is the graph Laplacian matrix. This model
    is reminiscent of the continuous leaky cascade but considering that
    tau(i) = 1./deg(i). As such, the input and the leaked flows are balanced
    at each node.

    Parameters
    ----------
    con : ndarray (2d) of shape (N,N).
        The connectivity matrix of the network.
    S0 : scalar or ndarray (1d) of length N, optional
        Amplitude of the stimuli applied to nodes at time t = 0.
        If scalar value given, `S0 = c`, all nodes are initialised as `S0[i] = c`
        Default, `S0 = 1.0` represents a unit perturbation to all nodes.
        If a 1d-array is given, node i receives initial stimulus `S0[i]`.
    tmax : scalar, optional
        Duration of the simulation, arbitrary time units.
    timestep : scalar, optional
        Temporal step (resolution) between consecutive calculations of responses.
    case : string (optional)
        - 'full' Computes the responses a given by the Green's function of the
        Jacobian of the system: e^{Jt} with J = - I / tau + A.
        - 'intrinsic' Computes the trivial responses due to the leakage through
        the nodes: e^{J0t} with J0 = I / tau. This represents a 'null' case where
        the network is empty (has no links) and the initial inputs passively
        leak through the nodes without propagating.
        - 'regressed' Computes the network responses due to the presence of the
        links: e^{Jt} - e^{J0t}. That is, the 'full' response minus the passive,
        'intrinsic' leakage.
    normed : boolean (optional)
        If True, employs the normalised graph Laplacian L' = D^-1 L.

    Returns
    -------
    resp_matrices : ndarray (3d) of shape (tmax+1,N,N)
        Temporal evolution of the pair-wise responses. Entries
        `resp_matrices[t,i,j]` represent the response of node j at time t, due
        to an initial perturbation on i.

    NOTE
    ----
    Simulation runs from t=0 to t=tmax, in sampled `timestep` apart. Thus,
    simulation steps go from it=0 to it=nt, where `nt = int(tmax*timestep) + 1`
    is the total number of time samples (number of response matrices calculated).
    Get the sampled time points as `tpoints = np.arange(0,tmax+timestep,timestep)`.
    """
    # 0) HANDLE AND CHECK THE INPUTS
    io_helpers.validate_con(con)
    N = len(con)
    S0 = io_helpers.validate_S0(S0,N)

    if tmax <= 0.0: raise ValueError("'tmax' must be positive")
    if timestep <= 0.0: raise ValueError( "'timestep' must be positive")
    if timestep >= tmax: raise ValueError("'timestep' must be smaller than 'tmax'")

    # Ensure all arrays are of same dtype (np.float64)
    if con.dtype != np.float64:     con = con.astype(np.float64)
    if S0.dtype != np.float64:      S0 = S0.astype(np.float64)

    caselist = ['regressed', 'full', 'intrinsic']
    if case not in caselist:
        raise ValueError( "Please enter one of accepted cases: %s" %str(caselist) )

    # NOTE: The graph Laplacian is the Jacobian matrix of the linear propagation
    # model based with diffusive coupling. Hence, the following codes is the
    # same as for the Leaky Cascade, only that here we call LaplacianMatrix()
    # instead of JacobianMOU().

    # 1) PREPARE FOR THE CALCULATIONS
    # Initialise the output array and enter the initial conditions
    nt = int(tmax / timestep) + 1
    resp_matrices = np.zeros((nt,N,N), dtype=np.float64 )
    # Compute the Jacobian matrices
    jac = LaplacianMatrix(con, normed=normed)
    jacdiag = np.diagonal(jac)
    # Convert the stimuli into a matrix
    if S0.ndim in [0,1]:
        S0mat = S0 * np.identity(N, dtype=np.float64)
    # S0mat = scipy.linalg.sqrtm(S0mat)

    if case == 'full':
        for it in range(nt):
            t = it * timestep
            # Calculate the Green's function at time t
            green_t = scipy.linalg.expm(jac * t)
            # Calculate the pair-wise responses at time t
            resp_matrices[it] = np.matmul( S0mat, green_t )

    elif case == 'intrinsic':
        for it in range(nt):
            t = it * timestep
            # Calculate the Green's function (of an empty graph) at time t
            greendiag_t = np.diag( np.exp(jacdiag * t) )
            # Calculate the pair-wise responses at time t
            resp_matrices[it] = np.matmul( S0mat, greendiag_t)

    elif case == 'regressed':
        for it in range(nt):
            t = it * timestep
            # Calculate the Green's function (of the full system) at time t
            green_t = scipy.linalg.expm(jac * t)
            # Calculate the Green's function (of an empty graph) at time t
            greendiag_t = np.diag( np.exp(jacdiag * t) )
            # Calculate the pair-wise responses at time t
            resp_matrices[it] = np.matmul( S0mat, green_t - greendiag_t )

    return resp_matrices



##
