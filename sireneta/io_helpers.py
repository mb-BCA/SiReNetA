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
Helper functions for handling inputs and checks
===============================================

This module contains functions to help IO operations, specially to carry the
validation checks for the user inputs (parameters to functions) and ensure all
relevant arrays are given in the correct data type.

Input handling functions
------------------------
validate_con
    Check if user input `con` is a suitable connectivity matrix.
validate_X0
    Check if user input `X0` is a suitable array of initial conditions.
validate_S0
    Check if user input `S0`is a suitable array of stimulus amplitudes.
validate_S0matrix
    Check if user input `Somatrix` is a suitable matrix of stimulus correlations.
validate_tau
    Check if user input `tau` is a suitable object of time-constants.
validate_noise
    Check if user input `noise` is a suitable object for noise.
"""

# Standard library imports
import numbers
# Third party packages
import numpy as np
import numpy.random



## INPUT HANDLING FUNCTIONS ###################################################
def validate_con(a):
    """
    This function returns nothing. It only checks the connectivity matrix given
    by the user.
    """
    # Make sure 'con' is a numpy array, of np.float64 dtype
    if isinstance(a, np.ndarray): pass
    else:
        raise TypeError( "'con' must be numpy array, but %s found" %type(a) )

    # Make sure 'con' is a 2D array
    conshape = np.shape(a)
    if np.ndim(a)==2 and conshape[0]==conshape[1]: pass
    else:
        raise ValueError( "'con' must be a square matrix, but shape %s found" %str(np.shape(a)) )

def validate_X0(a, N):
    """
    This function checks the initial conditions given by the user. If `X0` is
    a scalar, returns it as a numpy array to the handling function.
    """
    # Make sure 'X0' is a numpy array, of np.float64 dtype
    if isinstance(a, numbers.Number) and type(a) != bool:
        a = a * np.ones(N, np.float64)
    elif isinstance(a, np.ndarray): pass
    else:
        raise TypeError(
        "'X0' must be either scalar or numpy array, but %s found" %type(a) )

    # Make sure 'X0' is a 1D array
    if np.ndim(a) != 1:
        raise ValueError(
        "'X0' must be either scalar or 1-dimensional of length N, but shape %s found"
        %str(np.shape(a)) )

    return a

def validate_S0(a, N):
    """
    This function checks the initial stimulus amplitudes given by the user.
    If `S0` is a scalar, returns it as a numpy array to the handling function.
    """
    # Make sure 'S0' is a numpy array, of np.float64 dtype
    if isinstance(a, numbers.Number) and type(a) != bool:
        a = a * np.ones(N, np.float64)
    elif isinstance(a, np.ndarray): pass
    else:
        raise TypeError(
        "'S0' must be either scalar or numpy array, but %s found" %type(a) )

    # Make sure 'S0' is a 1D array
    if np.ndim(a) != 1:
        raise ValueError(
        "'S0' must be either scalar or 1-dimensional of length N, but shape %s found"
        %str(np.shape(a)) )

    return a

def validate_S0matrix(a, N):
    """
    This function checks the matrix of initial stimulus correlations given by
    the user. If `S0matrix` is a scalar or a 1D array, returns it as a 2D array
    to the handling function.
    """
    zero_tol = 1e-12

    # Check if 'S0' is a number or a numpy array
    if isinstance(a, numbers.Number) and type(a) != bool:
        if a < -zero_tol:
            raise ValueError("'S0' as numerical value entered, must be positive")
        else:
            a = a * np.identity(N, np.float64)
    elif isinstance(a, np.ndarray): pass
    else:
        raise TypeError(
        "'S0' must be either scalar or numpy array, but %s found" %type(a) )

    # If 'S0' is an array, convert input to a matrix
    if np.ndim(a)== 1:
        amin = a.min()
        if amin < -zero_tol:
            raise ValueError("'S0' as 1d-array entered, all values must be positive")
        else:
            a = a * np.identity(N, np.float64)
    # If 'S0' is a 2D array, make sure it is a square matrix
    elif np.ndim(a)==2:
        conshape = np.shape(a)
        if conshape[0]!=conshape[1]:
            raise ValueError( "'S0' not a square matrix, shape %s found" %str(np.shape(a)) )
    else:
        raise ValueError( "'S0' not a square matrix, shape %s found" %str(np.shape(a)) )

    # Finally, make sure all eigenvalues are positive
    evs = numpy.linalg.eigvals(a)
    evmin = evs.min()
    if evmin < -zero_tol:
        raise ValueError(
        "'S0' not a correlation matrix, at least one negative eigenvalue found: %f." %evmin )

    return a

def validate_tau(a, N):
    """
    This function checks the set of time-constants given by the user. If `tau` is
    a scalar, returns it as a numpy array to the handling function.
    """
    # Make sure 'tau' is a numpy array, of np.float64 dtype
    if isinstance(a, numbers.Number) and type(a) != bool:
        a = a * np.ones(N, np.float64)
    elif isinstance(a, np.ndarray): pass
    else:
        raise TypeError(
        "'tau' must be either scalar or numpy array, but %s found" %type(a) )

    # Make sure 'tau' is a 1D array
    if np.ndim(a) != 1:
        raise ValueError(
        "'tau' must be either scalar or 1-dimensional of length N, but shape %s found"
        %str(np.shape(a)) )

    return a

def validate_noise(a, N, tmax, timestep):
    """
    This function checks the noise object given by the user. If `noise` is
    a scalar, returns it as a numpy array to the handling function.
    """
    # When nothing is given by user, skip noise generation
    if a is None:
        pass
    # When 'noise' is a scalar ...
    elif isinstance(a, numbers.Number):
        if not a:
            # If zero or False, do nothing
            a = None
            pass
        elif a < 0:
            # 'noise' must be positive
            raise ValueError( "'noise' amplitude must be positive, %f found" %a )
        else:
            # If positive scalar given, generate the array for the noise
            namp = a
            nnorm = np.sqrt(2.0 * namp * timestep)
            nt = round(tmax / timestep) + 1
            a = nnorm * numpy.random.randn(nt, N)
    # Make sure 'noise' is a numpy array, of np.float64 dtype
    elif isinstance(a, np.ndarray):
        pass
    else:
        raise TypeError(
        "'noise' must be None, scalar or numpy array, but %s found" %type(a) )

    # Make sure 'noise' is returned a 2D array
    if a is not None and np.ndim(a) != 2:
        raise ValueError(
        "'noise' must be a 2-dimensional, but shape %s found" %str(np.shape(a)) )

    return a



# def validate_scalar_1darr(a, N):
#     """
#     """
#     # Get the global name of parameter 'a'
#     ## This doesn't work. I dunno why because it work in other examples :(
#     ## If working, I could use it for different arrays instead of having
#     ## on function per array (e.g., X0 and tau)
#     id_a = id(a)
#     localdict = locals()
#     for name in localdict.keys():
#         if id(localdict[name]) == id_a:
#             lname_a = name
#
#     globaldict = globals()
#     prround(len(globaldict))
#     for name in globaldict.keys():
#         if id(globaldict[name]) == id_a:
#             gname_a = name
#
#     prround(lname_a, gname_a)
#
#     # Make sure 'a' is a numpy array, of np.float64 dtype
#     if isinstance(a, numbers.Number):
#         a = a * np.ones(N, np.float64)
#     if isinstance(a, np.ndarray): pass
#     elif isinstance(a, (list,tuple)):
#         a = np.array(a, np.float64)
#     else:
#         raise TypeError( "'%s' must be either scalar or array-like (ndarray, list, tuple)" %gname_a)
#
#     # Make sure 'a' is a 1D array
#     if np.ndim(a) != 1:
#         raise ValueError( "'%s' must be a 1-dimensional array of length N" %gname_a)
#
#     return a




##
