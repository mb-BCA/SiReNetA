# Roadmap towards an object-based version of *SiReNetA* 

Version 2 of ReNetA should be an object oriented library because we want to be able to compute the response matrices for different canonical models. The user should be able to call just one function to generate the response matrices, and one type of object to store that information, from which the user should be able to compute further metrics. Therefore, the library should be based around a core object (class) for the tensor containing the response matrices, *ResponseMatrices*. This object should have the following attributes and methods.

### General TODO

- Write pseudocode of how we would like a typeical workflow for the Object-Oriented version. See [v2_PseudoCode.ipynb](v2_PseudoCode.ipynb). 
- Identify attributes that should be locked from further changes. E.g., each object is bound, at creation time, to one canonical model. User cannot change the canonical model of the object later on. User would need to create another object for the same network and a different propagation model.

### Which attributes should the *ResponseMatrices* object have

General attributes:

- The canonical propagation model, e.g. `canonicalmodel`. Unmutable.
- Indicator for time discrete or continuous, e.g. `timetype`. Unmutable, fixed by `canonicalmodel`.
- The connectivity matrix `con`
- Number of nodes, N.
- Temporal constraints of the simulation [`t0`, `tfinal`, `dt`]
- Array of time-points in the simulation time-scale: `tpoints`.
- Initial conditions `X0`
- The simulated tensor of shape (N, N, nt) with the flow.
- Eigenvalues and eigenvectors of the connectivity matrix. (?? Not sure of this one.)
- Largest eigenvalue `evmax`.


Attributes only for specific canonical models:

- Largest possible leakage rate `taumax`, where $\tau\_{max} = 1 / \lambda_{max}$.
- The leakage rate used for the simulation, $\tau$.
- The Sigma matrix (i.e., the matrix of the initial inputs), which is equal to input time (e.g. InstantaneousInput so far, StationaryInput for MOU). (((We need a check that Sigma is properly normalised. )))
- The "personalization vector" for the random walkers with teleportation.




### Which methods should the *ResponseMatrices* object have

I guess these are all the functions in modules *core.py*, *metrics.py* and *netmodels.py*.


### Questions

- What to do for example when we calculate surrogates? There will be plenty of redundant information wasting RAM memory, if there is an object for each surrogate realization. 
- Besides the object class for the response matrices, there should be a second class for the time-series simulated for the nodes **x**(t). 


### Parallelization / Optimization

- Jax for faster computation of e.g. expm (`from jax.scipy.linalg import expm` instead of `from scipy.linalg import expm`)
- joblib for parallelization on cluster? distribute 1 job per integration time point for tensor computation


### Testing

- use pytest for unitary testing of each function / object
- organize data for testing and testing code within scripts

