# Roadmap towards an object-based version of *SiReNetA* 

Version 2 of ReNetA should be an object oriented library because we want to be able to compute the response matrices for different canonical models. The user should be able to call just one function to generate the response matrices, and one type of object to store that information, from which the user should be able to compute further metrics. Therefore, the library should be based around a core object (class) for the tensor containing the response matrices, *ResponseMatrices*. This object should have the following attributes and methods.

### General TODO

- Write pseudocode of how we would like a typeical workflow for the Object-Oriented version. See [v2_PseudoCode.ipynb](v2_PseudoCode.ipynb). 
- Identify attributes that should be locked from further changes. E.g., each object is bound, at creation time, to one canonical model. User cannot change the canonical model of the object later on. User would need to create another object for the same network and a different propagation model.

### Which attributes should the *ResponseMatrices* object have

General attributes:

- `canmod` : The canonical propagation model. **Unmutable**.
- `timetype` : Indicator for time discrete or time continuous. **Unmutable**, fixed by `canmod`.
- `con` : The connectivity matrix. **Unmutable**.
- `N` : The number of nodes. Extracted from `con`, not user input.
- `directed` : Whether `con` is directed or not. Extracted from `con`, not user input.
- `labels` : The "names" of the *N* nodes, if any given by the user. Optional attribute, not needed for calculations.
- Temporal constraints of the simulation [`t0`, `tfinal`, `dt`]
- `tpoints` : ndarray of the time-points in the simulation. Calculated from `tfinal` and `dt`, not user input.
- `S0` : The initial amplitude of stimulus at every node.
- `data` : The "simulated" response tensor of response matrices. Shape = (tmax//dt+1, N, N). Alternatively, it could be named `arr_tij`. Then, the internal arrays for node-wise responses would be `arr_ti` and for global response `arr_t`.
- Eigenvalues and eigenvectors of the connectivity matrix. (?? Not sure of this one.)
- `evmax` : Largest eigenvalue of `con`. Maybe not needed?



Model specific attributes (DISCRETE CASCADE):

- `timestep` = 1, by default. **_Not mutable_**.


Model specific attributes (RANDOM WALK):

- `timestep` = 1, by default. **_Not mutable_**.


Model specific attributes (LEAKY CASCADE):

- `taumax` : largest possible leakage rate  where $\tau\_{max} = 1 / \lambda_{max}$.
- The leakage rate used for the "simulation", $\tau$.
- Stimulus `S0` can be a matrix, the Sigma matrix used for the MOU. (((We need a check that Sigma is properly normalised. )))


Model specific attributes (RANDOM WALK WITH TELEPORTATION)

- Has not been implemented yet into *SiReNetA*.
- `v` : The personalization vector.



<br>

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

