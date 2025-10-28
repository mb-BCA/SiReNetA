# Roadmap for first version of SiReNetA

Given that in the new Perspective Article in which we call for a model-based network analyses using different canonical models, we need a complete new roadmap for v2 of the library. We need to provide two new things:

1. The functions to calculate the response matrices (tensors) for the different canonical models, regardless of whether they are time-discrete or time-continuous.
2. A new module with the code to simulate and get the temporal solutions **x**(t) for the nodes for each canonical model.

I guess that version v1 of *SiReNetA* is going to be transparent from the point of view of the code but will have several redundant functions with similar names, specially those to generate the response matrices under the different models. We should make v2 an object-oriented library which would reduce the number of "functions" and names the users need to remember. That is, only one function (with different options, of course) to generate the tensor of the response matrices, and another function to compute the solutions **x**(t). Those two functions will then be called specifying the canonical model and the particular parameters if the model needs that. E.g., the decay time-constant for the leaky-cascade (MOU).

In any case, v2 has to be a clean and coherent library such that the transition to an object-oriented version should be as smooth as possible.


### TODO list

- Revise *ALL* strings for adequate (more modern Python) styles. The f'…' formatting should particularly be useful for all **warnings** and **error messages**. So, specifically to revise:
    - ~~The error messages on *io_helpers.py* module~~ !! 
    - Security checks at the beginning of every function.
- Write the **stringdoc** documentation for:
    - ~~Beginning of *io_helpers.py* module~~.
    - ~~All funtions in *io_helpers.py* module~~.
    - …
- Add functions to *responses.py* module to compute R(t) for the different models:
	- ~~Unify the functions for the MOU case into one function~~.
	- ~~R(t) for the constinuous cascade~~.
	- ~~R(t) for the discrete cascade~~.
	- ~~R(t) for the random walks~~.
	- ~~R(t) for the continuous diffusion~~.

- In module *metrics.py* module:
	- ~~To return the peak flows~~.
	- ~~Add a function to extract and study the evolution of the self-interactions~~. DONE,  see function `metrics.SelfResponses()`.
	- ~~Function `Time2Peak()` should return `np.inf` for those pair-wise elements when there is no input in a node. Now, it returns zeros in those cases~~.
	- ~~Add option to remove diagonal elements of tensor, but keep default for NodeResponses as summing all incoming/outgoing interactions including self~~. 
	- For response tensors $R(t)$, add validation they converged to zero. Send warning otherwirse, recommending to run longer simulation.
	- For `AreaUnderCurve()` function, send warning if tensor values are farther than zero for a given tolerance. At this moment, it is the user's responsability to guarantee that all the curves have decayed reasonably well. So, if the responses didn't properly decay, the function should return a warning recommending to run longer simulations.)
	- (FOR LATER) Same for function `Time2Decay()` and/or `Time2Convergence()`. Should send warning when it returns the duration of the simulation in those cases???
	- Add a function to estimate time-to-threshold. For models that diverge. This is an extension of the graph distance for binary network where threshold = 1 should be the default (for discrete cascade).
	- Can / shall we add function to estimate the "Markov time" distance/centrality, as in Arnaudon et al., Phys. Rev. Research (2020) ?

- ~~What to do about the `sigma` parameter that we only have for the MOU?~~ It expects a matrix, not a vector of input amplitudes to the nodes.
	- THE DECISION: For now `sigma` is called S0 with default value `S0=1`. That computes the canonical case with unit input at all nodes. `S0` is and optional parameter that accepts a number or a vector. Later, we may think of allowing a matrix again. But first, we must understand whether that makes sense for the three continuous canonical models. 
	- Two functions were implemented `Resp_LeakyCascade()` and `Resp_OrnsteinUhlenck()`, with the first taking only a vector `S0` as input and the second expecting the covariance matrix.
- Double-check and validate function `Resp_OrnsteinUhlenbeck()`. Unfinished function. E.g., input parameter is `S0` instead of `S0mat` which is internally checked … Compare to main function from *NetDynFlow* package.
- **ACHTUNG !!** Double check the normalization of Gaussian noise (depending of time-step) in *simulate.py*. It seems the variance of the results is ~2x the one it should (??)

- ~~Include a *netmodels.py* module for generating networks and surrogates~~. ALTERNATIVE: We don't need a module for this, we could just have all those generator functions in *GAlib** and import. BETTER OPTION, module *netmodels.py* could import and wrap the functions in GAlib. Include the followong functions:
	- ~~In spatially embedded networks, a function to assign the stronger links to the closest nodes~~. See function `SpatialWeightSorting()`.
	- Weighted ring lattice, with stronger weights between neighbouring nodes (model by Muldoon et al., 2016)
    - Check and validate the network generation functions in *netmodels.py*.


- ~~Add security checks at the beginning of all functions~~.

- Finish test normalizations: (i) Eigenvalue, (ii) total weight, (iii) same inputs.

- Make sure of docstrings are good (and homogeneous) in all modules.



### Finished  (OLDER LIST)

- Include a new module named `simulations.py` containing the code to simulate the network under the different canonical models and return the temporal solutions **x**(t) for the nodes.
- Add functions to the *metrics.py* module:
	- Dynamic distance (time-to-peak). For the links, for the nodes and for the whole network. It should be best to write a single function that can return the adequate results either if the NxNxnt tensor with the evolution of the links is given as input, or the Nxnt matrix for the nodes, or the array for the network flow. It requires knowlegde of the temporal span of the simmulation (t0, ffinal) and the time-step dt. That would not be the case if we end transforming everything into objects.
	- Dynamic distance (time-to-relaxation). This one will be the time it takes to reach 95% or 99% of the area under the curve. Try to write a single function that can handle the array for the link, node and network evolutions at once. It requires knowlegde of the temporal span of the simmulation (t0, ffinal) and the time-step dt.
	- Total flow over time. That is, the integral of the area under the curve. ACHTUNG! we need to take the simulation time-step into account. It is an integral, not just the sum over all the values in the time-series. Include optional time span, such that, for example, we can calculate the integral only until the peak (rise phase), or from the peak until the end (decay phase).
	- Function `NodeEvolution()`has been renamed as `NodeFlows()`. The optional parameter `directed` has been removed. Now the function always returns both the input and the output flows regardless of whether the underlying connectivity is directed or not. If it is symmetric, then the in- and out-flows will simply be the same. Although redundant, it is less confusing if the function always returns the structure of the function's output is always the same.
	- A new parameter `selfloops` has been added to function `NodeFlows()`. If `selfloops = True` the output will include the consequence of the initial perturbation applied to a node on itself. If `selfloops = False` (default) then the function only returns the in-flows into a node due to the perturbations on other nodes, and account only for the flow that a nodel provokes on other nodes, not itself.
	- Revisit the function to get the network and node flows over timein *metrics.py*: `TotalEvolution()` and  `NodeEvolution()`. We should have an optional parameter named `selfloops=False/True` to exclude or include the self-interactions from the cross-nodal interactions. At this moment, the calculations include the self-interactions but I am not sure we should do that. The evolution and strength of the self-interactions (response of a node to a perturbation on itself at t = 0) carry their own meaning and should be characterised separately. Re-think how to do this in the algorithm.


- Include a *netmodels.py* module for generating networks and surrogates. Include the followong functions:
	- I have imported *pyGAlib* in module *netmodels.py* such that we can use all the (di) graph generation and randomization functions. In the future we could think whether we want GAlib to be a dependence of ReNetA, or we prefere to duplicate those functions here.
	- Random weighted surrogate networks from an input connectivity matrix: `RandomiseWeightedNetwork()`.
	- A function to shuffle only the weights of the links in a network, conserving the location of the links (same binary net, randomised weights): `ShuffleLinkWeights()`.


- Add functions `NNt2tNN()` and `tNN2NNt()` for transposing the flow tensors in the *tools.py* module.

- (DONE) **IMPORTANT. Graph theory vs dynamical systems convention.** Decide whether *ReNetA* should follow the graph convention such that $A_{ij} = 1$ means a link $i \to j$, or the indexing of dynamical systems instead, meaning $j \to i$.
	- (DONE) Include functions in module *tools.py* to transform the tensors and the matrices between the two conventions.
	- THE DECISION: SiReNetA uses the dynamical systems convention. Therefore, users need to be aware their input adjacency matrices are transposed, if brought from graph generation tools.

- (DONE) Think very carefully the **naming of the canonical models**. There are historical implications here but … One should be pragmatical and besides, those names should really be informative for the user. I would prefer that than using names only because in one field or in another, the models are called in some way. See the *NamingConventions.md* file for proposals.

- (DONE) Think if we want another name for *core.py*. The module has been renamed to *responses.py*.

- (DONE) Think very carefully the **naming of all the metrics**. Both for the existing metrics and the new ones. Stablish a coherent naming system that is general enough, precise and will survive over time to avoid renaming things in the future again. See the *NamingConventions.md* file for proposals.

- (DONE, because we decided for the dynamical systems convention j -> i). Switch the `dot()` operations in *simulate.py* to avoid calculating the transpose matrix. As of now, this module uses internally the dynamical systems convention while *core.py* uses graph convention. We decided to stick to the graph convention.

 
 
 
 
 

####