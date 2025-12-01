## Naming conventions for the metrics and their corresponding functions

#### communicability, response, flows, … ??

A possible source of confusion is that some of the metrics are characterised at three different levels: network, node and link. So, this needs to be explicit in the naming. For example, we could say (network flow, node flow and link flow).


##### List of names to take care of, and decide upon


- model-based network analysis --> flow-based, diffusion-based, response-based, impulse-based, perturbation-based, ...
- response --> activity, influence, flow, **response**
- dynamic response --> temporal response.
- total response --> network response, global network response, network flow, … ("total" is not a good term because it could mean the whole-network, or the total response over time.)
- in-/out-response --> node response.
- in-response --> node sensitivity/reactivity? Integration?
- out-response --> node influence? Centrality? Broadcasting?
- If network/node/link response are the temporal evolution of the network/node/link, then how to call the sum over time (integral) of those quantities?


##### Naming for the canonical models, to decide upon

These names are relevant because they are the "short names" we should use to "tag" the functions that are specific for each model. And, for simplicity and coherence, these should be the same names we will use in v3 for the class attribute that specifies the canonical model.

- Discrete cascade
- Random walks
- (Random walks with teleportation)
- Continuous cascade
- Leaky cascade
- Continuous diffusion (simple diffusion)


####  A new name for the library, to replace 'NetDynFlow'

In the new paper, where we generalise the ideas of the "dynamic response" for plugin different canonical models, we converged into calling Rij(t) as the responses from one node to another. Therefore, we are no longer talking of flows or response and we should have a different name for the library. Also, we do not want to use the word "pertubartion" because some people think it refers to a lession of the network nodes or links. So, it seems that "Response Network Analysis" could be a reasonable naming.
 
- Response Network Analysis. 
	- ReNetA `rna.function()`
	- Renata
	- RespNet `rn.function()`
- Network Response Analysis. NetReA, `nra.function()`
- Perturbation-Response Network analysis
- Stimulus-Response Network Analysis
	- SRNetA `sra.function()` 
	- SiReNA `sirena.function()`  `sna.function()`
	- SiReNetA `sra.function()` `srn.function()`

**ACHTUNG !!** We must look in PyPI whether libraries with these names already exist :(


#### Suggested names for functions

Special care for the functions to compute the network (pair-wise) responses.

<br>

#### Internal variable naming conventions


Here the list of proposed variable and function changes. Add alternative names for each case if dissagree, or add other proposals as well.

- con_matrix --> con
- tau_const --> tau
- sigma_mat --> sigma
- n_nodes --> N
- GenerateTensors() --> CalcTensor()



<br/>

List here the agreed name changes:

- jac 	<-- J (in pyMOU.mou_model), jacobian (in NetDynFlow.core)
- jacdiag <-- jacobian_diag (in NetDynFlow.core)
- tau 	<-- tau_x (in pyMOU.mou_model), tau_const (in NetDynFlow.core)
- con 	<- C (in pyMOU.mou_model), con_matrix (in NetDynFlow.core)
- tensor	<-- dyn_tensor (in NetDynFlow.metrics)
- sigma or incov <-- Sigma (in pyMOU.mou_model)
- cov0, covlag <-- Q0, Qtau (in pyMOU.mou_model)
- lag 	<- tau (in pyMOU.mou_model.Mou.fit_LO)


