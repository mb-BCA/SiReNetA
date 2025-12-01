[![pypi version](https://img.shields.io/pypi/v/sireneta?logo=pypi)](https://pypi.org/project/sireneta/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/sireneta.svg?label=PyPI%20downloads)](
https://pypi.org/project/sireneta/)
[![Apache-2.0 License](https://img.shields.io/badge/license-Apache-blue.svg?style=flat)](http://choosealicense.com/licenses/Apache-2.0/)

# SiReNetA

> NOTE! Current version is an "*alpha - development*" for testing and validation. Heavy changes expected until release of version 1.0. Stay tuned or follow ‪@gzamora-lopez.bsky.social‬ in Bluesky for updates.

> If only interested in the *leaky-cascade* canonical model (multivariate Ornstein-Uhlenbeck), for now, we recommend to continue using the *[NetDynFlow](https://github.com/mb-BCA/NetDynFlow)* package until *SiReNetA* supersedes *NetDynFlow*.



**Sitmulus-Response Network Analysis (SiReNetA)** : A library for the study of complex networks in the light of canonical propagation models.

Graph theory constitutes a widely used and established field providing powerful tools for the characterisation of complex networks. However, the diversity of complex networks studied nowadays overcomes the capabilities of graph theory (originally developed for binary adjacency matrices) to understand networks and their function. In the recent years plenty of alternative metrics have been proposed which are–one way or another–based on dynamical phenomena happening on networks.

*Stimulus-Response Network Analysis (SRNA)* proposes a generalised course of action to derive network metrics and characterise networks from the viewpoint of dynamical systems, valid for different canonical propagation models. The first step of the analysis consists of selecting an adequate propagation model that respects minimal constraints and assumptions of the real network under investigation. Once the model is chosen, the temporal pair-wise (conditional) responses $R_{ij}(t)$ that nodes exert on each other are estimated. Finally information about the network is derived out of the observed responses.


>Visit [https://github.com/mb-BCA/SiReNetA_Tutorials](https://github.com/mb-BCA/SiReNetA_Tutorials) for tutorials and practical examples of how to use '*Response Network Analysis*' and the *SiReNetA* library.


##### References and Citation

- G. Zamora-López and M. Gilson "*[An integrative dynamical perspective for graph theory and the analysis of complex networks](https://doi.org/10.1063/5.0202241)*" Chaos **34**, 041501 (2024).
- M. Gilson, N. E. Kouvaris, et al. "*[Network analysis of whole-brain fMRI
dynamics: A new framework based on dynamic communicability](https://doi.org/10.1016/j.neuroimage.2019.116007)*" NeuroImage **201**, 116007 (2019).
- M. Gilson, N. E. Kouvaris, G. Deco and G. Zamora-López "*[Framework based on communicability and flow to analyze complex networks](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.97.052301)*" Phys. Rev. E **97**, 052301 (2018).



&nbsp;
### INSTALLATION

Installation of *SiReNetA* is simple. The only requirements are an existing python distribution and the [pip](https://github.com/pypa/pip) package manager. If Python was installed via the [Anaconda](https://www.anaconda.com) or another Python distribution, then 'pip' is surely installed. To check, open a terminal and type:

	$ pip --help

*SiReNetA* is still not registered in PyPI (the Python Package Index) and installation follows directly from GitHub. However, pip will automatically take care of the  dependencies (see the *requirements.txt* file). There are two alternative manners to install.

**A) Direct installation from GitHub**: Open a terminal and enter:

	$ pip install git+https://github.com/mb-BCA/SiReNetA.git@master

This will install the folder "*sireneta/*" of this repository into the "*…/site-packages/*" folder of your current python environment. To confirm the installation open an interactive session and try to import the library by typing `import sireneta`.

The installation command can also be run from a cell in a **Jupyter notebook**. In that case, begin the cell with "%", what allows to run terminal commands. Type the following in a cell of the notebook :

	%pip install git+https://github.com/mb-BCA/SiReNetA.git@master

**B) Download and install**: Visit the GitHub repository [https://github.com/mb-BCA/SiReNetA/](https://github.com/mb-BCA/SiReNetA/) and click on the "<> Code" button at the right hand side (the green button). Select "Download ZIP". Download to a preferred path, e.g.' "~/Downloads/" and unzip the file. Open a terminal and move to that folder, e.g.,

	$ cd ~/Downloads/sireneta-master/

Make sure this folder is the one containing the *setup.py* file. Then, type:

	$ pip install .

Do not forget the "." at the end which means "*install from this directory using the code in setup.py*." This will check for the dependencies and install *SiReNetA*. To confirm the installation open an interactive session and try to import the library by typing `import sireneta`. After installation the folder "*~/Downloads/sireneta-master/*" can be safely deleted.



&nbsp;
### ORGANISATION AND DOCUMENTATION

The library is organised into the following user modules:

- *__responses.py__* : Functions to calculate the spatio-temporal evolution of pair-wise node responses $R_{ij}(t)$ to initial unit stimuli, under different canonical models.
- *__metrics.py__* : Descriptors to characterise the networks out of the $R_{ij}(t)$ spatio-temporal responses.
- *__simulate.py__* : Functions to run simulations of the different canonical models on networks.
- *__tools.py__* : Miscellaneous functionalities.

The documentation of the library can be accessed 'online' typing  `help(module_name)` in a Python interactive session, or typing `module_name?` in IPython or a Jupyter notebook. For example, to see the principal description of *SiReNetA*, type :

	>>> import sireneta as sna
	>>> sna?

To see the list of functions available in each of the modules, call their documentation as :

	>>> sna.responses?
	>>> sna.metrics?
	>>> sna.simulate?
	>>> sna.tools?

Details of each function can also be seen using the usual help,

	>>> sna.module_name.func_name?

>**NOTE:** Importing *SiReNetA* brings all functions in the modules *responses.py* and *metrics.py* into the local namespace. Therefore, these functions can be called as `sna.func()` instead of `sna.responses.func()` or `sna.metrics.func()`.


&nbsp;
### LICENSE

Copyright (c) 2024 – 2025, Gorka Zamora-López and Matthieu Gilson. Contact: <gorka@Zamora-Lopez.xyz>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this software except in compliance with the License.
You may obtain a copy of the License at

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


<br>

-------------------------------------------------------------------------------
### VERSION HISTORY

##### December XX, 2025 (Release of Version 1.0)

TO BE REVISED BEFORE v1 RELEASE.
Stable version 1.0 checked, validated and released.

* The library has been reshaped to be compliant with the modern [PyPA specifications](https://packaging.python.org/en/latest/specifications/).
* [Hatch](https://hatch.pypa.io/latest/) was chosen as the tool to build and publish the package. See the *pyproject.toml* file. 
* Bug fixes to adapt to the various changes in Python and NumPy since last release.
* _Sample and validation scripts in the "*Examples/*" folder revised and adapted to recent changes in Python and NumPy._ 

##### March 14, 2024

Fixed the new  aliases for `int` and `float` in *Numpy*. All arrays are now declared as `np.int64` or `np.float64`, and individual numbers as standard Python `int` or `float`. 

##### March 7, 2024

*SiReNetA* is made publicly available in *alpha - development* version for testing and referencing. Both internal (e.g., specifics of algorithms) and external (e.g., names of functions) changes may happen before final release of version 1.0. Comments, bug reports and recommendations are welcome.