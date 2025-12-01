[![pypi version](https://img.shields.io/pypi/v/sireneta?logo=pypi)](https://pypi.org/project/sireneta/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/sireneta.svg?label=PyPI%20downloads)](
https://pypi.org/project/sireneta/)
[![Apache-2.0 License](https://img.shields.io/badge/license-Apache-blue.svg?style=flat)](http://choosealicense.com/licenses/Apache-2.0/)

# SiReNetA

> NOTE! Current version is an "*alpha - development*" for testing and validation. Heavy changes expected until release of version 1.0. Stay tuned or follow ‪@gzamora-lopez.bsky.social‬ in Bluesky for updates.

> If only interested in the *leaky-cascade* canonical model (multivariate Ornstein-Uhlenbeck), for now, we recommend to continue using the *[NetDynFlow](https://github.com/mb-BCA/NetDynFlow)* package until *SiReNetA* supersedes *NetDynFlow*.



_**Stimulus-Response Network Analysis (SiReNetA)** : A library for the study of complex networks in the light of canonical propagation models._

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

Installation of SiReNetA is simple, only the [pip](https://github.com/pypa/pip) package manager is needed. To check whether `pip` is installed, open a terminal and type:

	pip --help

> **NOTE**: If you use Anaconda (or any other third-party package manager), we recommend to install the dependencies (python>=3.6, numpy>=1.6, scipy and numba) into the target environment using Anaconda before installing SiReNetA. Otherwise, `pip` will download and install those packages directly from PyPI as well, and you won't be able to manage them through Acanconda.

#### Installing from PyPI 

SiReNetA is registered in the official *Python Package Index*, [PyPI](https://pypi.org/project/sireneta/) . To install, open a terminal window and type:

	python3 -m pip install sireneta

To confirm the installation, open an interactive session (e.g., IPython or a Notebook) and try to import the library by typing `import sireneta`.

#### Direct installation from GitHub 

If you have [git](https://git-scm.com) installed, you may like to install SiReNetA directly from its GitHub repository. Open a terminal and type:

	python3 -m pip install git+https://github.com/gorkazl/SiReNetA.git@master

This will only download and install the package (files in "*src/sireneta/*") into your current environment. Useful for development and testing purposes, you can choose to install the version in another branch by replacing the '*@master*' at the end of the command by '*@branchname*' of the desired branch.

#### Installing SiReNetA in editable mode

If you want to install SiReNetA such that you can make changes to it "*on the fly*" then, visit its GitHub repository [https://github.com/gorkazl/SiReNetA/](https://github.com/gorkazl/SiReNetA/), select a branch and then click on the green "*<> Code*" button on the top right and select "Download ZIP" from the pop-up menu. Once downloaded, move the *zip* file to a target folder (e.g., "*~/Documents/myLibraries/*") and unzip the file. Open a terminal and `cd` to the resulting folder, e.g.,

	cd ~/Documents/myLibraries/SiReNetA-master/

Once on the path (make sure it contains the *pyproject.toml* file), type:

	python3 -m pip install -e .

Do not forget the "." at the end which means "*look for the pyproject.toml file in the current directory*." This will install SiReNetA such that every time changes are made to the package (located in the path chosen), these will be inmediately available. You may need to restart the IPython or Jupyter notebook session, though.




&nbsp;
### HOW TO USE SiReNetA

> Please visit the SiReNetA repository of tutorials [https://github.com/mb-BCA/SiReNetA_Tutorials](https://github.com/mb-BCA/SiReNetA_Tutorials) for documentation and examples to get started with the use of *Response Network Analysis* and the *SiReNetA* library.

The package is organised into the following user modules:

- *__responses.py__* : Functions to calculate the spatio-temporal evolution of pair-wise node responses $R_{ij}(t)$ to initial unit stimuli, under different canonical models.
- *__metrics.py__* : Descriptors to characterise the networks out of the $R_{ij}(t)$ spatio-temporal responses.
- ~~*__simulate.py__* : Functions to run simulations of the different canonical models on networks.~~
- *__tools.py__* : Miscellaneous functionalities.


#### FINDING FURTHER DOCUMENTATION

While working in an interactive session, after importing a module, the built-in `help()` function will show further details. Import sireneta

	>>> import sireneta as sna
	>>> help(sna)

The command `help(sna)` will show the general summary of the package and a list of all the modules in the library. To display information of the individual modules, say, their individual description and the list of functions available, call their documentation as:

	>>> help(sna.responses)
	>>> help(sna.metrics)

For further details regarding each function, access their description and the list of parameters as :

	>>> help(sna.modulename.functionname)

For IPython and Jupyter notebook users, the `help` command is replaced by a question mark after the module's or function's name. For example:

	>>> sna?
	>>> sna.responses?
	>>> sna.metrics.functionname?

>**NOTE:** Importing *SiReNetA* brings all functions in the modules *responses.py* and *metrics.py* into the local namespace. Therefore, these functions can be called as `sna.func()` instead of `sna.responses.func()` or `sna.metrics.func()`.

For questions, bug reports, etc, please write to <gorka@Zamora-Lopez.xyz>, or open an issue in GitHub.


&nbsp;
### LICENSE

Copyright (c) 2024, Gorka Zamora-López and Matthieu Gilson.

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