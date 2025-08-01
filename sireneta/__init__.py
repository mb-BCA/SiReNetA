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
Response Network Analysis (SiReNetA)
==================================

A library to study complex networks in the light of canonical propagation models.

*SiReNetA* proposes a generalised course of action to derive network metrics and
characterise networks from the viewpoint of dynamical systems, valid for
different canonical propagation models. Since *SiReNetA* introduces a model-based
approach to network analysis, it consists on four fundamental steps:

1. Identify the constraints and basic assumptions about the real system studied.
2. Select accordingly an adequate propagation model.
3. Compute the response tensor *R(t)* evaluating the interactions between nodes.
4. Extract information of the network applying a variety of metrics on *R(t)*.

**Reference and Citation**

G. Zamora-Lopez and M. Gilson "An integrative dynamical perspective for graph
theory and the analysis of complex networks" Chaos 34, 041501 (2024).
DOI: `https://doi.org/10.1063/5.0202241
<https://doi.org/10.1063/5.0202241>`_

M. Gilson, N. Kouvaris, et al. "Network analysis of whole-brain fMRI
dynamics: A new framework based on dynamic communicability" NeuroImage 201,
116007 (2019).

M. Gilson, N. Kouvaris, G. Deco & G.Zamora-Lopez "Framework based on communi-
cability and flow to analyze complex networks" Phys. Rev. E 97, 052301 (2018).


Getting started and finding documentation
-----------------------------------------

..  Note:: Visit the repository https://github.com/mb-BCA/SiReNetA_Tutorials for
    tutorials and practical examples of how to use *Response Network Analysis*
    and the *SiReNetA* library.

The library is organised into the following user modules:

responses.py
    Functions to calculate the spatio-temporal evolution of pair-wise node
    responses for different canonical models.
metrics.py
    Descriptors to characterise the spatio-temporal evolution of perturbation-induced
    responses in a network.
simulate.py
    Functions to run simulations of the different canonical models on networks.
tools.py
    Miscellaneous functionalities.

The documentation of the library can be accessed 'online' typing
`help(module.py)` in a Python interactive session, or typing `modulename?`
in IPython or a Jupyter notebook. For example, to see the principal description
of *SiReNetA*, type: ::

  >>> import sireneta as sna
  >>> help(sna)
  >>> sna?

To see the list of functions available in each of the modules, call their
documentation as ::

  >>> sna.responses?
  >>> sna.metrics?
  >>> sna.simulate?
  >>> sna.tools?

Details of each function can also be seen using the usual help, ::

  >>> sna.modulename.funcname?

..  NOTE:: Importing *SiReNetA* brings all functions in the modules *responses.py*
    and *metrics.py* into the local namespace. Therefore, these functions can be
    called as `sna.func()` instead of `sna.responses.func()` or `sna.metrics.func()`.


License
-------
Copyright 2024, Gorka Zamora-López and Matthieu Gilson.
Contact: gorka@zamora-lopez.xyz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from . import responses
from .responses import *
from . import metrics
from .metrics import *
from . import simulate
from .simulate import *
from . import tools
# from . import netmodels
# from . import metrics_test


__author__ = "Gorka Zamora-Lopez and Matthieu Gilson"
__email__ = "gorka@Zamora-Lopez.xyz"
__copyright__ = "Copyright 2024"
__license__ = "Apache License version 2.0"
__version__ = "1.0.0.dev1"
__update__ = "17/04/2024"






#
