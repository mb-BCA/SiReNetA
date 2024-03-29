# TO-DO list before public release

#### Before merging new version into master:


- Open a new branch out of master --> '**v1-legacy**' where I will keep the test files and modules that won't make to the v1 release.
- Clean ____init____.py: 
	- Remove imports to test modules, and to irrelevant modules. 
	- Double check list of absolute imports.
	- Update general description. Explanation for the canonical models.
	- Double check the instructions in the initial docstring. New names of functions. Make sure the examples work if copy/pasted.
	- Update the publication list, include the latest dynflow paper.
- Revise the `__all__` list of functions in *core.py*. Do we need such a list of absolute imports in the other modules?
- Update the Copyright dates in all files.
- Update the version number to 1.0 in *____init___.py* and *setup.py*.
- Remove unnecessary files (to-do lists, NamingConventions, etc.)
- Update the **README.md** file.
	- Remove initial unnecessary text. 
	- Update general description(s), include explanation for the different canonical models.
	- Update the Copyright/license infos.
	- Double check the instructions in the initial docstring. New names of functions. Make sure the examples work if copy/pasted.
	- Update list of changes.

- Update the references to the papers / pre-prints.


#### Installation for testing as well

- Give all the steps to make the repository / package installable.
- Verify installation works.
- (If needed) Update the installation instructions in README.md + ____init____.py.


#### Make the repository public

- Create a GitHub release.
- Add the library to PYPI (?)


