# TO-DO list before public release


In order to properly finalise v1, ask Nacho:
- How to deal with inputs to functions. Check every input parameter?
- How to deal with the updates of the Copyright statement !?
- What to do with the *tools.py* module ?
- ~~Reminder of how to build Python packages and upload them to PyPI, following the "new" rules.~~


#### Before merging new version into master:

- ~~Reshape repository and prepare everything for SiReNetA to be compatible with the new Python Packaging rules~~. **Reshaped and prepared to use "Hatch"**.
- Move all functions from **netmodels.py** to pyGAlib and from **simulate.py** to a new *light* simulation package.
- Clean *\_\_init\_\_.py*: 
	- Remove imports to test modules, and to irrelevant modules. 
	- Double check list of absolute imports.
	- Update general description. Explanation for the canonical models.
	- Double check the instructions in the initial docstring. New names of functions. Make sure the examples work if copy/pasted.
	- Update the publication list, include the latest dynflow paper.
- ~~Update the Copyright dates in all files.~~
- Update the version number to 1.0 in *\_\_init\_\_.py*.
- Remove unnecessary files (to-do lists, NamingConventions, etc.)
- Update the **README.md** file.
	- Remove initial unnecessary text. 
	- ~~Update general description(s), include explanation for the different canonical models.~~ **No, that is why we created the tutorial repository.**
	- ~~Update the Copyright / license infos.~~
	- Double check the instructions in the initial docstring. New names of functions. Make sure the examples work if copy / pasted.
	- Update list of changes.

- Update the references to the papers / pre-prints.


#### Installation for testing as well

- Give all the steps to make the repository / package installable.
- Verify installation works.
- (If needed) Update the installation instructions in README.md + *\_\_init\_\_.py*.


#### Make the repository public

- Create a GitHub release for v1.
- Upload SiReNetA to PyPI.


