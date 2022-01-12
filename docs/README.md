# Introduction

`indl` (Intracranial Neurophys and Deep Learning) is a Python package providing some tools to assist with deep-learning analysis of neurophysiology data, with an emphasis on intracranial neurophysiology.

This library is a dependency in some of the lab's research projects and its [Tutorial on Intracranial Neurophysiology and Deep Learning](https://github.com/SachsLab/IntracranialNeurophysDL).

You may be interested in the notebook on [disentangling sequential autoencoders](https://sachslab.github.io/indl/DSAE/dsae/) that explores many aspects of this library.

## Install

`pip install git+https://github.com/SachsLab/indl.git`

### Dependencies

The Python package dependencies should be handled automatically during pip install.

If you need Tensorflow with GPU support then this requires cuda toolkit. The easiest way to get a compatible set of tensorflow, cuda toolkit, python, kernel, etc. is to use a conda environment. Use `conda install tensorflow-gpu` in a conda environment before pip-installing this package. 

## Documentation

The documentation is under construction but can be found hosted at https://SachsLab.github.io/indl/ . Use the navigation bar to select different elements. The `API` docs are auto-generated from the code. The `DSAE` docs contain some information about how the library can be used for disentangling sequential auto-encoders.

## Maintenance Notes

Some notes for ongoing maintenance of this repository.

### Repository Organization

* `docs` -- Documentation
  * If you build the docs locally then you'll also get the /site directory, but this should be git ignored.
* `indl` -- Library code
* `tests` -- Unit tests. Note this uses the `pytest` framework and conventions.
  * The unit tests are also good examples on how to use specific functions.

### Setting up a developer environment.

* Clone this repo and change into its directory.
* Create a conda env and install the cuda toolkit that is compatible with tensorflow-gpu for python 3.9.
  * `conda create -n indl python=3.9 tensorflow-gpu nodejs`
  * `conda activate indl`
* You will need additional packages for development.
  * `pip install build packaging twine jupyter matplotlib nni pytest jupyterlab`
  * See this list under "Maintaining the Documentation" below.
* `pip install -e .` to install this package in developer mode.

At this point I open the `indl` directory in PyCharm and set its interpreter to be the `indl` conda environment.

### Maintaining the Documentation

You will need to install several Python packages to maintain the documentation.

* `pip install mkdocs mkdocstrings mknotebooks mkdocs-material Pygments`

The `docs/API` folder has stubs to tell the [`mkdocstrings`](https://github.com/mkdocstrings/mkdocstrings) plugin to build the API documentation from the docstrings in the library code itself.

The `docs/{top-level-section}` folders contain a mix of .md and .ipynb documentation. The latter are converted to .md by the [`mknotebooks`](https://github.com/greenape/mknotebooks/projects) plugin during building.

[Here is a guide](https://mkdocstrings.github.io/usage/) for mkdocstrings syntax.

Configure your IDE to use Google-style docstrings.

#### Testing the Documentation Locally

* `mkdocs serve`

#### Deploying the Documentation

* `mkdocs gh-deploy`
  
This builds the documentation, commits to the `gh-deploy` branch, and pushes to GitHub. This will make the documentation available at https://SachsLab.github.io/indl/

### Running the unit tests

I typically run the unit tests within PyCharm as part of my development process, and I'm the only developer on the project, so I haven't paid much attention to testing in CI.

### Publishing the package

* `python -m build`
* `twine upload dist/*`
  * username: `__token__`
  * password: {<}actual token that you saved previously}
