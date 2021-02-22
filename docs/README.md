# `indl`

`indl` (Intracranial Neurophys and Deep Learning) is a Python package providing some tools to assist with deep-learning analysis of neurophysiology data, with an emphasis on intracranial neurophysiology. It is intended to be a companion to the SachsLab tutorial found at https://github.com/SachsLab/IntracranialNeurophysDL . The SachsLab also uses this library in some of their research.

## Install

`pip install git+https://github.com/SachsLab/indl.git`

### Dependencies

I haven't added all the requirements to settings.ini yet so they won't be installed automatically. Chances are that if you are using this Python package then you already have the required dependencies installed in your environment. Some of the dependencies are:

* numpy
* scipy
* tensorflow
* tensorflow-probability
* h5py (only required for importing our custom monkey neurophys files)
* pandas (only required for importing our custom monkey neurophys files)

## Documentation

The documentation is under construction but can be found at https://SachsLab.github.io/indl/

## Maintenance Notes

Some notes for ongoing maintaining of this repository.

### Repository Organization

* Library code in the /indl folder
* Unit tests in the /tests folder. Note this uses the `pytest` framework and conventions.
  * I'm still building out the unit tests.
* Documentation in the /docs folder
  * If you build the docs locally then you'll also get the /site directory, but this should be git ignored.

### Maintaining the Documentation

You will need to install several Python packages to maintain the documentation.

* `pip install mkdocs mkdocstrings mknotebooks mkdocs-material Pygments`

The /docs/API folder has stubs to tell the [`mkdocstrings`](https://github.com/mkdocstrings/mkdocstrings) plugin to build the API documentation from the docstrings in the library code itself.

The /docs/{top-level-section} folders contain a mix of .md and .ipynb documentation. The latter are converted to .md by the [`mknotebooks`](https://github.com/greenape/mknotebooks/projects) plugin during building.

Run `mkdocs gh-deploy` to build the documentation, commit to the `gh-deploy` branch, and push to GitHub. This will make the documentation available at https://SachsLab.github.io/indl/

#### Testing the documentation locally

* `mkdocs build`
* `mkdocs serve`

### Running the unit tests

There aren't that many tests yet, I'm still building them out.
