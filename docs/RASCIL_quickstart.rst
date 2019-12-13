.. Quick start

Quick start
===========


Installation
++++++++++++

Installation should be straightforward. We recommend the use of virtual environment. A prepackaged python
system such as Anaconda https://www.anaconda.com is usually best as a base.

RASCIL requires python 3.6 or higher.

# Use git to make a local clone of the Github respository::

   git clone https://github.com/SKA-ScienceDataProcessor/algorithm-reference-library

# Change into that directory::

   cd algorithm-reference-library

# Install required python packages::

   pip install -r requirements.txt

There may be some dependencies that require either conda (or brew install on a mac).

# Setup RASCIL::

   python setup.py install

# Get the data files form Git LFS::

   git-lfs pull

The README.md file contains much more information about installation.

Running notebooks
+++++++++++++++++

The best way to get familiar with RASCIL is via jupyter notebooks. For example::

   jupyter-notebook processing_components/notebooks/imaging.ipynb

See the jupyter note books below:

.. toctree::
   :maxdepth: 3

   processing_components/imaging.rst
   workflows/imaging-fits_rsexecute.rst
   workflows/imaging-wterm_rsexecute.rst
   workflows/simple-dask_rsexecute.rst
   workflows/imaging-pipelines_rsexecute.rst
   workflows/bandpass-calibration_rsexecute.rst

In addition, there are other notebooks that are not built as part of this documentation.

* :ref:`genindex`
* :ref:`modindex`

.. _feedback: mailto:realtimcornwell@gmail.com
