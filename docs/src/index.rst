.. _documentation_master:

.. toctree::

SKA SDP Python Processing Functions
###########################################################
This is a `repository <https://gitlab.com/ska-telescope/sdp/ska-sdp-func-python.git>`_
for the Python Processing Functions used in the SKA SDP. The aim of this repository is to
provide the Processing Functions involved in radio astronomy visibility processing.
The processing functions are specifically meant to facilitate passing data between services
and data models within the SDP.

Eventually this should cover:

- In-memory communication within the same process, both between Python software as
  well as Python and C++ software (such as `ska-sdp-func <https://gitlab.com/ska-telescope/sdp/ska-sdp-func>`_)

- In-memory communication between different processes, such as via shared memory
  (e.g. as done using Apache Plasma in real-time processing)

- Network communication between different processes for the purpose of distributed computing
  (e.g. via Dask or Kafka)

The code is written in Python. The structure is modeled after the
standard processing functions used in `RASCIL <https://gitlab.com/ska-telescope/external/rascil-main.git>`_.
The interfaces operate with familiar data structures such as image,
visibility table, gain table, etc. The python source code is directly accessible from these documentation pages:
see the source link in the top right corner.


Installation Instructions
=========================

The package is installable via pip.

If you would like to view the source code or install from git, use::

    git clone https://gitlab.com/ska-telescope/sdp/ska-sdp-func-python.git

Please ensure you have all the dependency packages installed. The installation is managed
through `poetry <https://python-poetry.org/docs/>`_.
Refer to their page for instructions.


.. toctree::
  :maxdepth: 2



