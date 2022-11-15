.. _ska_sdp_func_python_calibration:

.. py:currentmodule:: ska_sdp_func_python.calibration

***********
Calibration
***********

Calibration is performed by fitting observed visibilities to a model visibility.

The scalar equation to be minimised is:

.. math:: S = \sum_{t,f}^{}{\sum_{i,j}^{}{w_{t,f,i,j}\left| V_{t,f,i,j}^{\text{obs}} - J_{i}{J_{j}^{*}V}_{t,f,i,j}^{\text{mod}} \right|}^{2}}

The least squares fit algorithm uses an iterative substitution (or relaxation) algorithm from Larry D'Addario in the
late seventies.

.. toctree::
  :maxdepth: 3

.. automodapi::    ska_sdp_func_python.calibration.chain_calibration
  :no-inheritance-diagram:

.. automodapi::    ska_sdp_func_python.calibration.iterators
  :no-inheritance-diagram:

.. automodapi::    ska_sdp_func_python.calibration.operations
  :no-inheritance-diagram:

.. automodapi::    ska_sdp_func_python.calibration.solvers
  :no-inheritance-diagram:


