.. _functions:

.. toctree::
  :maxdepth: 2

Important Functions
===================


Visibility weighting and tapering
---------------------------------

* Weighting: :py:func:`ska_sdp_func_python.imaging.weighting.weight_visibility`
* Gaussian tapering: :py:func:`ska_sdp_func_python.imaging.weighting.taper_visibility_gaussian`
* Tukey tapering: :py:func:`ska_sdp_func_python.imaging.weighting.taper_visibility_tukey`

Visibility predict and invert
-----------------------------

* Predict by de-gridding visibilities with Nifty Gridder: py:func:`ska_sdp_func_python.imaging.ng.predict_ng`
* Invert by gridding visibilities with Nifty Gridder: py:func:`ska_sdp_func_python.imaging.ng.invert_ng`
* Predict by de-gridding visibilities with GPU-based WAGG: py:func:`ska_sdp_func_python.imaging.wg.predict_wg`
* Invert by gridding visibilities with GPU-based WAGG: py:func:`ska_sdp_func_python.imaging.wg.invert_wg`

Deconvolution
-------------

* Deconvolution: py:func:`ska_sdp_func_python.image.deconvolution.deconvolve_cube` wraps:

 * Hogbom Clean: :py:func:`ska_sdp_func_python.image.cleaners.hogbom`
 * Hogbom Complex Clean: :py:func:`ska_sdp_func_python.image.cleaners.hogbom_complex`
 * Multi-scale Clean: :py:func:`ska_sdp_func_python.image.cleaners.msclean`
 * Multi-scale multi-frequency Clean: :py:func:`ska_sdp_func_python.image.cleaners.msmfsclean`

* Restore: :py:func:`ska_sdp_func_python.image.deconvolution.restore_cube`

Calibration
-----------

* Solve for complex gains: :py:func:`ska_sdp_func_python.calibration.solvers.solve_gaintable`


Coordinate transforms
---------------------

* Station/baseline (XYZ <-> UVW): :py:mod:`ska_sdp_func_python.util.coordinate_support`
* Source (spherical -> tangent plane): :py:mod:`ska_sdp_func_python.util.coordinate_support`



