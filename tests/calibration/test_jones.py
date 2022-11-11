""" Unit tests for Jones matrix application

"""

import logging

import numpy
from numpy.testing import assert_array_almost_equal
from ska_sdp_datamodels.science_data_model.polarisation_functions import (
    convert_pol_frame,
)
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)

from ska_sdp_func_python.calibration.jones import apply_jones

log = logging.getLogger("func-python-logger")

log.setLevel(logging.WARNING)


def test_apply_jones():
    nsucceeded = 0
    nfailures = 0
    for flux in (
        numpy.array([100.0, 0.0, 0.0, 0.0]),
        numpy.array([100.0, 100.0, 0.0, 0.0]),
        numpy.array([100.0, 0.0, 100.0, 0.0]),
        numpy.array([100.0, 0.0, 0.0, 100.0]),
        numpy.array([100.0, 1.0, -10.0, +60.0]),
    ):
        vpol = PolarisationFrame("linear")
        cpol = PolarisationFrame("stokesIQUV")
        cflux = convert_pol_frame(flux, cpol, vpol, 0).reshape([2, 2])

        diagonal = numpy.array(
            [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0]]
        )
        skew = numpy.array([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0]])
        leakage = numpy.array(
            [[1.0 + 0.0j, 0.0 + 0.1j], [0.0 - 0.1j, 1.0 + 0.0]]
        )
        unbalanced = numpy.array(
            [[100.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.03 + 0.0]]
        )

        for ej in (diagonal, skew, leakage, unbalanced):
            try:
                jflux = apply_jones(ej, cflux, inverse=False)
                rflux = apply_jones(ej, jflux, inverse=True).reshape([4])
                rflux = convert_pol_frame(rflux, vpol, cpol, 0)
                assert_array_almost_equal(flux, numpy.real(rflux), 12)
                nsucceeded += 1
            except AssertionError as e:
                print(e)
                print("{0} {1} {2} failed".format(vpol, str(ej), str(flux)))
                nfailures += 1
    assert nfailures == 0, "{0} tests succeeded, {1} failed".format(
        nsucceeded, nfailures
    )
