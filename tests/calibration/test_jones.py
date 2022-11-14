""" Unit tests for Jones matrix application

"""

import logging

import numpy
import pytest
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

DIAGONAL = numpy.array(
    [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0]]
)
SKEW = numpy.array([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0]])
LEAKAGE = numpy.array(
    [[1.0 + 0.0j, 0.0 + 0.1j], [0.0 - 0.1j, 1.0 + 0.0]]
)
UNBALANCED = numpy.array(
    [[100.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.03 + 0.0]]
)


@pytest.mark.parametrize("flux, jones_matrix", [
    (numpy.array([100.0, 0.0, 0.0, 0.0]), DIAGONAL),
    (numpy.array([100.0, 100.0, 0.0, 0.0]), DIAGONAL),
    (numpy.array([100.0, 0.0, 100.0, 0.0]), DIAGONAL),
    (numpy.array([100.0, 0.0, 0.0, 100.0]), DIAGONAL),
    (numpy.array([100.0, 1.0, -10.0, +60.0]), DIAGONAL),

    (numpy.array([100.0, 0.0, 0.0, 0.0]), SKEW),
    (numpy.array([100.0, 100.0, 0.0, 0.0]), SKEW),
    (numpy.array([100.0, 0.0, 100.0, 0.0]), SKEW),
    (numpy.array([100.0, 0.0, 0.0, 100.0]), SKEW),
    (numpy.array([100.0, 1.0, -10.0, +60.0]), SKEW),

    (numpy.array([100.0, 0.0, 0.0, 0.0]), LEAKAGE),
    (numpy.array([100.0, 100.0, 0.0, 0.0]), LEAKAGE),
    (numpy.array([100.0, 0.0, 100.0, 0.0]), LEAKAGE),
    (numpy.array([100.0, 0.0, 0.0, 100.0]), LEAKAGE),
    (numpy.array([100.0, 1.0, -10.0, +60.0]), LEAKAGE),

    (numpy.array([100.0, 0.0, 0.0, 0.0]), UNBALANCED),
    (numpy.array([100.0, 100.0, 0.0, 0.0]), UNBALANCED),
    (numpy.array([100.0, 0.0, 100.0, 0.0]), UNBALANCED),
    (numpy.array([100.0, 0.0, 0.0, 100.0]), UNBALANCED),
    (numpy.array([100.0, 1.0, -10.0, +60.0]), UNBALANCED),
])
def test_apply_jones(flux, jones_matrix):
    """
    Unit tests for the apply_jones function

    :param jones_matrix: what the code calls "ej",
            see: ska_sdp_func_python.calibration.jones.apply_jones
    """
    vpol = PolarisationFrame("linear")
    cpol = PolarisationFrame("stokesIQUV")
    cflux = convert_pol_frame(flux, cpol, vpol, 0).reshape([2, 2])

    jflux = apply_jones(jones_matrix, cflux, inverse=False)
    rflux = apply_jones(jones_matrix, jflux, inverse=True).reshape([4])
    rflux = convert_pol_frame(rflux, vpol, cpol, 0)

    assert_array_almost_equal(flux, numpy.real(rflux), 12)
