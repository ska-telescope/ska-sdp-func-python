""" Unit tests for image Taylor terms

"""
import logging

import numpy
import pytest
from astropy import units
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.image.image_create import create_image

from ska_sdp_func_python.image.gather_scatter import image_scatter_channels
from ska_sdp_func_python.image.taylor_terms import (
    calculate_frequency_taylor_terms_from_image_list,
    calculate_image_frequency_moments,
    calculate_image_from_frequency_taylor_terms,
)

log = logging.getLogger("func-python-logger")

log.setLevel(logging.WARNING)


@pytest.fixture(scope="module", name="result_taylor_terms")
def taylor_terms_fixture():

    npixel = 512
    cellsize = 0.0001
    phase_centre = SkyCoord(
        ra=+180.0 * units.deg,
        dec=-60.0 * units.deg,
        frame="icrs",
        equinox="J2000",
    )
    image = create_image(npixel, cellsize, phase_centre)
    params = {"image": image, "phasecentre": phase_centre}
    return params


@pytest.mark.skip(
    reason="Issues remaining with create_image_frequency_moments : "
    "bad operand type for unary -: 'WCS' in create_image()"
)
def test_calculate_image_frequency_moments(result_taylor_terms):
    cube = create_image(512, 0.0001, result_taylor_terms["phasecentre"])
    moment_cube = calculate_image_frequency_moments(cube, nmoment=3)
    reconstructed_cube = calculate_image_from_frequency_taylor_terms(
        cube, moment_cube
    )
    error = numpy.std(
        reconstructed_cube["pixels"].data
        - result_taylor_terms["image"].data_vars["pixels"].data
    )
    assert error < 0.2, error


@pytest.mark.skip(
    reason="Issues remaining with create_image_frequency_moments as above"
)
def test_calculate_image_frequency_moments_1(result_taylor_terms):
    original_cube = result_taylor_terms["image"]
    cube = create_image(512, 0.0001, result_taylor_terms["phasecentre"])
    moment_cube = calculate_image_frequency_moments(cube, nmoment=1)
    reconstructed_cube = calculate_image_from_frequency_taylor_terms(
        cube, moment_cube
    )

    error = numpy.std(
        reconstructed_cube["pixels"].data - original_cube["pixels"].data
    )
    assert error < 0.2


def test_calculate_taylor_terms(result_taylor_terms):
    original_cube = result_taylor_terms["image"]
    original_list = image_scatter_channels(original_cube)
    taylor_term_list = calculate_frequency_taylor_terms_from_image_list(
        original_list, nmoment=3
    )
    assert len(taylor_term_list) == 3
