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
    calculate_image_list_frequency_moments,
    calculate_image_list_from_frequency_taylor_terms,
)

log = logging.getLogger("func-python-logger")

log.setLevel(logging.WARNING)


@pytest.fixture(scope="module", name="input_params")
def taylor_terms_fixture():
    """Fixture for the taylor_terms.py unit tests"""
    npixel = 512
    cellsize = 0.00015
    phase_centre = SkyCoord(
        ra=+180.0 * units.deg,
        dec=-60.0 * units.deg,
        frame="icrs",
        equinox="J2000",
    )
    image = create_image(npixel, cellsize, phase_centre)
    params = {"image": image, "phasecentre": phase_centre}
    return params


def test_calculate_image_frequency_moments_3(input_params):
    """Unit test for the calculate_image_frequency_moments function
    with 3 moments
    """
    original_cube = input_params["image"]
    cube = create_image(512, 0.0001, input_params["phasecentre"])
    moment_cube = calculate_image_frequency_moments(cube, nmoment=3)
    reconstructed_cube = calculate_image_from_frequency_taylor_terms(
        cube, moment_cube
    )
    error = numpy.std(
        reconstructed_cube["pixels"].data - original_cube["pixels"].data
    )
    assert error < 0.2, error


def test_calculate_image_frequency_moments_1(input_params):
    """Unit test for the calculate_image_frequency_moments function
    with 1 moment
    """
    original_cube = input_params["image"]
    cube = create_image(512, 0.0001, input_params["phasecentre"])
    moment_cube = calculate_image_frequency_moments(cube, nmoment=1)
    reconstructed_cube = calculate_image_from_frequency_taylor_terms(
        cube, moment_cube
    )

    error = numpy.std(
        reconstructed_cube["pixels"].data - original_cube["pixels"].data
    )
    assert error < 0.2


def test_calculate_image_list_frequency_moments(input_params):
    """Unit test for the calculate_image_list_frequency_moments function
    with 3 moments
    """
    original_cube = input_params["image"]
    image_list = [original_cube, original_cube, original_cube]
    reconstructed_cube = calculate_image_list_frequency_moments(
        image_list, nmoment=3
    )

    error = numpy.std(
        reconstructed_cube["pixels"].data - original_cube["pixels"].data
    )
    assert error < 0.2


def test_calculate_image_list_from_frequency_taylor_terms(input_params):
    """Unit test for the calculate_image_list_frequency_taylor_terms"""
    original_cube = input_params["image"]
    image_list = [original_cube, original_cube, original_cube]
    moment_cube = calculate_image_frequency_moments(original_cube, nmoment=1)
    reconstructed_cube_list = calculate_image_list_from_frequency_taylor_terms(
        image_list, moment_cube
    )
    for image in reconstructed_cube_list:
        error = numpy.std(image["pixels"].data - original_cube["pixels"].data)
        assert error < 0.2


def test_calculate_taylor_terms(input_params):
    """Unit test for the calculate_taylor_termss function"""
    original_cube = input_params["image"]
    original_list = image_scatter_channels(original_cube)
    taylor_term_list = calculate_frequency_taylor_terms_from_image_list(
        original_list, nmoment=3
    )
    assert len(taylor_term_list) == 3
