# pylint: skip-file
""" Unit tests for image Taylor terms

"""
import logging

import astropy.units as u
import numpy
import pytest
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.image.image_create import create_image
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)
from ska_sdp_datamodels.sky_model.sky_model import SkyComponent

from ska_sdp_func_python.skycomponent.taylor_terms import (
    calculate_skycomponent_list_taylor_terms,
    find_skycomponents_frequency_taylor_terms,
    gather_skycomponents_from_channels,
    interpolate_skycomponents_frequency,
    transpose_skycomponents_to_channels,
)

log = logging.getLogger("func-python-logger")

log.setLevel(logging.WARNING)


@pytest.fixture(scope="module", name="input_params")
def taylor_terms_fixture():

    phase_centre = SkyCoord(
        ra=+180.0 * u.deg,
        dec=-60.0 * u.deg,
        frame="icrs",
        equinox="J2000",
    )

    frequency = numpy.array([1.1e8])
    name = "test_sc"
    flux = numpy.ones((1, 1))
    shape = "Point"
    polarisation_frame = PolarisationFrame("stokesI")

    sky_components = SkyComponent(
        phase_centre,
        frequency,
        name,
        flux,
        shape,
        polarisation_frame,
    )
    params = {
        "phasecentre": phase_centre,
        "frequency": frequency,
        "skycomponents": sky_components,
    }
    return params


def test_calculate_taylor_terms(input_params):
    """Check interpolate_list = 2 as, 2 skycomponents given"""
    sc = input_params["skycomponents"]
    sc_list = [sc, sc]

    taylor_term_list = calculate_skycomponent_list_taylor_terms(
        sc_list,
        nmoment=3,
    )
    assert len(taylor_term_list) == 2


@pytest.mark.skip(
    reason="This function uses many taylor_terms functions,"
    "testing those individually"
)
def test_find_skycomponents_frequency_taylor_terms(input_params):

    im = create_image(
        npixel=512,
        cellsize=0.00015,
        phasecentre=input_params["phasecentre"],
        nchan=1,
    )
    im["pixels"].data = numpy.ones(shape=im["pixels"].data.shape)

    im_list = [im]

    sc_list = find_skycomponents_frequency_taylor_terms(im_list, nmoment=3)
    assert len(sc_list) == 2
    assert len(sc_list[0]) == 3


def test_interpolate_skycomponents_frequency(input_params):
    """Check interpolate_list = 2 as, 2 skycomponents given"""
    sc = input_params["skycomponents"]
    sc_list = [sc, sc]

    interpolate_list = interpolate_skycomponents_frequency(
        sc_list,
        nmoment=3,
    )

    assert len(interpolate_list) == 2


def test_transpose_skycomponents_to_channels(input_params):
    """Check that transpose list returns list of len = 1, as here nchan = 1"""

    sc = input_params["skycomponents"]
    sc_list = [sc, sc]
    transpose_list = transpose_skycomponents_to_channels(
        sc_list,
    )

    assert len(transpose_list) == 1


def test_gather_skycomponents_from_channels(input_params):
    """Check gather_list1/2 = 2/3, as there are 2/3 skycomponents
    in each list"""
    sc = input_params["skycomponents"]
    sc_list1 = [sc, sc]
    sc_list2 = [sc, sc, sc]
    sc_list_of_lists1 = [sc_list1, sc_list1]
    sc_list_of_lists2 = [sc_list2, sc_list2, sc_list2]
    gather_list1 = gather_skycomponents_from_channels(
        sc_list_of_lists1,
    )
    gather_list2 = gather_skycomponents_from_channels(
        sc_list_of_lists2,
    )

    assert len(gather_list1) == 2
    assert len(gather_list2) == 3
