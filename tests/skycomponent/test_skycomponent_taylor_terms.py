# pylint: disable=invalid-name, too-many-arguments
# pylint: disable=invalid-envvar-default
# pylint: disable= missing-class-docstring, missing-function-docstring
# pylint: disable=import-error, no-name-in-module
""" Unit tests for image Taylor terms

"""
import pytest
# Needs copy_skycomponent and decision about smooth_image and create_low_test_skycomponents_from_gleam
pytestmark = pytest.skip(allow_module_level=True)
import logging

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from ska_sdp_func_python.skycomponent.taylor_terms import (
    calculate_skycomponent_list_taylor_terms,
    find_skycomponents_frequency_taylor_terms,
)
from ska_sdp_datamodels.image.image_create import create_image

# fix the below imports
from src.ska_sdp_func_python import (
    create_low_test_skycomponents_from_gleam,
    smooth_image,
)

log = logging.getLogger("func-python-logger")

log.setLevel(logging.WARNING)


@pytest.fixture(scope="module", name="result_taylor_terms")
def taylor_terms_fixture():
    phasecentre = SkyCoord(
        ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame="icrs", equinox="J2000"
    )
    frequency = numpy.linspace(0.9e8, 1.1e8, 9)
    params = {
        "phasecentre": phasecentre,
        "frequency": frequency,
    }
    return params


@pytest.mark.skip(reason="create_low_test_skycomponent_from_glean needs to be replaced")
def test_calculate_taylor_terms(result_taylor_terms):
    sc = create_low_test_skycomponents_from_gleam(
        phasecentre=result_taylor_terms["phasecentre"],
        frequency=result_taylor_terms["frequency"],
        flux_limit=10.0
    )[0:10]

    taylor_term_list = calculate_skycomponent_list_taylor_terms(
        sc, nmoment=3
    )
    assert len(taylor_term_list) == 10


def test_find_skycomponents_frequency_taylor_terms(result_taylor_terms):
    im_list = [
        create_image(
            npixel=512,
            cellsize=0.001,
            phasecentre=result_taylor_terms["phasecentre"],
            frequency=[f],
        )
        for f in result_taylor_terms["frequency"]
    ]
    # im_list = [smooth_image(im, width=2.0) for im in im_list]

    for moment in [1, 2, 3]:
        sc_list = find_skycomponents_frequency_taylor_terms(
            im_list, nmoment=moment, component_threshold=20.0
        )
        assert len(sc_list) == 9
        assert len(sc_list[0]) == 3
