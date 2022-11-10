# pylint: skip-file
""" Unit tests for image Taylor terms

"""
import pytest

pytestmark = pytest.skip(
    allow_module_level=True, reason="Needs copy_skycomponents"
)
import logging

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.image.image_create import create_image
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)
from ska_sdp_datamodels.sky_model.sky_model import SkyComponent

from ska_sdp_func_python.skycomponent.taylor_terms import (
    calculate_skycomponent_list_taylor_terms,
    find_skycomponents_frequency_taylor_terms,
)

log = logging.getLogger("func-python-logger")

log.setLevel(logging.WARNING)


@pytest.fixture(scope="module", name="result_taylor_terms")
def taylor_terms_fixture():

    phase_centre = SkyCoord(
        ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame="icrs", equinox="J2000"
    )
    frequency = numpy.linspace(0.9e8, 1.1e8, 9)
    name = "test_sc"
    flux = numpy.array([1, 1])
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


def test_calculate_taylor_terms(result_taylor_terms):
    sc = result_taylor_terms["skycomponents"]

    taylor_term_list = calculate_skycomponent_list_taylor_terms(sc, nmoment=3)
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

    for moment in [1, 2, 3]:
        sc_list = find_skycomponents_frequency_taylor_terms(
            im_list, nmoment=moment, component_threshold=20.0
        )
        assert len(sc_list) == 9
        assert len(sc_list[0]) == 3
