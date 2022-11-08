""" Unit tests for image operations

"""
import logging

import numpy
import pytest
from astropy import units
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.image.image_create import create_image

from ska_sdp_func_python.image.operations import (
    convert_clean_beam_to_degrees,
    convert_clean_beam_to_pixels,
)

log = logging.getLogger("func-python-logger")


@pytest.fixture(scope="module", name="operations_image")
def operations_fixture():
    # Create a visibility object

    npixels = 512
    cellsize = 0.000015
    phase_centre = SkyCoord(
        ra=+180.0 * units.deg,
        dec=-35.0 * units.deg,
        frame="icrs",
        equinox="J2000",
    )

    im = create_image(npixels, cellsize, phase_centre)
    return im


def test_convert_clean_beam_to_degrees(operations_image):
    """
    Unit test for the convert_clean_beam_to_degrees function
    """
    beam_pixels = (24.7, 49.4, -1.05)
    clean_beam = convert_clean_beam_to_degrees(operations_image, beam_pixels)
    expected_results = (
        numpy.rad2deg(beam_pixels[0])
        * (0.000015 * numpy.sqrt(8.0 * numpy.log(2.0))),
        numpy.rad2deg(beam_pixels[1])
        * (0.000015 * numpy.sqrt(8.0 * numpy.log(2.0))),
        numpy.rad2deg(beam_pixels[2]),
    )

    assert clean_beam["bmin"] == pytest.approx(expected_results[0])
    assert clean_beam["bmaj"] == pytest.approx(expected_results[1])
    assert clean_beam["bpa"] == pytest.approx(expected_results[2])


def test_convert_clean_beam_to_pixels(operations_image):
    """
    Unit test for the convert_clean_beam_to_pixels function
    """
    clean_beam = {"bmaj": 0.1, "bmin": 0.05, "bpa": -60.0}
    beam_pixels = convert_clean_beam_to_pixels(operations_image, clean_beam)
    expected_results = (
        numpy.deg2rad(clean_beam["bmin"])
        / (0.000015 * numpy.sqrt(8.0 * numpy.log(2.0))),
        numpy.deg2rad(clean_beam["bmaj"])
        / (0.000015 * numpy.sqrt(8.0 * numpy.log(2.0))),
        numpy.deg2rad(clean_beam["bpa"]),
    )

    assert beam_pixels[0] == pytest.approx(expected_results[0])
    assert beam_pixels[1] == pytest.approx(expected_results[1])
    assert beam_pixels[2] == pytest.approx(expected_results[2])
