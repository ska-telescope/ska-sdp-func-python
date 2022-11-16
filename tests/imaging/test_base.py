"""
Unit tests for base imaging functions
"""

import logging

import numpy
import pytest
from astropy import units
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.configuration.config_create import (
    create_named_configuration,
)
from ska_sdp_datamodels.image.image_create import create_image
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)
from ska_sdp_datamodels.visibility.vis_create import create_visibility

from ska_sdp_func_python.imaging.base import (
    create_image_from_visibility,
    fill_vis_for_psf,
    invert_awprojection,
    normalise_sumwt,
    predict_awprojection,
    shift_vis_to_image,
    visibility_recentre,
)

log = logging.getLogger("func-python-logger")


@pytest.fixture(scope="module", name="input_params")
def base_fixture():
    """Fixture to generate inputs for tested functions"""
    lowcore = create_named_configuration("LOWBD2-CORE")
    times = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 30.0)
    frequency = numpy.linspace(0.8e8, 1.0e8, 5)
    channel_bandwidth = numpy.array([1e7, 1e7, 1e7, 1e7, 1e7])
    polarisation_frame = PolarisationFrame("stokesI")
    phase_centre = SkyCoord(
        ra=+180.0 * units.deg,
        dec=-35.0 * units.deg,
        frame="icrs",
        equinox="J2000",
    )

    vis = create_visibility(
        lowcore,
        times,
        frequency,
        channel_bandwidth=channel_bandwidth,
        polarisation_frame=polarisation_frame,
        phasecentre=phase_centre,
        weight=1.0,
    )
    npixels = 512
    cellsize = 0.00015
    phase_centre = SkyCoord(
        ra=-180.0 * units.deg,
        dec=+35.0 * units.deg,
        frame="icrs",
        equinox="J2000",
    )
    im = create_image(npixels, cellsize, phase_centre)

    params = {
        "visibility": vis,
        "image": im,
    }
    return params


def test_shift_vis_to_image(input_params):
    """Unit tests for shift_vis_to_image function:
    check that the phasecentre does change
    """
    vis = input_params["visibility"]
    old_pc = vis.attrs["phasecentre"]
    shifted_vis = shift_vis_to_image(vis, input_params["image"])
    expected_phase_centre = SkyCoord(
        ra=-180.0 * units.deg,
        dec=+35.0 * units.deg,
        frame="icrs",
        equinox="J2000",
    )

    assert old_pc != shifted_vis.attrs["phasecentre"]
    assert shifted_vis.attrs["phasecentre"] == expected_phase_centre


@pytest.mark.skip(reason="gcfcf examples needed for predict_awprojection")
def test_predict_awprojection(input_params):
    """
    Test predict_awprojection
    """
    vis = input_params["visibility"]
    svis = predict_awprojection(
        vis,
        input_params["image"],
    )

    assert vis != svis


def test_fill_vis_for_psf(input_params):
    """Unit tests for fill_vis_for_psf function"""
    svis = fill_vis_for_psf(input_params["visibility"])

    assert (svis["vis"].data[...] == 1.0 + 0.0j).all()


def test_create_image_from_visibility(input_params):
    """Unit tests for create_image_from_visibility function:
    check image created here is the same as image in result_base
    """
    phase_centre = SkyCoord(
        ra=-180.0 * units.deg,
        dec=+35.0 * units.deg,
        frame="icrs",
        equinox="J2000",
    )
    expected_image = input_params["image"]
    new_image = create_image_from_visibility(
        vis=input_params["visibility"],
        phasecentre=phase_centre,
    )

    assert (new_image == expected_image).all()


def test_normalise_sumwt(input_params):
    """Unit tests for normalise_sumwt function:
    check image created here is the same as image in result_base
    """
    image = input_params["image"]
    sumwt = image
    norm_image = normalise_sumwt(image, sumwt)

    assert image != norm_image


@pytest.mark.skip(reason="Need more info on gcfcf values")
def test_invert_awprojection(input_params):
    """Unit tests for normalise_sumwt function:
    check image created here is the same as image in result_base
    """
    vis = input_params["visibility"]
    image = input_params["image"]
    inverted_im = invert_awprojection(vis, image, gcfcf="")

    assert inverted_im != image


def test_visibility_recentre():
    """Unit tests for normalise_sumwt function:
    check image created here is the same as image in result_base
    """
    uvw = numpy.array([1, 2, 3])
    dl = 0.1
    dm = 0.5
    uvw_recentred = visibility_recentre(uvw, dl, dm)
    assert uvw_recentred[0] == 0.7
    assert uvw_recentred[1] == 0.5
    assert uvw_recentred[2] == 3
