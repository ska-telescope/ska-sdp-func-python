# pylint: disable=unused-variable, invalid-name, consider-using-f-string
""" Unit tests for visibility weighting
"""

import logging

import numpy
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.configuration import create_named_configuration
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)
from ska_sdp_datamodels.visibility import create_visibility

from ska_sdp_func_python.image.deconvolution import fit_psf
from ska_sdp_func_python.imaging.base import create_image_from_visibility
from ska_sdp_func_python.imaging.imaging import invert_visibility
from ska_sdp_func_python.imaging.weighting import (
    taper_visibility_gaussian,
    taper_visibility_tukey,
    weight_visibility,
)

log = logging.getLogger("func-python-logger")

log.setLevel(logging.WARNING)


@pytest.fixture(scope="module", name="result_weighting")
def weighting_fixture():
    """Fixture for weighting.py unit tests"""

    npixel = 512
    image_pol = PolarisationFrame("stokesI")
    lowcore = create_named_configuration("LOWBD2", rmax=600)
    times = (numpy.pi / 12.0) * numpy.linspace(-3.0, 3.0, 5)
    frequency = numpy.array([1e8])
    channel_bandwidth = numpy.array([1e7])
    vis_pol = PolarisationFrame("stokesI")
    f = numpy.array([100.0])
    numpy.array([f])

    phasecentre = SkyCoord(
        ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
    )

    componentvis = create_visibility(
        lowcore,
        times,
        frequency,
        channel_bandwidth=channel_bandwidth,
        phasecentre=phasecentre,
        weight=1.0,
        polarisation_frame=vis_pol,
    )

    componentvis["vis"].data *= 0.0

    # Create model
    model = create_image_from_visibility(
        componentvis,
        npixel=npixel,
        cellsize=0.0005,
        nchan=len(frequency),
        polarisation_frame=image_pol,
    )

    params = {
        "componentvis": componentvis,
        "model": model,
    }
    return params


def test_tapering_gaussian(result_weighting):
    """Apply a Gaussian taper to the visibility and check to see if
    the PSF size is close
    """
    size_required = 0.020
    result_weighting["componentvis"] = weight_visibility(
        result_weighting["componentvis"],
        result_weighting["model"],
        algoritm="uniform",
    )
    result_weighting["componentvis"] = taper_visibility_gaussian(
        result_weighting["componentvis"], beam=size_required
    )
    psf, sumwt = invert_visibility(
        result_weighting["componentvis"],
        result_weighting["model"],
        dopsf=True,
        context="2d",
    )
    fit = fit_psf(psf)

    assert (
        numpy.abs(fit["bmaj"] - 1.279952050682638) < 1
    ), "Fit should be %f, actually is %f" % (
        1.279952050682638,
        fit["bmaj"],
    )


def test_tapering_tukey(result_weighting):
    """Apply a Tukey window taper and output the psf and FT of the PSF.
       No quantitative check.

    :return:
    """
    result_weighting["componentvis"] = weight_visibility(
        result_weighting["componentvis"],
        result_weighting["model"],
        algorithm="uniform",
    )
    result_weighting["componentvis"] = taper_visibility_tukey(
        result_weighting["componentvis"], tukey=0.1
    )
    psf, sumwt = invert_visibility(
        result_weighting["componentvis"],
        result_weighting["model"],
        dopsf=True,
        context="2d",
    )
    fit = fit_psf(psf)
    assert (
        numpy.abs(fit["bmaj"] - 0.14492670913355402) < 1.0
    ), "Fit should be %f, actually is %f" % (
        0.14492670913355402,
        fit["bmaj"],
    )
