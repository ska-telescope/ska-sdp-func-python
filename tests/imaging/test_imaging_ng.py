""" Unit tests for imaging using nifty gridder

"""
import logging
import os
import sys
import tempfile

import numpy
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.configuration.config_create import (
    create_named_configuration,
)
from ska_sdp_datamodels.image.image_create import create_image
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)
from ska_sdp_datamodels.visibility.vis_create import create_visibility

from ska_sdp_func_python.imaging.ng import invert_ng, predict_ng

log = logging.getLogger("func-python-logger")

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))


@pytest.fixture(scope="module", name="result_ng")
def ng_fixture():

    persist = os.getenv("FUNC_PYTHON_PERSIST", False)
    verbosity = 0
    npixel = 256
    low = create_named_configuration("LOWBD2", rmax=750.0)
    ntimes = 5
    times = numpy.linspace(-3.0, +3.0, ntimes) * numpy.pi / 12.0
    frequency = numpy.array([1e8])
    channelwidth = numpy.array([1e6])

    vis_pol = PolarisationFrame("stokesI")
    f = numpy.array([100.0])

    phase_centre = SkyCoord(
        ra=+180.0 * u.deg, dec=-45.0 * u.deg, frame="icrs", equinox="J2000"
    )
    vis = create_visibility(
        low,
        times,
        frequency,
        phase_centre,
        channelwidth,
        polarisation_frame=vis_pol,
    )

    model = create_image(
        npixel=npixel, cellsize=0.00015, phasecentre=phase_centre
    )

    params = {
        "model": model,
        "persist": persist,
        "verbosity": verbosity,
        "visibility": vis,
    }
    return params


def test_predict_ng(result_ng):

    vis = result_ng["visibility"]
    model = result_ng["model"]
    verbosity = result_ng["verbosity"]
    persist = result_ng["persist"]
    original_vis = vis.copy(deep=True)
    vis = predict_ng(vis, model, verbosity=verbosity)
    vis["vis"].data = vis["vis"].data - original_vis["vis"].data
    dirty = invert_ng(
        vis,
        model,
        dopsf=False,
        normalise=True,
        verbosity=verbosity,
    )

    if persist:
        with tempfile.TemporaryDirectory() as tempdir:
            dirty[0].image_acc.export_to_fits(
                f"{tempdir}/test_imaging_ng_predict_ng_residual.fits"
            )

    maxabs = numpy.max(numpy.abs(dirty[0]["pixels"].data))
    assert maxabs < 1, "Error %.3f greater than fluxthreshold %.3f " % (
        maxabs,
        1,
    )


@pytest.mark.skip(reason="Need an non-empty image")
def test_invert_ng(result_ng):
    vis = result_ng["visibility"]
    model = result_ng["model"]
    verbosity = result_ng["verbosity"]
    persist = result_ng["persist"]
    dirty = invert_ng(
        vis,
        model,
        normalise=True,
        verbosity=verbosity,
    )

    if persist:
        with tempfile.TemporaryDirectory() as tempdir:
            dirty[0].image_acc.export_to_fits(
                f"{tempdir}/test_imaging_ng_invert_ng_dirty.fits"
            )
    assert numpy.max(numpy.abs(dirty[0]["pixels"].data)), "Image is empty"


def test_invert_ng_psf(result_ng):
    vis = result_ng["visibility"]
    model = result_ng["model"]
    verbosity = result_ng["verbosity"]
    persist = result_ng["persist"]
    dirty = invert_ng(
        vis,
        model,
        normalise=True,
        dopsf=True,
        verbosity=verbosity,
    )

    if persist:
        with tempfile.TemporaryDirectory() as tempdir:
            dirty[0].image_acc.export_to_fits(
                f"{tempdir}/test_imaging_ng_invert_ng_psf_dirty.fits"
            )
    assert numpy.max(numpy.abs(dirty[0]["pixels"].data)), "Image is empty"
