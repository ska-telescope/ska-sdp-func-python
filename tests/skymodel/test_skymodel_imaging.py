# pylint: disable=invalid-name, too-many-arguments
# pylint: disable=attribute-defined-outside-init, unused-variable
# pylint: disable=too-many-instance-attributes, invalid-envvar-default
# pylint: disable=consider-using-f-string, logging-not-lazy
# pylint: disable=missing-class-docstring, missing-function-docstring
# pylint: disable=import-error, no-name-in-module, import-outside-toplevel
""" Regression test for skymodel predict and invert functions
"""
import pytest
# Issues with the dft_skycomponent_visibility import in skymodel_imaging.py (ska-sdp-func) and copy_skycomponent()
pytestmark = pytest.skip(allow_module_level=True)
import logging
import os
import sys
import tempfile

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.configuration.config_create import (
    create_named_configuration,
)
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)
from ska_sdp_func_python.visibility.visibility_geometry import calculate_visibility_parallactic_angles
from ska_sdp_datamodels.image.image_create import create_image
from ska_sdp_datamodels.calibration.calibration_create import create_gaintable_from_visibility
from ska_sdp_datamodels.sky_model.sky_model import SkyModel
from ska_sdp_datamodels.visibility.vis_create import create_visibility

from ska_sdp_func_python.skymodel.skymodel_imaging import (
    skymodel_calibrate_invert,
    skymodel_predict_calibrate,
)

# fix the below imports (in imaging\primary_beam.py)
# from src.ska_sdp_func_python import (
#     convert_azelvp_to_radec,
#     create_low_test_beam,
# )

log = logging.getLogger("func-python-logger")

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))


@pytest.fixture(scope="module", nmae="result_imaging")
def skymodel_imaging_fixture():

    persist = os.getenv("FUNC_PYTHON_PERSIST", False)
    npixel = 512
    low = create_named_configuration("LOW", rmax=300.0)
    freqwin = 1
    dopol = False
    zerow = False
    vis = []
    ntimes = 3
    cellsize = 0.001
    radius = npixel * cellsize / 2.0
    # Choose the interval so that the maximum change in w is smallish
    integration_time = numpy.pi * (24 / (12 * 60))
    times = numpy.linspace(
        -integration_time * (ntimes // 2),
        integration_time * (ntimes // 2),
        ntimes,
    )

    if freqwin > 1:
        frequency = numpy.linspace(0.8e8, 1.2e8, freqwin)
        channelwidth = numpy.array(
            freqwin * [frequency[1] - frequency[0]]
        )
    else:
        frequency = numpy.array([1.0e8])
        channelwidth = numpy.array([4e7])

    if dopol:
        vis_pol = PolarisationFrame("linear")

    else:
        vis_pol = PolarisationFrame("stokesI")

    phase_centre = SkyCoord(
        ra=+30.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
    )
    vis = create_visibility(
        low,
        frequency,
        channelwidth,
        times,
        vis_pol,
        phase_centre,
        zerow=zerow,
    )
    gt = create_gaintable_from_visibility(vis, times, "T")
    im = create_image(npixel, cellsize, phase_centre)
    params = {
        "npixel": npixel,
        "cellsize": cellsize,
        "frequency": frequency,
        "image": im,
        "gaintable": gt,
        "phasecentre": phase_centre,
        "persist": persist,
        "radius": radius,
        "vis": vis,
    }

    return params


def test_predict_no_pb(result_imaging):
    """Predict with no primary beam"""
    skymodel = SkyModel.constructor(result_imaging["image"], None,
                                    result_imaging["gaintable"], "Test_mask", False)

    assert len(skymodel.components) == 11, len(
        skymodel.components
    )
    assert (
        numpy.max(numpy.abs(skymodel.image["pixels"].data)) > 0.0
    ), "Image is empty"

    skymodel_vis = skymodel_predict_calibrate(
        result_imaging["vis"], skymodel, context="ng"
    )
    qa = skymodel_vis.visibility_acc.qa_visibility()
    numpy.testing.assert_almost_equal(
        qa.data["maxabs"], 60.35140880932053, err_msg=str(qa)
    )


@pytest.mark.skip(reason="Skipping to not use various primary beam functions")
def test_predict_with_pb(result_imaging):
    """Test predict while applying a time-variable primary beam"""

    skymodel = SkyModel.constructor(result_imaging["image"], None,
                                    result_imaging["gaintable"], "Test_mask", False)

    assert len(skymodel.components) == 11, len(
        skymodel.components
    )
    assert (
        numpy.max(numpy.abs(skymodel.image["pixels"].data)) > 0.0
    ), "Image is empty"

    def get_pb(vis, model):
        pb = create_low_test_beam(model)
        pa = numpy.mean(calculate_visibility_parallactic_angles(vis))
        pb = convert_azelvp_to_radec(pb, model, pa)
        return pb

    skymodel_vis = skymodel_predict_calibrate(
        result_imaging["vis"],
        skymodel,
        context="ng",
        get_pb=get_pb,
    )
    qa = skymodel_vis.visibility_acc.qa_visibility()
    numpy.testing.assert_almost_equal(
        qa.data["maxabs"], 32.20530966848842, err_msg=str(qa)
    )


def test_invert_no_pb(result_imaging):
    """Test invert"""

    skymodel = SkyModel.constructor(result_imaging["image"], None,
                                    result_imaging["gaintable"], "Test_mask", False)

    assert len(skymodel.components) == 11, len(
        skymodel.components
    )
    assert (
        numpy.max(numpy.abs(skymodel.image["pixels"].data)) > 0.0
    ), "Image is empty"

    skymodel_vis = skymodel_predict_calibrate(
        result_imaging["vis"],
        skymodel,
        context="ng",
    )
    assert numpy.max(numpy.abs(skymodel_vis.vis)) > 0.0

    dirty, sumwt = skymodel_calibrate_invert(
        skymodel_vis,
        skymodel,
        normalise=True,
        flat_sky=False,
    )
    if result_imaging["persist"]:
        with tempfile.TemporaryDirectory() as tempdir:
            dirty.image_acc.export_to_fits(
                f"{tempdir}/test_skymodel_invert_dirty.fits"
            )
    qa = dirty.image_acc.qa_image()

    numpy.testing.assert_allclose(
        qa.data["max"], 4.179714181498791, atol=1e-7, err_msg=f"{qa}"
    )
    numpy.testing.assert_allclose(
        qa.data["min"], -0.33300435260339034, atol=1e-7, err_msg=f"{qa}"
    )


@pytest.mark.skip(reason="Skipping to not use various primary beam functions")
def test_invert_with_pb(result_imaging):
    """Test invert while applying a time-variable primary beam"""

    skymodel = SkyModel.constructor(result_imaging["image"], None,
                                    result_imaging["gaintable"], "Test_mask", False)

    assert len(skymodel.components) == 11, len(
        skymodel.components
    )
    assert (
        numpy.max(numpy.abs(skymodel.image["pixels"].data)) > 0.0
    ), "Image is empty"

    def get_pb(bvis, model):
        pb = create_low_test_beam(model)
        pa = numpy.mean(calculate_visibility_parallactic_angles(bvis))
        pb = convert_azelvp_to_radec(pb, model, pa)
        return pb

    skymodel_vis = skymodel_predict_calibrate(
        result_imaging["vis"],
        skymodel,
        context="ng",
        get_pb=get_pb,
    )
    assert numpy.max(numpy.abs(skymodel_vis.vis)) > 0.0

    skymodel = skymodel_calibrate_invert(
        skymodel_vis,
        skymodel,
        get_pb=get_pb,
        normalise=True,
        flat_sky=False,
    )
    if result_imaging["persist"]:
        with tempfile.TemporaryDirectory() as tempdir:
            skymodel[0].image_acc.export_to_fits(
                f"{tempdir}/test_skymodel_invert_flat_noise_dirty.fits"
            )
            skymodel[1].image_acc.export_to_fits(
                f"{tempdir}/test_skymodel_invert_flat_noise_sensitivity.fits"
            )
    qa = skymodel[0].image_acc.qa_image()

    numpy.testing.assert_allclose(
        qa.data["max"], 3.767454977596991, atol=1e-7, err_msg=f"{qa}"
    )
    numpy.testing.assert_allclose(
        qa.data["min"], -0.23958139130004705, atol=1e-7, err_msg=f"{qa}"
    )

    # Now repeat with flat_sky=True
    skymodel = skymodel_calibrate_invert(
        skymodel_vis,
        skymodel,
        get_pb=get_pb,
        normalise=True,
        flat_sky=True,
    )
    if result_imaging["persist"]:
        with tempfile.TemporaryDirectory() as tempdir:
            skymodel[0].image_acc.export_to_fits(
                f"{tempdir}/test_skymodel_invert_flat_sky_dirty.fits"
            )
            skymodel[1].image_acc.export_to_fits(
                f"{tempdir}/test_skymodel_invert_flat_sky_sensitivity.fits"
            )
    qa = skymodel[0].image_acc.qa_image()

    numpy.testing.assert_allclose(
        qa.data["max"], 4.025153684707801, atol=1e-7, err_msg=f"{qa}"
    )
    numpy.testing.assert_allclose(
        qa.data["min"], -0.24826345131847594, atol=1e-7, err_msg=f"{qa}"
    )


def test_predict_nocomponents(result_imaging):
    """Test predict with no components"""

    skymodel = SkyModel.constructor(result_imaging["image"], None,
                                    result_imaging["gaintable"], "Test_mask", False)

    skymodel.components = []

    assert (
        numpy.max(numpy.abs(skymodel.image["pixels"].data)) > 0.0
    ), "Image is empty"

    skymodel_vis = skymodel_predict_calibrate(
        result_imaging["vis"], skymodel, context="ng"
    )
    qa = skymodel_vis.visibility_acc.qa_visibility()
    numpy.testing.assert_almost_equal(
        qa.data["maxabs"], 39.916746503252156, err_msg=str(qa)
    )


def test_predict_noimage(result_imaging):
    """Test predict with no image"""

    skymodel = SkyModel.constructor(result_imaging["image"], None,
                                    result_imaging["gaintable"], "Test_mask", False)

    skymodel.image = None

    assert len(skymodel.components) == 11, len(
        skymodel.components
    )

    skymodel_vis = skymodel_predict_calibrate(
        result_imaging["vis"], skymodel, context="ng"
    )
    qa = skymodel_vis.visibility_acc.qa_visibility()
    numpy.testing.assert_almost_equal(
        qa.data["maxabs"], 20.434662306068372, err_msg=str(qa)
    )
