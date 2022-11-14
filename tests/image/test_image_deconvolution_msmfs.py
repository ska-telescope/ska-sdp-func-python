# pylint: disable=duplicate-code
"""Unit tests for image deconvolution vis MSMFS


"""
import logging

import astropy.units as u
import numpy
import pytest
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.configuration import (
    create_named_configuration,
    decimate_configuration,
)
from ska_sdp_datamodels.image.image_create import create_image
from ska_sdp_datamodels.image.image_model import Image
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)
from ska_sdp_datamodels.visibility import create_visibility

from ska_sdp_func_python.image.deconvolution import (
    deconvolve_list,
    restore_list,
)
from ska_sdp_func_python.image.gather_scatter import (
    image_gather_channels,
    image_scatter_channels,
)
from ska_sdp_func_python.imaging.base import create_image_from_visibility
from ska_sdp_func_python.imaging.imaging import (
    invert_visibility,
    predict_visibility,
)
from ska_sdp_func_python.imaging.weighting import (
    taper_visibility_gaussian,
    weight_visibility,
)

pytest.skip(allow_module_level=True, reason="FixMe")
# fix the below imports
# from ska_sdp_func_python import create_pb
# from ska_sdp_func_python.imaging.primary_beams import create_low_test_beam

log = logging.getLogger("func-python-logger")

log.setLevel(logging.INFO)


@pytest.fixture(scope="module", name="result_deconv_msmfs")
def deconvolution_msmfs_fixture():
    niter = 1000
    lowcore = create_named_configuration("LOWBD2-CORE")
    lowcore = decimate_configuration(lowcore, skip=3)
    nchan = 6
    times = (numpy.pi / 12.0) * numpy.linspace(-3.0, 3.0, 7)
    frequency = numpy.linspace(0.9e8, 1.1e8, nchan)
    channel_bandwidth = numpy.array(nchan * [frequency[1] - frequency[0]])
    phasecentre = SkyCoord(
        ra=+0.0 * u.deg, dec=-45.0 * u.deg, frame="icrs", equinox="J2000"
    )
    vis = create_visibility(
        config=lowcore,
        times=times,
        frequency=frequency,
        channel_bandwidth=channel_bandwidth,
        phasecentre=phasecentre,
        weight=1.0,
        polarisation_frame=PolarisationFrame("stokesI"),
        zerow=True,
    )
    vis["vis"].data *= 0.0

    # Create image
    test_image = create_image(
        npixel=256,
        cellsize=0.001,
        phasecentre=vis.phasecentre,
        polarisation_frame=PolarisationFrame("stokesI"),
        frequency=frequency,
        channel_bandwidth=channel_bandwidth,
    )
    # beam = create_low_test_beam(test_image)
    # test_image["pixels"].data *= beam["pixels"].data
    vis = predict_visibility(vis, test_image, context="2d")
    assert numpy.max(numpy.abs(vis.vis)) > 0.0
    model = create_image_from_visibility(
        vis,
        npixel=512,
        cellsize=0.001,
        polarisation_frame=PolarisationFrame("stokesI"),
    )
    vis = weight_visibility(vis, model)
    vis = taper_visibility_gaussian(vis, 0.002)
    dirty, sumwt = invert_visibility(vis, model, context="2d")
    psf, sumwt = invert_visibility(vis, model, context="2d", dopsf=True)
    dirty = image_scatter_channels(dirty)
    psf = image_scatter_channels(psf)
    window = numpy.ones(shape=model["pixels"].shape, dtype=bool)
    window[..., 65:192, 65:192] = True
    innerquarter = Image.constructor(
        window,
        polarisation_frame=PolarisationFrame("stokesI"),
        wcs=model.image_acc.wcs,
    )
    innerquarter = image_scatter_channels(innerquarter)
    # sensitivity = create_pb(model, "LOW")
    # sensitivity = image_scatter_channels(sensitivity)
    params = {
        "dirty": dirty,
        "niter": niter,
        "psf": psf,
    }
    return params


def test_deconvolve_mmclean_no_taylor(result_deconv_msmfs):
    comp, residual = deconvolve_list(
        result_deconv_msmfs["dirty"],
        result_deconv_msmfs["psf"],
        niter=result_deconv_msmfs["niter"],
        gain=0.1,
        algorithm="mmclean",
        scales=[0, 3, 10],
        threshold=0.01,
        nmoment=1,
        findpeak="RASCIL",
        fractional_threshold=0.01,
        window_shape="quarter",
    )
    cmodel = restore_list(comp, result_deconv_msmfs["psf"], residual)
    check_images(
        "mmclean_no_taylor",
        comp,
        residual,
        cmodel,
        12.806085871833158,
        -0.14297206892008504,
    )


def test_deconvolve_mmclean_no_taylor_edge(result_deconv_msmfs):
    comp, residual = deconvolve_list(
        result_deconv_msmfs["dirty"],
        result_deconv_msmfs["psf"],
        niter=result_deconv_msmfs["niter"],
        gain=0.1,
        algorithm="mmclean",
        scales=[0, 3, 10],
        threshold=0.01,
        nmoment=1,
        findpeak="RASCIL",
        fractional_threshold=0.01,
        window_shape="no_edge",
        window_edge=32,
    )
    cmodel = restore_list(comp, result_deconv_msmfs["psf"], residual)
    check_images(
        "mmclean_no_taylor_edge",
        comp,
        residual,
        cmodel,
        12.806085871833158,
        -0.1429720689200851,
    )


def test_deconvolve_mmclean_no_taylor_noscales(result_deconv_msmfs):
    comp, residual = deconvolve_list(
        result_deconv_msmfs["dirty"],
        result_deconv_msmfs["psf"],
        niter=result_deconv_msmfs["niter"],
        gain=0.1,
        algorithm="mmclean",
        scales=[0],
        threshold=0.01,
        nmoment=1,
        findpeak="RASCIL",
        fractional_threshold=0.01,
        window_shape="quarter",
    )
    cmodel = restore_list(comp, result_deconv_msmfs["psf"], residual)
    check_images(
        "mmclean_notaylor_noscales",
        comp,
        residual,
        cmodel,
        12.874215203967717,
        -0.14419436344642067,
    )


def test_deconvolve_mmclean_linear(result_deconv_msmfs):
    comp, residual = deconvolve_list(
        result_deconv_msmfs["dirty"],
        result_deconv_msmfs["psf"],
        niter=result_deconv_msmfs["niter"],
        gain=0.1,
        algorithm="mmclean",
        scales=[0, 3, 10],
        threshold=0.01,
        nmoment=2,
        findpeak="RASCIL",
        fractional_threshold=0.01,
        window_shape="quarter",
    )
    cmodel = restore_list(comp, result_deconv_msmfs["psf"], residual)
    check_images(
        "mmclean_linear",
        comp,
        residual,
        cmodel,
        15.207396524333546,
        -0.14224980487729696,
    )


def test_deconvolve_mmclean_linear_sensitivity(result_deconv_msmfs):
    comp, residual = deconvolve_list(
        result_deconv_msmfs["dirty"],
        result_deconv_msmfs["psf"],
        sensitivity=result_deconv_msmfs["sensitivity"],
        niter=result_deconv_msmfs["niter"],
        gain=0.1,
        algorithm="mmclean",
        scales=[0, 3, 10],
        threshold=0.01,
        nmoment=2,
        findpeak="RASCIL",
        fractional_threshold=0.01,
        window_shape="quarter",
    )
    cmodel = restore_list(comp, result_deconv_msmfs["psf"], residual)
    check_images(
        "mmclean_linear_sensitivity",
        comp,
        residual,
        cmodel,
        15.207396524333546,
        -0.14224980487729716,
    )


def test_deconvolve_mmclean_linear_noscales(result_deconv_msmfs):
    comp, residual = deconvolve_list(
        result_deconv_msmfs["dirty"],
        result_deconv_msmfs["psf"],
        niter=result_deconv_msmfs["niter"],
        gain=0.1,
        algorithm="mmclean",
        scales=[0],
        threshold=0.01,
        nmoment=2,
        findpeak="RASCIL",
        fractional_threshold=0.01,
        window_shape="quarter",
    )
    cmodel = restore_list(comp, result_deconv_msmfs["psf"], residual)
    check_images(
        "mmclean_linear_noscales",
        comp,
        residual,
        cmodel,
        15.554039669750269,
        -0.14697685168807129,
    )


def test_deconvolve_mmclean_quadratic(result_deconv_msmfs):
    comp, residual = deconvolve_list(
        result_deconv_msmfs["dirty"],
        result_deconv_msmfs["psf"],
        niter=result_deconv_msmfs["niter"],
        gain=0.1,
        algorithm="mmclean",
        scales=[0, 3, 10],
        threshold=0.01,
        nmoment=3,
        findpeak="RASCIL",
        fractional_threshold=0.01,
        window_shape="quarter",
    )
    cmodel = restore_list(comp, result_deconv_msmfs["psf"], residual)
    check_images(
        "mmclean_quadratic",
        comp,
        residual,
        cmodel,
        15.302992891627193,
        -0.15373682171426403,
    )


def test_deconvolve_mmclean_quadratic_noscales(result_deconv_msmfs):
    comp, residual = deconvolve_list(
        result_deconv_msmfs["dirty"],
        result_deconv_msmfs["psf"],
        niter=result_deconv_msmfs["niter"],
        gain=0.1,
        algorithm="mmclean",
        scales=[0],
        threshold=0.01,
        nmoment=3,
        findpeak="RASCIL",
        fractional_threshold=0.01,
        window_shape="quarter",
    )
    cmodel = restore_list(comp, result_deconv_msmfs["psf"], residual)
    check_images(
        "mmclean_quadratic_noscales",
        comp,
        residual,
        cmodel,
        15.69172353540307,
        -0.1654330930047646,
    )


def check_images(
    result_deconv_msmfs,
    tag,
    comp,
    residual,
    cmodel,
    flux_max=0.0,
    flux_min=0.0,
):
    """Save the images with standard names

    :param tag: Informational, unique tag
    :return:
    """
    cmodel = image_gather_channels(cmodel)
    qa = cmodel.image_acc.qa_image()
    numpy.testing.assert_allclose(
        qa.data["max"], flux_max, atol=1e-7, err_msg=f"{qa}"
    )
    numpy.testing.assert_allclose(
        qa.data["min"], flux_min, atol=1e-7, err_msg=f"{qa}"
    )


def test_deconvolve_mmclean_quadratic_psf_support(result_deconv_msmfs):
    comp, residual = deconvolve_list(
        result_deconv_msmfs["dirty"],
        result_deconv_msmfs["psf"],
        niter=result_deconv_msmfs["niter"],
        gain=0.1,
        algorithm="mmclean",
        scales=[0, 3, 10],
        threshold=0.01,
        nmoment=3,
        findpeak="RASCIL",
        fractional_threshold=0.01,
        window_shape="quarter",
        psf_support=32,
    )
    cmodel = restore_list(comp, result_deconv_msmfs["psf"], residual)
    check_images(
        "mmclean_quadratic_psf",
        comp,
        residual,
        cmodel,
        15.322874439605584,
        -0.23892365313457908,
    )
