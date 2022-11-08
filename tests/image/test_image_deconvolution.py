# pylint: disable=invalid-name, too-many-arguments, too-many-public-methods
# pylint: disable=attribute-defined-outside-init, unused-variable
# pylint: disable=too-many-instance-attributes, invalid-envvar-default
# pylint: disable=consider-using-f-string, logging-not-lazy
# pylint: disable=missing-class-docstring, missing-function-docstring
# pylint: disable=import-error, no-name-in-module, import-outside-toplevel
""" Unit tests for image deconvolution


"""
import logging
import os
import tempfile

import astropy.units as u
import numpy
import pytest
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.configuration import create_named_configuration
from ska_sdp_datamodels.image.image_create import create_image
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)
from ska_sdp_datamodels.sky_model.sky_model import SkyComponent
from ska_sdp_datamodels.visibility import create_visibility

from ska_sdp_func_python.image.cleaners import overlapIndices
from ska_sdp_func_python.image.deconvolution import (
    deconvolve_cube,
    find_window_list,
    fit_psf,
    hogbom_kernel_list,
    restore_cube,
    restore_list,
)
from ska_sdp_func_python.imaging.base import create_image_from_visibility
from ska_sdp_func_python.imaging.imaging import (
    invert_visibility,
    predict_visibility,
)

# from ska_sdp_func_python.skycomponent.operations import restore_skycomponent

# fix the below imports
# from ska_sdp_func_python.imaging import create_pb

log = logging.getLogger("func-python-logger")

log.setLevel(logging.INFO)


@pytest.fixture(scope="module", name="result_deconvolution")
def deconvolution_fixture():

    persist = os.getenv("FUNC_PYTHON_PERSIST", False)
    lowcore = create_named_configuration("LOWBD2-CORE")
    times = (numpy.pi / (12.0)) * numpy.linspace(-3.0, 3.0, 7)
    frequency = numpy.array([1e8])
    channel_bandwidth = numpy.array([1e6])
    phase_centre = SkyCoord(
        ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
    )
    vis = create_visibility(
        lowcore,
        times,
        frequency,
        channel_bandwidth=channel_bandwidth,
        phasecentre=phase_centre,
        weight=1.0,
        polarisation_frame=PolarisationFrame("stokesI"),
        zerow=True,
    )
    vis["vis"].data *= 0.0

    # Create model
    test_model = create_image(
        npixel=512,
        cellsize=0.001,
        phasecentre=vis.phasecentre,
    )
    vis = predict_visibility(vis, test_model, context="2d")
    # assert numpy.max(numpy.abs(vis.vis)) > 0.0
    model = create_image_from_visibility(
        vis,
        npixel=512,
        cellsize=0.001,
        polarisation_frame=PolarisationFrame("stokesI"),
    )
    dirty = invert_visibility(vis, model, context="2d")[0]
    psf = invert_visibility(vis, model, context="2d", dopsf=True)[0]
    # sensitivity = create_pb(model, "LOW")
    params = {
        "dirty": dirty,
        "model": model,
        "persist": persist,
        "psf": psf,
        # "sensitivity": sensitivity,
    }
    return params


def overlaptest(a1, a2, s1, s2):
    #
    a1[s1[0] : s1[1], s1[2] : s1[3]] = 1
    a2[s2[0] : s2[1], s2[2] : s2[3]] = 1
    return numpy.sum(a1) == numpy.sum(a2)


def test_overlap():
    res = numpy.zeros([512, 512])
    psf = numpy.zeros([100, 100])
    peak = (499, 249)
    s1, s2 = overlapIndices(res, psf, peak[0], peak[1])
    assert len(s1) == 4
    assert len(s2) == 4
    overlaptest(res, psf, s1, s2)
    assert s1 == (449, 512, 199, 299)
    assert s2 == (0, 63, 0, 100)


def test_restore(result_deconvolution):
    result_deconvolution["model"].data_vars["pixels"].data[
        0, 0, 256, 256
    ] = 1.0
    cmodel = restore_cube(
        result_deconvolution["model"], result_deconvolution["psf"]
    )
    assert numpy.abs(numpy.max(cmodel["pixels"].data) - 1.0) < 1e-7, numpy.max(
        cmodel["pixels"].data
    )
    if result_deconvolution["persist"]:
        with tempfile.TemporaryDirectory() as tempdir:
            cmodel.image_acc.export_to_fits(f"{tempdir}/test_restore.fits")


def test_restore_list(result_deconvolution):
    result_deconvolution["model"]["pixels"].data[0, 0, 256, 256] = 1.0
    cmodel = restore_list(
        [result_deconvolution["model"]], [result_deconvolution["psf"]]
    )[0]
    assert numpy.abs(numpy.max(cmodel["pixels"].data) - 1.0) < 1e-7, numpy.max(
        cmodel["pixels"].data
    )
    if result_deconvolution["persist"]:
        with tempfile.TemporaryDirectory() as tempdir:
            cmodel.image_acc.export_to_fits(f"{tempdir}/test_restore.fits")


def test_restore_clean_beam(result_deconvolution):
    """Test restoration with specified beam beam

    :return:
    """
    result_deconvolution["model"]["pixels"].data[0, 0, 256, 256] = 1.0
    # The beam is specified in degrees
    bmaj = 0.006 * 180.0 / numpy.pi
    cmodel = restore_cube(
        result_deconvolution["model"],
        result_deconvolution["psf"],
        clean_beam={"bmaj": bmaj, "bmin": bmaj, "bpa": 0.0},
    )
    assert numpy.abs(numpy.max(cmodel["pixels"].data) - 1.0) < 1e-7, numpy.max(
        cmodel["pixels"].data
    )
    if result_deconvolution["persist"]:
        with tempfile.TemporaryDirectory() as tempdir:
            cmodel.image_acc.export_to_fits(
                f"{tempdir}/test_restore_6mrad_beam.fits"
            )


@pytest.mark.skip(reason="Import issues in SkyComponent/operations.py")
def test_restore_skycomponent(result_deconvolution):
    """Test restoration of single pixel and skycomponent"""
    result_deconvolution["model"]["pixels"].data[0, 0, 256, 256] = 0.5

    sc = SkyComponent(
        flux=numpy.array([[1.0]]),
        direction=SkyCoord(
            ra=+180.0 * u.deg,
            dec=-61.0 * u.deg,
            frame="icrs",
            equinox="J2000",
        ),
        shape="Point",
        frequency=result_deconvolution["frequency"],
        polarisation_frame=PolarisationFrame("stokesI"),
    )
    bmaj = 0.012 * 180.0 / numpy.pi
    clean_beam = {"bmaj": bmaj, "bmin": bmaj / 2.0, "bpa": 15.0}
    cmodel = restore_cube(result_deconvolution["model"], clean_beam=clean_beam)
    cmodel = restore_skycomponent(cmodel, sc, clean_beam=clean_beam)
    if result_deconvolution["persist"]:
        with tempfile.TemporaryDirectory() as tempdir:
            cmodel.image_acc.export_to_fits(
                f"{tempdir}/test_restore_skycomponent.fits"
            )
    assert (
        numpy.abs(numpy.max(cmodel["pixels"].data) - 0.9959046879055156) < 1e-7
    ), numpy.max(cmodel["pixels"].data)


def test_fit_psf(result_deconvolution):
    clean_beam = fit_psf(result_deconvolution["psf"])
    if result_deconvolution["persist"]:
        with tempfile.TemporaryDirectory() as tempdir:
            result_deconvolution["psf"].image_acc.export_to_fits(
                f"{tempdir}/test_fit_psf.fits"
            )
    # Sanity check: by eyeball the FHWM = 4 pixels = 0.004 rad = 0.229 deg
    assert (
        numpy.abs(clean_beam["bmaj"] - 0.24790689057765794) < 1.0e-7
    ), clean_beam
    assert (
        numpy.abs(clean_beam["bmin"] - 0.2371401153972545) < 1.0e-7
    ), clean_beam
    assert (
        numpy.abs(clean_beam["bpa"] + 1.0126425267576473) < 1.0e-7
    ), clean_beam


@pytest.mark.skip(
    reason="Issues with inputs for create_image in deconvolution.py"
)
def test_deconvolve_hogbom(result_deconvolution):
    comp, residual = deconvolve_cube(
        result_deconvolution["dirty"],
        result_deconvolution["psf"],
        niter=10000,
        gain=0.1,
        algorithm="hogbom",
        threshold=0.01,
    )
    cmodel = restore_cube(comp, result_deconvolution["psf"], residual)
    if result_deconvolution["persist"]:
        save_results("hogbom", comp, residual, cmodel)
    assert numpy.max(residual["pixels"].data) < 1.2


@pytest.mark.skip(
    reason="Issues with inputs for create_image in deconvolution.py"
)
def test_deconvolve_msclean(result_deconvolution):
    comp, residual = deconvolve_cube(
        result_deconvolution["dirty"],
        result_deconvolution["psf"],
        niter=1000,
        gain=0.7,
        algorithm="msclean",
        scales=[0, 3, 10, 30],
        threshold=0.01,
    )
    cmodel = restore_cube(comp, result_deconvolution["psf"], residual)
    if result_deconvolution["persist"]:
        save_results("msclean", comp, residual, cmodel)
    assert numpy.max(residual["pixels"].data) < 1.2


def save_results(tag, comp, residual, cmodel):
    with tempfile.TemporaryDirectory() as tempdir:
        comp.image_acc.export_to_fits(
            f"{tempdir}/test_deconvolve_{tag}-deconvolved.fits"
        )
        residual.image_acc.export_to_fits(
            f"{tempdir}/test_deconvolve_{tag}-residual.fits",
        )
        cmodel.image_acc.export_to_fits(
            f"{tempdir}/test_deconvolve_{tag}-restored.fits"
        )


@pytest.mark.skip(reason="Missing import create_pb")
def test_deconvolve_msclean_sensitivity(result_deconvolution):
    comp, residual = deconvolve_cube(
        result_deconvolution["dirty"],
        result_deconvolution["psf"],
        sensitivity=result_deconvolution["sensitivity"],
        niter=1000,
        gain=0.7,
        algorithm="msclean",
        scales=[0, 3, 10, 30],
        threshold=0.01,
    )
    cmodel = restore_cube(comp, result_deconvolution["psf"], residual)
    if result_deconvolution["persist"]:
        save_results("msclean-sensitivity", comp, residual, cmodel)

    qa = residual.image_acc.qa_image()
    numpy.testing.assert_allclose(
        qa.data["max"], 0.8040729590477751, atol=1e-7, err_msg=f"{qa}"
    )
    numpy.testing.assert_allclose(
        qa.data["min"], -0.9044553283128349, atol=1e-7, err_msg=f"{qa}"
    )


@pytest.mark.skip(
    reason="Issues with inputs for create_image in deconvolution.py"
)
def test_deconvolve_msclean_1scale(result_deconvolution):

    comp, residual = deconvolve_cube(
        result_deconvolution["dirty"],
        result_deconvolution["psf"],
        niter=10000,
        gain=0.1,
        algorithm="msclean",
        scales=[0],
        threshold=0.01,
    )
    cmodel = restore_cube(comp, result_deconvolution["psf"], residual)
    if result_deconvolution["persist"]:
        save_results("msclean-1scale", comp, residual, cmodel)
    assert numpy.max(residual["pixels"].data) < 1.2


@pytest.mark.skip(
    reason="Issues with inputs for create_image in deconvolution.py"
)
def test_deconvolve_hogbom_no_edge(result_deconvolution):
    comp, residual = deconvolve_cube(
        result_deconvolution["dirty"],
        result_deconvolution["psf"],
        window_shape="no_edge",
        niter=10000,
        gain=0.1,
        algorithm="hogbom",
        threshold=0.01,
    )
    cmodel = restore_cube(comp, result_deconvolution["psf"], residual)
    if result_deconvolution["persist"]:
        save_results("hogbom_no_edge", comp, residual, cmodel)
    assert numpy.max(residual["pixels"].data) < 1.2


@pytest.mark.skip(
    reason="Issues with inputs for create_image in deconvolution.py"
)
def test_deconvolve_hogbom_inner_quarter(result_deconvolution):
    comp, residual = deconvolve_cube(
        result_deconvolution["dirty"],
        result_deconvolution["psf"],
        window_shape="quarter",
        niter=10000,
        gain=0.1,
        algorithm="hogbom",
        threshold=0.01,
    )
    cmodel = restore_cube(comp, result_deconvolution["psf"], residual)
    if result_deconvolution["persist"]:
        save_results("hogbom_no_inner_quarter", comp, residual, cmodel)
    assert numpy.max(residual["pixels"].data) < 1.2


@pytest.mark.skip(
    reason="Issues with inputs for create_image in deconvolution.py"
)
def test_deconvolve_msclean_inner_quarter(result_deconvolution):

    comp, residual = deconvolve_cube(
        result_deconvolution["dirty"],
        result_deconvolution["psf"],
        window_shape="quarter",
        niter=1000,
        gain=0.7,
        algorithm="msclean",
        scales=[0, 3, 10, 30],
        threshold=0.01,
    )
    cmodel = restore_cube(comp, result_deconvolution["psf"], residual)
    if result_deconvolution["persist"]:
        save_results("msclean_inner_quarter", comp, residual, cmodel)
    assert numpy.max(residual["pixels"].data) < 1.2


@pytest.mark.skip(
    reason="Issues with inputs for create_image in deconvolution.py"
)
def test_deconvolve_hogbom_subpsf(result_deconvolution):

    comp, residual = deconvolve_cube(
        result_deconvolution["dirty"],
        result_deconvolution["psf"],
        psf_support=200,
        window_shape="quarter",
        niter=10000,
        gain=0.1,
        algorithm="hogbom",
        threshold=0.01,
    )
    cmodel = restore_cube(comp, result_deconvolution["psf"], residual)
    if result_deconvolution["persist"]:
        save_results("hogbom_subpsf", comp, residual, cmodel)
    assert numpy.max(residual["pixels"].data[..., 56:456, 56:456]) < 1.2


@pytest.mark.skip(
    reason="Issues with inputs for create_image in deconvolution.py"
)
def test_deconvolve_msclean_subpsf(result_deconvolution):

    comp, residual = deconvolve_cube(
        result_deconvolution["dirty"],
        result_deconvolution["psf"],
        psf_support=200,
        window_shape="quarter",
        niter=1000,
        gain=0.7,
        algorithm="msclean",
        scales=[0, 3, 10, 30],
        threshold=0.01,
    )
    cmodel = restore_cube(comp, result_deconvolution["psf"], residual)
    if result_deconvolution["persist"]:
        save_results("msclean_subpsf", comp, result_deconvolution, cmodel)
    assert numpy.max(residual["pixels"].data[..., 56:456, 56:456]) < 1.0


def _check_hogbom_kernel_list_test_results(component, residual):
    result_comp_data = component["pixels"].data
    non_zero_idx_comp = numpy.where(result_comp_data != 0.0)
    expected_comp_non_zero_data = numpy.array(
        [
            0.508339,
            0.590298,
            0.533506,
            0.579212,
            0.549127,
            0.622576,
            0.538019,
            0.717473,
            0.716564,
            0.840854,
        ]
    )
    result_residual_data = residual["pixels"].data
    non_zero_idx_residual = numpy.where(result_residual_data != 0.0)
    expected_residual_non_zero_data = numpy.array(
        [
            0.214978,
            0.181119,
            0.145942,
            0.115912,
            0.100664,
            0.106727,
            0.132365,
            0.167671,
            0.200349,
            0.222765,
        ]
    )

    # number of non-zero values
    assert len(result_comp_data[non_zero_idx_comp]) == 82
    assert len(result_residual_data[non_zero_idx_residual]) == 262144
    # test first 10 non-zero values don't change with each run of test
    numpy.testing.assert_array_almost_equal(
        result_comp_data[non_zero_idx_comp][:10],
        expected_comp_non_zero_data,
    )
    numpy.testing.assert_array_almost_equal(
        result_residual_data[non_zero_idx_residual][:10],
        expected_residual_non_zero_data,
    )


@pytest.mark.skip(
    reason="Issues with inputs for create_image in deconvolution.py"
)
def test_hogbom_kernel_list_single_dirty(result_deconvolution):
    prefix = "test_hogbom_list"
    dirty_list = [result_deconvolution["dirty"]]
    psf_list = [result_deconvolution["psf"]]
    window_list = find_window_list(dirty_list, prefix)

    comp_list, residual_list = hogbom_kernel_list(
        dirty_list, prefix, psf_list, window_list
    )

    assert len(comp_list) == 1
    assert len(residual_list) == 1
    _check_hogbom_kernel_list_test_results(comp_list[0], residual_list[0])


@pytest.mark.skip(
    reason="Issues with inputs for create_image in deconvolution.py"
)
def test_hogbom_kernel_list_multiple_dirty(result_deconvolution):
    """
    Bugfix: hogbom_kernel_list produced an IndexError, when dirty_list has more than
    one elements, and those elements are for a single frequency each, and window_shape is None.
    """

    prefix = "test_hogbom_list"
    dirty_list = [result_deconvolution["dirty"], result_deconvolution["dirty"]]
    psf_list = [result_deconvolution["psf"], result_deconvolution["psf"]]
    window_list = find_window_list(dirty_list, prefix, window_shape=None)

    comp_list, residual_list = hogbom_kernel_list(
        dirty_list, prefix, psf_list, window_list
    )

    assert len(comp_list) == 2
    assert len(residual_list) == 2
    # because the two dirty images and psfs are the same, the expected results are also the same
    _check_hogbom_kernel_list_test_results(comp_list[0], residual_list[0])
    _check_hogbom_kernel_list_test_results(comp_list[1], residual_list[1])


@pytest.mark.skip(reason="Incomplete test")
def test_hogbom_kernel_list_multiple_dirty_window_shape(result_deconvolution):
    """
    Bugfix: hogbom_kernel_list produced an IndexError.
    Test the second branch of the if statement
    when dirty_list has more than one elements.
    """
    test_hogbom_kernel_list_multiple_dirty(window_shape="quarter")
