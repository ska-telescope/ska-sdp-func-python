# pylint: disable=duplicate-code
# flake8: noqa: E203
""" Unit tests for image deconvolution


"""
import logging

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
from ska_sdp_func_python.skycomponent.operations import restore_skycomponent

log = logging.getLogger("func-python-logger")

log.setLevel(logging.INFO)

pytest.skip(
    allow_module_level=True,
    reason="Need to fix!",
)


@pytest.fixture(scope="module", name="input_params")
def deconvolution_fixture():
    """Pytest fixture for the deconvolution.py unit tests"""
    lowcore = create_named_configuration("LOWBD2-CORE")
    times = (numpy.pi / 12.0) * numpy.linspace(-3.0, 3.0, 7)
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
    test_model["pixels"].data = numpy.ones(
        shape=test_model["pixels"].data.shape, dtype=float
    )
    vis = predict_visibility(vis, test_model, context="2d")
    assert numpy.max(numpy.abs(vis.vis)) > 0.0
    model = create_image_from_visibility(
        vis,
        npixel=512,
        cellsize=0.001,
        polarisation_frame=PolarisationFrame("stokesI"),
    )
    dirty = invert_visibility(vis, model, context="2d")[0]
    dirty["pixels"].data = numpy.ones(
        shape=dirty["pixels"].data.shape, dtype=float
    )
    psf = invert_visibility(vis, model, context="2d", dopsf=True)[0]
    params = {
        "dirty": dirty,
        "model": model,
        "psf": psf,
    }
    return params


def overlap_test(a1, a2, s1, s2):
    a1[s1[0] : s1[1], s1[2] : s1[3]] = 1
    a2[s2[0] : s2[1], s2[2] : s2[3]] = 1
    return numpy.sum(a1) == numpy.sum(a2)


def test_overlap():
    """Unit tests for the overlapIndices function"""
    res = numpy.zeros([512, 512])
    psf = numpy.zeros([100, 100])
    peak = (499, 249)
    s1, s2 = overlapIndices(res, psf, peak[0], peak[1])
    assert len(s1) == 4
    assert len(s2) == 4
    overlap_test(res, psf, s1, s2)
    assert s1 == (449, 512, 199, 299)
    assert s2 == (0, 63, 0, 100)


def test_restore(input_params):
    """Unit tests for the restore_cube function"""
    input_params["model"].data_vars["pixels"].data[0, 0, 256, 256] = 1.0
    cmodel = restore_cube(input_params["model"], input_params["psf"])
    assert numpy.abs(numpy.max(cmodel["pixels"].data) - 1.0) < 1e-7, numpy.max(
        cmodel["pixels"].data
    )


def test_restore_list(input_params):
    """Unit tests for the restore_list function"""
    input_params["model"]["pixels"].data[0, 0, 256, 256] = 1.0
    cmodel = restore_list([input_params["model"]], [input_params["psf"]])[0]
    assert numpy.abs(numpy.max(cmodel["pixels"].data) - 1.0) < 1e-7, numpy.max(
        cmodel["pixels"].data
    )


def test_restore_clean_beam(input_params):
    """Test restoration with specified beam

    :return:
    """
    input_params["model"]["pixels"].data[0, 0, 256, 256] = 1.0
    # The beam is specified in degrees
    bmaj = 0.006 * 180.0 / numpy.pi
    cmodel = restore_cube(
        input_params["model"],
        input_params["psf"],
        clean_beam={"bmaj": bmaj, "bmin": bmaj, "bpa": 0.0},
    )
    assert numpy.abs(numpy.max(cmodel["pixels"].data) - 1.0) < 1e-7, numpy.max(
        cmodel["pixels"].data
    )


def test_restore_skycomponent(input_params):
    """Test restoration of single pixel and skycomponent"""
    input_params["model"]["pixels"].data[0, 0, 256, 256] = 0.5

    sc = SkyComponent(
        flux=numpy.ones((1, 1)),
        direction=SkyCoord(
            ra=+180.0 * u.deg,
            dec=-61.0 * u.deg,
            frame="icrs",
            equinox="J2000",
        ),
        shape="Point",
        frequency=numpy.array([1e8]),
        polarisation_frame=PolarisationFrame("stokesI"),
    )
    bmaj = 0.012 * 180.0 / numpy.pi
    clean_beam = {"bmaj": bmaj, "bmin": bmaj / 2.0, "bpa": 15.0}
    cmodel = restore_cube(input_params["model"], clean_beam=clean_beam)
    cmodel = restore_skycomponent(cmodel, sc, clean_beam=clean_beam)
    assert (
        numpy.abs(numpy.max(cmodel["pixels"].data) - 0.9959046879055156) < 1e-7
    ), numpy.max(cmodel["pixels"].data)


def test_fit_psf(input_param):
    """Unit tests for the fit_psf function"""
    clean_beam = fit_psf(input_param["psf"])

    assert (
        numpy.abs(clean_beam["bmaj"] - 0.24790689057765794) < 6.0e-6
    ), clean_beam["bmax"]
    assert (
        numpy.abs(clean_beam["bmin"] - 0.2371401153972545) < 6.0e-6
    ), clean_beam["bmin"]


def test_deconvolve_hogbom(input_params):
    """Unit tests for the deconvolve_cube function using hogbom"""
    comp, residual = deconvolve_cube(
        input_params["dirty"],
        input_params["psf"],
        niter=10000,
        gain=0.1,
        algorithm="hogbom",
        threshold=0.01,
    )
    assert numpy.max(residual["pixels"].data) < 1.2


def test_deconvolve_msclean(input_params):
    """Unit tests for the deconvolve_cube function using msclean"""
    comp, residual = deconvolve_cube(
        input_params["dirty"],
        input_params["psf"],
        niter=1000,
        gain=0.7,
        algorithm="msclean",
        scales=[0, 3, 10, 30],
        threshold=0.01,
    )
    assert numpy.max(residual["pixels"].data) < 1.2


def test_deconvolve_msclean_1scale(input_params):
    """Unit tests for the deconvolve_cube function using msclean and scale 1"""
    comp, residual = deconvolve_cube(
        input_params["dirty"],
        input_params["psf"],
        niter=10000,
        gain=0.1,
        algorithm="msclean",
        scales=[0],
        threshold=0.01,
    )
    assert numpy.max(residual["pixels"].data) < 1.2


def test_deconvolve_hogbom_no_edge(input_params):
    """Unit tests for the deconvolve_cube function using hogbom and no_edge"""
    comp, residual = deconvolve_cube(
        input_params["dirty"],
        input_params["psf"],
        window_shape="no_edge",
        niter=10000,
        gain=0.1,
        algorithm="hogbom",
        threshold=0.01,
    )
    assert numpy.max(residual["pixels"].data) < 1.2


def test_deconvolve_hogbom_inner_quarter(input_params):
    """Unit tests for the deconvolve_cube function using hogbom and quarter"""
    comp, residual = deconvolve_cube(
        input_params["dirty"],
        input_params["psf"],
        window_shape="quarter",
        niter=10000,
        gain=0.1,
        algorithm="hogbom",
        threshold=0.01,
    )
    assert numpy.max(residual["pixels"].data) < 1.2


def test_deconvolve_msclean_inner_quarter(input_params):
    """Unit tests for the deconvolve_cube function using msclean and quarter"""
    comp, residual = deconvolve_cube(
        input_params["dirty"],
        input_params["psf"],
        window_shape="quarter",
        niter=1000,
        gain=0.7,
        algorithm="msclean",
        scales=[0, 3, 10, 30],
        threshold=0.01,
    )
    assert numpy.max(residual["pixels"].data) < 1.2


def test_deconvolve_hogbom_subpsf(input_params):
    """Unit tests for the deconvolve_cube function"""
    comp, residual = deconvolve_cube(
        input_params["dirty"],
        input_params["psf"],
        psf_support=200,
        window_shape="quarter",
        niter=10000,
        gain=0.1,
        algorithm="hogbom",
        threshold=0.01,
    )
    assert numpy.max(residual["pixels"].data[..., 56:456, 56:456]) < 1.2


def test_deconvolve_msclean_subpsf(input_params):
    """Unit tests for the deconvolve_cube function"""
    comp, residual = deconvolve_cube(
        input_params["dirty"],
        input_params["psf"],
        psf_support=200,
        window_shape="quarter",
        niter=1000,
        gain=0.7,
        algorithm="msclean",
        scales=[0, 3, 10, 30],
        threshold=0.01,
    )
    assert numpy.max(residual["pixels"].data[..., 56:456, 56:456]) < 1.0


def _check_hogbom_kernel_list_test_results(component, residual):
    """Checkinf function used to test non zero values"""
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


def test_hogbom_kernel_list_single_dirty(input_params):
    """Unit tests for the find_window_list function"""
    prefix = "test_hogbom_list"
    dirty_list = [input_params["dirty"]]
    psf_list = [input_params["psf"]]
    window_list = find_window_list(dirty_list, prefix)

    comp_list, residual_list = hogbom_kernel_list(
        dirty_list, prefix, psf_list, window_list
    )

    assert len(comp_list) == 1
    assert len(residual_list) == 1
    _check_hogbom_kernel_list_test_results(comp_list[0], residual_list[0])


def test_hogbom_kernel_list_multiple_dirty(input_params):
    """
    Bugfix: hogbom_kernel_list produced an IndexError, when
    dirty_list has more than one elements, and those elements are
    for a single frequency each, and window_shape is None.
    """

    prefix = "test_hogbom_list"
    dirty_list = [input_params["dirty"], input_params["dirty"]]
    psf_list = [input_params["psf"], input_params["psf"]]
    window_list = find_window_list(dirty_list, prefix, window_shape=None)

    comp_list, residual_list = hogbom_kernel_list(
        dirty_list, prefix, psf_list, window_list
    )

    assert len(comp_list) == 2
    assert len(residual_list) == 2
    # because the two dirty images and psfs are the same,
    # the expected results are also the same
    _check_hogbom_kernel_list_test_results(comp_list[0], residual_list[0])
    _check_hogbom_kernel_list_test_results(comp_list[1], residual_list[1])


def test_hogbom_kernel_list_multiple_dirty_window_shape(input_params):
    """
    Bugfix: hogbom_kernel_list produced an IndexError.
    Test the second branch of the if statement
    when dirty_list has more than one elements.
    """
    prefix = "test_hogbom_list"
    dirty_list = [input_params["dirty"], input_params["dirty"]]
    psf_list = [input_params["psf"], input_params["psf"]]
    window_list = find_window_list(dirty_list, prefix, window_shape="quarter")

    comp_list, residual_list = hogbom_kernel_list(
        dirty_list, prefix, psf_list, window_list
    )

    assert len(comp_list) == 2
    assert len(residual_list) == 2
    # because the two dirty images and psfs are the same,
    # the expected results are also the same
    _check_hogbom_kernel_list_test_results(comp_list[0], residual_list[0])
    _check_hogbom_kernel_list_test_results(comp_list[1], residual_list[1])
