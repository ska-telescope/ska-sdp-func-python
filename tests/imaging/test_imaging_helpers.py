"""
Unit tests for imaging functions
"""
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

from ska_sdp_func_python.imaging.base import normalise_sumwt
from ska_sdp_func_python.imaging.imaging_helpers import (
    remove_sumwt,
    sum_invert_results,
    sum_predict_results,
    threshold_list,
)


@pytest.fixture(scope="module", name="input_params")
def imaging_helpers_fixture():
    """Fixture to generate inputs for tested functions"""
    phase_centre = SkyCoord(
        ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
    )
    test_image = create_image(512, 0.000015, phase_centre)

    image_single_list = [(test_image, 2.0)]
    image_multiple_list = [
        (test_image, 1),
        (test_image, 1),
        (test_image, 1),
    ]
    config = create_named_configuration("LOWBD2")
    integration_time = numpy.pi * (24 / (12 * 60))
    times = numpy.linspace(
        -integration_time * (3 // 2),
        integration_time * (3 // 2),
        3,
    )
    vis = create_visibility(
        config,
        frequency=numpy.array([1.0e8]),
        channel_bandwidth=numpy.array([4e7]),
        times=times,
        polarisation_frame=PolarisationFrame("stokesI"),
        phasecentre=phase_centre,
    )

    vis_list = [vis, vis, vis]

    params = {
        "image": test_image,
        "single_list": image_single_list,
        "multiple_list": image_multiple_list,
        "visibility_list": vis_list,
    }
    return params


def test_sum_invert_results_single_list(input_params):
    """Sum invert results of a single image"""
    im, smwt = sum_invert_results(input_params["single_list"])
    assert im == input_params["image"]
    assert smwt == 2.0


@pytest.mark.skip(reason="shape issue when incrementing im[pixels].data:")
def test_sum_invert_results_multiple_list(input_params):
    """Sum invert results of multiple images"""
    im, smwt = sum_invert_results(input_params["multiple_list"])
    expected_image = normalise_sumwt(input_params["image"], 3)

    assert im == expected_image
    assert smwt == 3


def test_remove_sumwt(input_params):
    """Test removing sumwt from tuple"""
    ims_only_list = remove_sumwt(input_params["multiple_list"])

    assert ims_only_list[0] == input_params["image"]


def test_sum_predict_results(input_params):
    """Test summing predict results"""
    sum_results = sum_predict_results(input_params["visibility_list"])

    assert (
        sum_results["vis"].data
        == 3 * input_params["visibility_list"][0]["vis"].data
    ).all()


def test_threshold_list(input_params):
    """Test finding a threshold for a list of images"""
    image = input_params["image"]
    image_list = [image, image, image]
    actual_threshold = threshold_list(
        image_list,
        threshold=0.0,
        fractional_threshold=0.01,
    )
    expected_data = numpy.max(
        numpy.abs(
            input_params["image"]["pixels"].data[0, ...]
            / input_params["image"]["pixels"].shape[0]
        )
    )

    assert (actual_threshold == expected_data * 0.01).all()
