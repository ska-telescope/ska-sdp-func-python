# pylint: disable=invalid-name, too-many-arguments, too-many-public-methods
# pylint: disable=attribute-defined-outside-init, unused-variable
# pylint: disable=too-many-instance-attributes, invalid-envvar-default
# pylint: disable=consider-using-f-string, logging-not-lazy
# pylint: disable=missing-class-docstring, missing-function-docstring
# pylint: disable=import-error, no-name-in-module, import-outside-toplevel
""" Unit tests for imaging functions


"""
import pytest

from astropy import units as u
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.image.image_create import create_image
from ska_sdp_func_python.imaging.base import normalise_sumwt
from ska_sdp_func_python.imaging.imaging_helpers import sum_invert_results, remove_sumwt


@pytest.fixture(scope="module", name="result_helpers")
def imaging_helpers_fixture():
    phase_centre = SkyCoord(
        ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
    )
    test_image = create_image(512, 0.000015, phase_centre)

    image_single_list = [(test_image, 2.0)]
    image_multiple_list = [(test_image, 1),
                           (test_image, 1),
                           (test_image, 1),
                           ]
    params = {
        "image": test_image,
        "single_list": image_single_list,
        "multiple_list": image_multiple_list,
    }
    return params


def test_sum_invert_results_single_list(result_helpers):

    im, smwt = sum_invert_results(result_helpers["single_list"])

    assert im == result_helpers["image"]
    assert smwt == 2.0


@pytest.mark.skip(reason="shape issue in imaging_helpers when incrementing im[pixels].data")
def test_sum_invert_results_multiple_list(result_helpers):

    im, smwt = sum_invert_results(result_helpers["multiple_list"])
    expected_image = normalise_sumwt(result_helpers["image"], 6)

    assert im == expected_image
    assert smwt == 3.0


def test_remove_sumwt(result_helpers):

    ims_only_list = remove_sumwt(result_helpers["multiple_list"])

    assert ims_only_list[0] == result_helpers["image"]


