"""
Unit tests for imaging functions
"""
import numpy

from ska_sdp_func_python.imaging.imaging_helpers import (
    remove_sumwt,
    sum_invert_results,
    sum_predict_results,
    threshold_list,
)


def test_sum_invert_results_single_list(image):
    """Sum invert results of a single image"""
    nchan = image.image_acc.nchan
    npol = image.image_acc.npol
    sumwt = numpy.ones((nchan, npol)) * 2.0
    image_list = [(image, sumwt)]

    result_im, result_smwt = sum_invert_results(image_list)
    assert result_im == image
    assert (result_smwt == sumwt).all()


def test_sum_invert_results_multiple_list(image):
    """Sum invert results of multiple images"""
    # image.pixels is all 0.0
    img_copy = image.copy(
        deep=True, data={"pixels": image.data_vars["pixels"] + 2.0}
    )
    nchan = image.image_acc.nchan
    npol = image.image_acc.npol
    sumwt = numpy.ones((nchan, npol)) * 2.0

    # expected pixels is the sum of pixels of all input images
    expected_image = image.copy(
        deep=True, data={"pixels": image.data_vars["pixels"] + 6.0}
    )

    img_list = [
        (img_copy, sumwt),
        (img_copy, sumwt),
        (img_copy, sumwt),
    ]
    result_img, result_sumwt = sum_invert_results(img_list)

    assert result_img == expected_image
    assert (result_sumwt == 3 * sumwt).all()


def test_remove_sumwt(image):
    """Test removing sumwt from tuple"""
    nchan = image.image_acc.nchan
    npol = image.image_acc.npol
    sumwt = numpy.ones((nchan, npol)) * 2.0

    image_list = [(image, sumwt), (image, sumwt), (image, sumwt)]
    ims_only_list = remove_sumwt(image_list)

    assert len(ims_only_list) == 3
    for im in ims_only_list:
        assert im == image


def test_sum_predict_results(visibility):
    """Test summing predict results"""
    sum_results = sum_predict_results([visibility, visibility, visibility])

    assert (sum_results["vis"].data == 3 * visibility["vis"].data).all()


def test_threshold_list(image):
    """Test finding a threshold for a list of images"""
    image_list = [image, image, image]
    actual_threshold = threshold_list(
        image_list,
        threshold=0.0,
        fractional_threshold=0.01,
    )
    expected_data = numpy.max(
        numpy.abs(
            image.data_vars["pixels"].data[0, ...]
            / image.data_vars["pixels"].shape[0]
        )
    )

    assert (actual_threshold == expected_data * 0.01).all()
