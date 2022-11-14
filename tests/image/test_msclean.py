""" Unit processing_components for image deconvolution via MSClean


"""
import logging

import numpy
import pytest

from ska_sdp_func_python.image.cleaners import (
    argmax,
    convolve_convolve_scalestack,
    convolve_scalestack,
    create_scalestack,
)

log = logging.getLogger("func-python-logger")

log.setLevel(logging.WARNING)


@pytest.fixture(scope="module", name="msclean_params")
def msclean_fixture():
    npixel = 256
    scales = [0.0, 8.0 / numpy.sqrt(2.0), 8.0]
    stackshape = [len(scales), npixel, npixel]
    scalestack = create_scalestack(stackshape, scales)
    params = {
        "npixel": npixel,
        "stackshape": stackshape,
        "scalestack": scalestack,
    }
    return params


def test_convolve(msclean_params):
    img = numpy.zeros([msclean_params["npixel"], msclean_params["npixel"]])
    img[75, 31] = 1.0
    result = convolve_scalestack(msclean_params["scalestack"], img)
    assert argmax(result)[1:] == (75, 31)
    numpy.testing.assert_array_almost_equal(
        result[0, 75, 31],
        msclean_params["scalestack"][
            0, msclean_params["npixel"] // 2, msclean_params["npixel"] // 2
        ],
        7,
    )
    numpy.testing.assert_array_almost_equal(
        result[1, 75, 31],
        msclean_params["scalestack"][
            1, msclean_params["npixel"] // 2, msclean_params["npixel"] // 2
        ],
        7,
    )
    numpy.testing.assert_array_almost_equal(
        result[2, 75, 31],
        msclean_params["scalestack"][
            2, msclean_params["npixel"] // 2, msclean_params["npixel"] // 2
        ],
        7,
    )


def test_convolve_convolve(msclean_params):
    img = numpy.zeros([msclean_params["npixel"], msclean_params["npixel"]])
    img[75, 31] = 1.0
    result = convolve_convolve_scalestack(msclean_params["scalestack"], img)
    assert argmax(result)[2:] == (75, 31)
    numpy.testing.assert_array_almost_equal(
        result[0, 0, 75, 31],
        msclean_params["scalestack"][
            0, msclean_params["npixel"] // 2, msclean_params["npixel"] // 2
        ],
        7,
    )
    numpy.testing.assert_array_almost_equal(
        result[0, 1, 75, 31],
        msclean_params["scalestack"][
            1, msclean_params["npixel"] // 2, msclean_params["npixel"] // 2
        ],
        7,
    )
    numpy.testing.assert_array_almost_equal(
        result[0, 2, 75, 31],
        msclean_params["scalestack"][
            2, msclean_params["npixel"] // 2, msclean_params["npixel"] // 2
        ],
        7,
    )
    # This is a coarse test since the scales do not
    # having the property of widths adding incoherently under
    # convolution
    numpy.testing.assert_array_almost_equal(
        result[1, 1, 75, 31],
        msclean_params["scalestack"][
            2, msclean_params["npixel"] // 2, msclean_params["npixel"] // 2
        ],
        2,
    )
