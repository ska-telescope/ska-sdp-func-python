# pylint: disable=invalid-name, too-many-arguments, duplicate-code
# pylint: disable=invalid-envvar-default, consider-using-f-string
# pylint: disable= missing-class-docstring, missing-function-docstring
# pylint: disable=import-error, no-name-in-module, import-outside-toplevel
""" Unit tests for image Taylor terms

"""
import logging
import os
import tempfile
import unittest

import numpy

from ska_sdp_func_python.image.gather_scatter import image_scatter_channels
from ska_sdp_func_python.image.taylor_terms import (
    calculate_frequency_taylor_terms_from_image_list,
    calculate_image_frequency_moments,
    calculate_image_from_frequency_taylor_terms,
)

# fix the below imports
from src.ska_sdp_func_python import create_empty_image_like
from src.ska_sdp_func_python.simulation import (
    create_low_test_image_from_gleam,
    create_test_image,
)

log = logging.getLogger("func-python-logger")

log.setLevel(logging.WARNING)


class TestImage(unittest.TestCase):
    def setUp(self):
        self.m31image = create_test_image()

        # assert numpy.max(self.m31image["pixels"]) > 0.0, "Test image is empty"
        self.cellsize = 180.0 * 0.0001 / numpy.pi
        self.persist = os.getenv("FUNC_PYTHON_PERSIST", False)

    def test_calculate_image_frequency_moments(self):
        frequency = numpy.linspace(0.9e8, 1.1e8, 9)
        original_cube = create_low_test_image_from_gleam(
            npixel=512, cellsize=0.0001, frequency=frequency, flux_limit=1.0
        )
        cube = create_empty_image_like(original_cube)
        moment_cube = calculate_image_frequency_moments(cube, nmoment=3)
        reconstructed_cube = calculate_image_from_frequency_taylor_terms(
            cube, moment_cube
        )
        if self.persist:
            with tempfile.TemporaryDirectory() as tempdir:
                original_cube.image_acc.export_to_fits(
                    f"{tempdir}/test_moments_cube.fits"
                )
                moment_cube.image_acc.export_to_fits(
                    f"{tempdir}/test_moments_moment_cube.fits"
                )
                reconstructed_cube.image_acc.export_to_fits(
                    f"{tempdir}/test_moments_reconstructed_cube.fits"
                )
        error = numpy.std(
            reconstructed_cube["pixels"].data - original_cube["pixels"].data
        )
        assert error < 0.2, error

    def test_calculate_image_frequency_moments_1(self):
        frequency = numpy.linspace(0.9e8, 1.1e8, 9)
        original_cube = create_low_test_image_from_gleam(
            npixel=512, cellsize=0.0001, frequency=frequency, flux_limit=1.0
        )
        cube = create_empty_image_like(original_cube)
        moment_cube = calculate_image_frequency_moments(cube, nmoment=1)
        reconstructed_cube = calculate_image_from_frequency_taylor_terms(
            cube, moment_cube
        )

        if self.persist:
            with tempfile.TemporaryDirectory() as tempdir:
                original_cube.image_acc.export_to_fits(
                    f"{tempdir}/test_moments_1_cube.fits"
                )
                moment_cube.image_acc.export_to_fits(
                    f"{tempdir}/test_moments_1_moment_cube.fits"
                )
                reconstructed_cube.image_acc.export_to_fits(
                    f"{tempdir}/test_moments_1_reconstructed_cube.fits"
                )
        error = numpy.std(
            reconstructed_cube["pixels"].data - original_cube["pixels"].data
        )
        assert error < 0.2

    def test_calculate_taylor_terms(self):
        frequency = numpy.linspace(0.9e8, 1.1e8, 9)
        original_cube = create_low_test_image_from_gleam(
            npixel=512, cellsize=0.0001, frequency=frequency, flux_limit=1.0
        )
        original_list = image_scatter_channels(original_cube)
        taylor_term_list = calculate_frequency_taylor_terms_from_image_list(
            original_list, nmoment=3
        )
        assert len(taylor_term_list) == 3


if __name__ == "__main__":
    unittest.main()
