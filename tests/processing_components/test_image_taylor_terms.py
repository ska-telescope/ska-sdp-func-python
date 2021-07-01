""" Unit tests for image Taylor terms

"""
import logging
import os
import unittest

import numpy

from rascil.processing_components.image.taylor_terms import (
    calculate_image_frequency_moments,
    calculate_image_from_frequency_taylor_terms,
)
from rascil.processing_components import (
    create_empty_image_like,
)
from rascil.processing_components.image.operations import (
    export_image_to_fits,
)
from rascil.processing_components.simulation import (
    create_test_image,
    create_low_test_image_from_gleam,
)

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)


class TestImage(unittest.TestCase):
    def setUp(self):

        from rascil.data_models.parameters import rascil_path

        self.dir = rascil_path("test_results")

        self.m31image = create_test_image()

        # assert numpy.max(self.m31image["pixels"]) > 0.0, "Test image is empty"
        self.cellsize = 180.0 * 0.0001 / numpy.pi
        self.persist = os.getenv("RASCIL_PERSIST", False)

    def test_calculate_image_frequency_moments(self):
        frequency = numpy.linspace(0.9e8, 1.1e8, 9)
        original_cube = create_low_test_image_from_gleam(
            npixel=512, cellsize=0.0001, frequency=frequency, flux_limit=1.0
        )
        if self.persist:
            export_image_to_fits(
                original_cube, fitsfile="%s/test_moments_cube.fits" % (self.dir)
            )
        cube = create_empty_image_like(original_cube)
        moment_cube = calculate_image_frequency_moments(cube, nmoment=3)
        print(moment_cube.image_acc.wcs)
        if self.persist:
            export_image_to_fits(
                moment_cube, fitsfile="%s/test_moments_moment_cube.fits" % (self.dir)
            )
        reconstructed_cube = calculate_image_from_frequency_taylor_terms(
            cube, moment_cube
        )
        print(reconstructed_cube.image_acc.wcs)
        if self.persist:
            export_image_to_fits(
                reconstructed_cube,
                fitsfile="%s/test_moments_reconstructed_cube.fits" % (self.dir),
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
        if self.persist:
            export_image_to_fits(
                original_cube, fitsfile="%s/test_moments_1_cube.fits" % (self.dir)
            )
        cube = create_empty_image_like(original_cube)
        moment_cube = calculate_image_frequency_moments(cube, nmoment=1)
        if self.persist:
            export_image_to_fits(
                moment_cube, fitsfile="%s/test_moments_1_moment_cube.fits" % (self.dir)
            )
        reconstructed_cube = calculate_image_from_frequency_taylor_terms(
            cube, moment_cube
        )
        if self.persist:
            export_image_to_fits(
                reconstructed_cube,
                fitsfile="%s/test_moments_1_reconstructed_cube.fits" % (self.dir),
            )
        error = numpy.std(
            reconstructed_cube["pixels"].data - original_cube["pixels"].data
        )
        assert error < 0.2


if __name__ == "__main__":
    unittest.main()
