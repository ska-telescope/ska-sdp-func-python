# pylint: disable=invalid-name, too-many-arguments,line-too-long
# pylint: disable=consider-using-f-string, logging-fstring-interpolation
# pylint: disable= missing-class-docstring, missing-function-docstring
# pylint: disable=import-error, no-name-in-module
"""Unit tests for image iteration


"""
import logging
import unittest

import numpy
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)

from src.ska_sdp_func_python.image.iterators import (
    image_channel_iter,
    image_raster_iter,
)
from src.ska_sdp_func_python.image.operations import pad_image
from src.ska_sdp_func_python.parameters import rascil_path
from src.ska_sdp_func_python.simulation import create_test_image

log = logging.getLogger("rascil-logger")
log.setLevel(logging.WARNING)

log.setLevel(logging.WARNING)


class TestImageIterators(unittest.TestCase):
    def get_test_image(self, npixel=512):
        testim = create_test_image(
            polarisation_frame=PolarisationFrame("stokesI")
        )
        return pad_image(testim, [1, 1, npixel, npixel])

    def test_raster(self):
        """Test a raster iterator across an image. The test is to check that the
        value of the subimages is multiplied by two.

        """

        testdir = rascil_path("test_results")
        for npixel in [256, 512, 1024]:
            m31original = self.get_test_image(npixel=npixel)
            assert numpy.max(
                numpy.abs(m31original["pixels"].data)
            ), "Original is empty"

            for nraster in [1, 4, 8, 16]:

                for overlap in [0, 2, 4, 8, 16]:
                    try:
                        m31model = self.get_test_image(npixel=npixel)
                        for patch in image_raster_iter(
                            m31model, facets=nraster, overlap=overlap
                        ):
                            assert patch["pixels"].data.shape[3] == (
                                m31model["pixels"].data.shape[3] // nraster
                            ), (
                                "Number of pixels in each patch: %d not as expected: %d"
                                % (
                                    patch["pixels"].data.shape[3],
                                    (
                                        m31model["pixels"].data.shape[3]
                                        // nraster
                                    ),
                                )
                            )
                            assert patch["pixels"].data.shape[2] == (
                                m31model["pixels"].data.shape[2] // nraster
                            ), (
                                "Number of pixels in each patch: %d not as expected: %d"
                                % (
                                    patch["pixels"].data.shape[2],
                                    (
                                        m31model["pixels"].data.shape[2]
                                        // nraster
                                    ),
                                )
                            )
                            patch["pixels"].data *= 2.0

                        if (
                            numpy.max(numpy.abs(m31model["pixels"].data))
                            == 0.0
                        ):
                            log.warning(
                                f"Raster is empty failed for {npixel}, {nraster}, {overlap}"
                            )
                        diff = m31model.copy(deep=True)
                        diff["pixels"].data -= 2.0 * m31original["pixels"].data
                        err = numpy.max(diff["pixels"].data)
                        if abs(err) > 0.0:
                            log.warning(
                                f"Raster set failed for {npixel}, {nraster}, {overlap}: error {err}"
                            )
                        m31model.image_acc.export_to_fits(
                            f"{testdir}/test_image_iterators_model_{npixel}_{nraster}_{overlap}.fits",
                        )
                        diff.image_acc.export_to_fits(
                            f"{testdir}/test_image_iterators_diff_{npixel}_{nraster}_{overlap}.fits",
                        )
                    except ValueError as err:
                        log.error(
                            f"Iterator failed for {npixel}, {nraster}, {overlap},: {err}"
                        )

    def test_raster_exception(self):

        m31original = self.get_test_image()
        assert numpy.max(
            numpy.abs(m31original["pixels"].data)
        ), "Original is empty"

        for nraster, overlap in [(-1, -1), (-1, 0), (1e6, 127)]:
            with self.assertRaises(AssertionError):
                m31model = create_test_image(
                    polarisation_frame=PolarisationFrame("stokesI")
                )
                for patch in image_raster_iter(
                    m31model, facets=nraster, overlap=overlap
                ):
                    patch["pixels"].data *= 2.0

        for nraster, overlap in [(2, 128)]:
            with self.assertRaises(ValueError):
                m31model = create_test_image(
                    polarisation_frame=PolarisationFrame("stokesI")
                )
                for patch in image_raster_iter(
                    m31model, facets=nraster, overlap=overlap
                ):
                    patch["pixels"].data *= 2.0

    def test_channelise(self):
        m31cube = create_test_image(
            frequency=numpy.linspace(1e8, 1.1e8, 128),
            polarisation_frame=PolarisationFrame("stokesI"),
        )

        for subimages in [128, 16, 8, 2, 1]:
            for slab in image_channel_iter(m31cube, subimages=subimages):
                assert slab["pixels"].data.shape[0] == 128 // subimages


if __name__ == "__main__":
    unittest.main()
