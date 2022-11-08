# pylint: disable=invalid-name, too-many-arguments,line-too-long
# pylint: disable=missing-class-docstring, missing-function-docstring
# pylint: disable=consider-using-f-string, logging-fstring-interpolation
# pylint: disable=import-error, no-name-in-module
"""Unit tests for image iteration


"""
import pytest

# Need to fix pad_image import and use pytest.parameterise instead of get_test_image()
pytestmark = pytest.skip(allow_module_level=True)

import logging
import tempfile

import numpy
from astropy import units
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.image.image_create import create_image

from src.ska_sdp_func_python.image.iterators import (
    image_channel_iter,
    image_raster_iter,
)

# fix the below imports
from src.ska_sdp_func_python.image.operations import pad_image

log = logging.getLogger("func-python-logger")
log.setLevel(logging.WARNING)

log.setLevel(logging.WARNING)


@pytest.fixture(scope="module", name="result_iterators")
def iterators_fixture():
    phase_centre = SkyCoord(
        ra=+180.0 * units.deg,
        dec=-35.0 * units.deg,
        frame="icrs",
        equinox="J2000",
    )
    im = create_image(
        npixel,
        cellsize=0.000015,
        phasecentre=phase_centre,
    )
    return pad_image(im, [1, 1, npixel, npixel])


def test_raster(result_iterators):
    """Test a raster iterator across an image. The test is to check that the
    value of the subimages is multiplied by two.

    """

    for npixel in [256, 512, 1024]:
        m31original = get_test_image(npixel=npixel)
        assert numpy.max(
            numpy.abs(m31original["pixels"].data)
        ), "Original is empty"

        for nraster in [1, 4, 8, 16]:

            for overlap in [0, 2, 4, 8, 16]:
                try:
                    m31model = get_test_image(npixel=npixel)
                    for patch in image_raster_iter(
                        m31model, facets=nraster, overlap=overlap
                    ):
                        assert patch["pixels"].data.shape[3] == (
                            m31model["pixels"].data.shape[3] // nraster
                        ), (
                            "Number of pixels in each patch: %d not as expected: %d"
                            % (
                                patch["pixels"].data.shape[3],
                                (m31model["pixels"].data.shape[3] // nraster),
                            )
                        )
                        assert patch["pixels"].data.shape[2] == (
                            m31model["pixels"].data.shape[2] // nraster
                        ), (
                            "Number of pixels in each patch: %d not as expected: %d"
                            % (
                                patch["pixels"].data.shape[2],
                                (m31model["pixels"].data.shape[2] // nraster),
                            )
                        )
                        patch["pixels"].data *= 2.0

                    if numpy.max(numpy.abs(m31model["pixels"].data)) == 0.0:
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
                    with tempfile.TemporaryDirectory() as testdir:
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


def test_raster_exception(result_iterators):

    m31original = result_iterators
    assert numpy.max(
        numpy.abs(m31original["pixels"].data)
    ), "Original is empty"

    for nraster, overlap in [(-1, -1), (-1, 0), (1e6, 127)]:
        with assertRaises(AssertionError):
            m31model = result_iterators
            for patch in image_raster_iter(
                m31model, facets=nraster, overlap=overlap
            ):
                patch["pixels"].data *= 2.0

    for nraster, overlap in [(2, 128)]:
        with assertRaises(ValueError):
            m31model = result_iterators
            for patch in image_raster_iter(
                m31model, facets=nraster, overlap=overlap
            ):
                patch["pixels"].data *= 2.0


def test_channelise(result_iterators):
    m31cube = result_iterators

    for subimages in [128, 16, 8, 2, 1]:
        for slab in image_channel_iter(m31cube, subimages=subimages):
            assert slab["pixels"].data.shape[0] == 128 // subimages
