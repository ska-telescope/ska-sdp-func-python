# pylint: disable=duplicate-code
"""Unit tests for image iteration


"""
import logging
import tempfile

import numpy
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.image.image_create import create_image
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)
from ska_sdp_datamodels.sky_model.sky_model import SkyComponent

from ska_sdp_func_python.image.iterators import (
    image_channel_iter,
    image_raster_iter,
)
from ska_sdp_func_python.skycomponent.operations import insert_skycomponent

log = logging.getLogger("func-python-logger")
log.setLevel(logging.WARNING)

log.setLevel(logging.WARNING)


@pytest.fixture(scope="module", name="input_params")
def iterators_fixture():
    """Fixture for the iterators.py unit tests"""
    phase_centre = SkyCoord(
        ra=+180.0 * u.deg,
        dec=-35.0 * u.deg,
        frame="icrs",
        equinox="J2000",
    )
    image = create_image(
        npixel=512,
        cellsize=0.00015,
        phasecentre=phase_centre,
    )

    image_sc = SkyComponent(
        direction=SkyCoord(
            ra=+110.0 * u.deg, dec=-40.0 * u.deg, frame="icrs", equinox="J2000"
        ),
        frequency=numpy.array([1e8]),
        name="image_sc",
        flux=numpy.ones((1, 1)),
        shape="Point",
        polarisation_frame=PolarisationFrame("stokesI"),
    )

    image = insert_skycomponent(image, image_sc)
    params = {
        "image": image,
        "phasecentre": phase_centre,
    }
    return params


def test_raster(input_params):
    """Test a raster iterator across an image. The test is to check that the
    value of the subimages is multiplied by two.

    """

    for npixel in [256, 512, 1024]:
        m31original = create_image(
            npixel=npixel,
            cellsize=0.00015,
            phasecentre=input_params["phasecentre"],
        )
        m31original["pixels"].data = numpy.ones(
            shape=m31original["pixels"].data.shape, dtype=float
        )
        assert numpy.max(
            numpy.abs(m31original["pixels"].data)
        ), "Original is empty"

        for nraster in [1]:

            for overlap in [0, 2, 4, 8, 16]:
                try:
                    m31model = create_image(
                        npixel=npixel,
                        cellsize=0.00015,
                        phasecentre=input_params["phasecentre"],
                    )
                    m31model["pixels"].data = numpy.ones(
                        shape=m31model["pixels"].data.shape, dtype=float
                    )
                    for patch in image_raster_iter(
                        m31model, facets=nraster, overlap=overlap
                    ):
                        assert patch["pixels"].data.shape[3] == (
                            m31model["pixels"].data.shape[3] // nraster
                        ), (
                            "Number of pixels in each patch: %d not as "
                            "expected: %d"
                            % (
                                patch["pixels"].data.shape[3],
                                (m31model["pixels"].data.shape[3] // nraster),
                            )
                        )
                        assert patch["pixels"].data.shape[2] == (
                            m31model["pixels"].data.shape[2] // nraster
                        ), (
                            "Number of pixels in each patch: %d not as "
                            "expected: %d"
                            % (
                                patch["pixels"].data.shape[2],
                                (m31model["pixels"].data.shape[2] // nraster),
                            )
                        )
                        patch["pixels"].data *= 2.0

                    if numpy.max(numpy.abs(m31model["pixels"].data)) == 0.0:
                        log.warning(
                            f"Raster is empty failed for {npixel}, {nraster},"
                            f"{overlap}"
                        )
                    diff = m31model.copy(deep=True)
                    diff["pixels"].data -= 2.0 * m31original["pixels"].data
                    err = numpy.max(diff["pixels"].data)
                    if abs(err) > 0.0:
                        log.warning(
                            f"Raster set failed for {npixel}, {nraster}, "
                            f"{overlap}: error {err}"
                        )
                    with tempfile.TemporaryDirectory() as testdir:
                        m31model.image_acc.export_to_fits(
                            f"{testdir}/test_image_iterators_model_{npixel}_"
                            f"{nraster}_{overlap}.fits",
                        )
                        diff.image_acc.export_to_fits(
                            f"{testdir}/test_image_iterators_diff_{npixel}_"
                            f"{nraster}_{overlap}.fits",
                        )
                except ValueError as err:
                    log.error(
                        f"Iterator failed for {npixel}, {nraster}, {overlap},:"
                        f" {err}"
                    )


def test_raster_exception(input_params):
    """Check that raster captures the right exceptions"""
    m31original = input_params["image"]
    m31original["pixels"].data = numpy.ones(
        shape=m31original["pixels"].data.shape, dtype=float
    )
    assert numpy.max(
        numpy.abs(m31original["pixels"].data)
    ), "Original is empty"

    for nraster, overlap in [(-1, -1), (-1, 0), (1e6, 127)]:
        with pytest.raises(AssertionError):
            m31model = input_params["image"]
            m31model["pixels"].data = numpy.ones(
                shape=m31model["pixels"].data.shape, dtype=float
            )
            for patch in image_raster_iter(
                m31model, facets=nraster, overlap=overlap
            ):
                patch["pixels"].data *= 2.0

    for nraster, overlap in [(2, 513)]:
        with pytest.raises(ValueError):
            m31model = input_params["image"]
            for patch in image_raster_iter(
                m31model, facets=nraster, overlap=overlap
            ):
                patch["pixels"].data *= 2.0


@pytest.mark.skip("Test uses image from file")
def test_channelise(result_iterators):
    """Unit test for the image_channel_iter function"""
    m31cube = create_test_image(
        frequency=numpy.linspace(1e8, 1.1e8, 128),
        polarisation_frame=PolarisationFrame("stokesI"),
    )

    for subimages in [128, 16, 8, 2, 1]:
        for slab in image_channel_iter(m31cube, subimages=subimages):
            assert slab["pixels"].data.shape[0] == 128 // subimages

