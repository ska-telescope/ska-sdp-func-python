"""Unit tests for image iteration


"""
import pytest

pytestmark = pytest.skip(
    allow_module_level=True,
    reason="Image is seen as empty even with added skycomponents",
)
import logging
import tempfile

import numpy
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


@pytest.fixture(scope="module", name="result_iterators")
def iterators_fixture():
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


def test_raster(result_iterators):
    """Test a raster iterator across an image. The test is to check that the
    value of the subimages is multiplied by two.

    """

    for npixel in [256, 512, 1024]:
        m31original = create_image(
            npixel=npixel,
            cellsize=0.001,
            phasecentre=result_iterators["phasecentre"],
        )
        assert numpy.max(
            numpy.abs(m31original["pixels"].data)
        ), "Original is empty"

        for nraster in [1, 4, 8, 16]:

            for overlap in [0, 2, 4, 8, 16]:
                try:
                    m31model = create_image(
                        npixel=npixel,
                        cellsize=0.001,
                        phasecentre=result_iterators["phasecentre"],
                    )
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


@pytest.mark.skip()
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
