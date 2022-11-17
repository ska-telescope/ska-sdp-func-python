"""
Unit tests for image iteration
"""
import logging

import numpy
import pytest
from astropy import units
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.image.image_create import create_image

from src.ska_sdp_func_python.image.gather_scatter import (
    image_gather_channels,
    image_gather_facets,
    image_scatter_channels,
    image_scatter_facets,
)

log = logging.getLogger("func-python-logger")

log.setLevel(logging.WARNING)


@pytest.fixture(scope="module", name="phase_centre")
def phase_centre_fixture():
    """Fixture for the gather_scatter.py unit tests"""
    phase_centre = SkyCoord(
        ra=+180.0 * units.deg,
        dec=-35.0 * units.deg,
        frame="icrs",
        equinox="J2000",
    )
    return phase_centre


def test_scatter_gather_facet(phase_centre):
    """Unit test for the image_gather_facets function"""
    m31original = create_image(
        npixel=10,
        cellsize=0.00015,
        phasecentre=phase_centre,
    )
    m31original["pixels"].data = numpy.ones(
        shape=m31original["pixels"].data.shape, dtype=float
    )
    assert numpy.max(
        numpy.abs(m31original["pixels"].data)
    ), "Original is empty"

    nraster = 1
    m31model = create_image(
        npixel=512,
        cellsize=0.00015,
        phasecentre=phase_centre,
    )
    m31model["pixels"].data = numpy.ones(
        shape=m31model["pixels"].data.shape, dtype=float
    )
    image_list = image_scatter_facets(m31model, facets=nraster)
    for patch in image_list:
        assert patch["pixels"].data.shape[3] == (
            m31model["pixels"].data.shape[3] // nraster
        ), (
            f"Number of pixels in each patch: "
            f"{patch['pixels'].data.shape[3]} not as expected: "
            f"{(m31model['pixels'].data.shape[3] // nraster)}"
        )

        assert patch["pixels"].data.shape[2] == (
            m31model["pixels"].data.shape[2] // nraster
        ), (
            f"Number of pixels in each patch: "
            f"{patch['pixels'].data.shape[2]} not as expected: "
            f"{(m31model['pixels'].data.shape[2] // nraster)}"
        )

        patch["pixels"].data[...] = 1.0

    m31reconstructed = create_image(
        npixel=512,
        cellsize=0.00015,
        phasecentre=phase_centre,
    )
    m31reconstructed["pixels"].data = numpy.ones(
        shape=m31reconstructed["pixels"].data.shape, dtype=float
    )
    m31reconstructed = image_gather_facets(
        image_list, m31reconstructed, facets=nraster
    )
    flat = image_gather_facets(
        image_list, m31reconstructed, facets=nraster, return_flat=True
    )

    assert numpy.max(
        numpy.abs(flat["pixels"].data)
    ), f"Flat is empty for {nraster}"
    assert numpy.max(
        numpy.abs(m31reconstructed["pixels"].data)
    ), f"Raster is empty for {nraster}"


def test_scatter_gather_facet_overlap(phase_centre):
    """Unit test for the image_gather_facets function with overlap"""
    m31original = create_image(
        npixel=512,
        cellsize=0.00015,
        phasecentre=phase_centre,
    )
    m31original["pixels"].data = numpy.ones(
        shape=m31original["pixels"].data.shape, dtype=float
    )
    assert numpy.max(
        numpy.abs(m31original["pixels"].data)
    ), "Original is empty"

    for nraster, overlap in [(1, 0), (1, 8), (1, 16)]:
        m31model = create_image(
            npixel=512,
            cellsize=0.00015,
            phasecentre=phase_centre,
        )
        m31model["pixels"].data = numpy.ones(
            shape=m31model["pixels"].data.shape, dtype=float
        )
        image_list = image_scatter_facets(
            m31model, facets=nraster, overlap=overlap
        )
        for patch in image_list:
            assert patch["pixels"].data.shape[3] == (
                m31model["pixels"].data.shape[3] // nraster
            ), (
                f"Number of pixels in each patch: "
                f"{patch['pixels'].data.shape[3]} not as expected: "
                f"{(m31model['pixels'].data.shape[3] // nraster)}"
            )

            assert patch["pixels"].data.shape[2] == (
                m31model["pixels"].data.shape[2] // nraster
            ), (
                f"Number of pixels in each patch: "
                f"{patch['pixels'].data.shape[2]} not as expected: "
                f"{(m31model['pixels'].data.shape[2] // nraster)}"
            )
            patch["pixels"].data[...] = 1.0

        m31reconstructed = create_image(
            npixel=512,
            cellsize=0.00015,
            phasecentre=phase_centre,
        )
        m31reconstructed["pixels"].data = numpy.ones(
            shape=m31reconstructed["pixels"].data.shape, dtype=float
        )
        m31reconstructed = image_gather_facets(
            image_list, m31reconstructed, facets=nraster, overlap=overlap
        )
        flat = image_gather_facets(
            image_list,
            m31reconstructed,
            facets=nraster,
            overlap=overlap,
            return_flat=True,
        )

        assert numpy.max(
            numpy.abs(flat["pixels"].data)
        ), f"Flat is empty for {nraster}"
        assert numpy.max(
            numpy.abs(m31reconstructed["pixels"].data)
        ), f"Raster is empty for {nraster}"


def test_scatter_gather_facet_overlap_taper(phase_centre):
    """Unit test for the image_gather_facets function with overlap and taper"""
    m31original = create_image(
        npixel=512,
        cellsize=0.00015,
        phasecentre=phase_centre,
    )
    m31original["pixels"].data = numpy.ones(
        shape=m31original["pixels"].data.shape, dtype=float
    )
    assert numpy.max(
        numpy.abs(m31original["pixels"].data)
    ), "Original is empty"

    for taper in ["linear", "tukey", None]:
        for nraster, overlap in [
            (1, 0),
            (1, 1),
            (1, 8),
            (1, 4),
            (1, 8),
            (1, 8),
            (1, 16),
        ]:
            m31model = create_image(
                npixel=512,
                cellsize=0.00015,
                phasecentre=phase_centre,
            )
            m31model["pixels"].data = numpy.ones(
                shape=m31model["pixels"].data.shape, dtype=float
            )
            image_list = image_scatter_facets(
                m31model, facets=nraster, overlap=overlap, taper=taper
            )
            for patch in image_list:
                assert patch["pixels"].data.shape[3] == (
                    m31model["pixels"].data.shape[3] // nraster
                ), (
                    f"Number of pixels in each patch: "
                    f"{patch.data.shape[3]} not as expected: "
                    f"{(m31model['pixels'].data.shape[3] // nraster)}"
                )
                assert patch["pixels"].data.shape[2] == (
                    m31model["pixels"].data.shape[2] // nraster
                ), (
                    f"Number of pixels in each patch: "
                    f"{patch.data.shape[2]} not as expected: "
                    f"{(m31model['pixels'].data.shape[2] // nraster)}"
                )
            m31reconstructed = create_image(
                npixel=512,
                cellsize=0.00015,
                phasecentre=phase_centre,
            )
            m31reconstructed = image_gather_facets(
                image_list,
                m31reconstructed,
                facets=nraster,
                overlap=overlap,
                taper=taper,
            )
            flat = image_gather_facets(
                image_list,
                m31reconstructed,
                facets=nraster,
                overlap=overlap,
                taper=taper,
                return_flat=True,
            )

            assert numpy.max(
                numpy.abs(flat["pixels"].data)
            ), f"Flat is empty for {nraster}"
            assert numpy.max(
                numpy.abs(m31reconstructed["pixels"].data)
            ), f"Raster is empty for {nraster}"


def test_scatter_gather_channel(phase_centre):
    """Unit test for image_scatter_channels &
    image_gather_channels functions"""
    for _ in [128, 16]:
        m31cube = create_image(
            npixel=512,
            cellsize=0.00015,
            phasecentre=phase_centre,
        )
        m31cube["pixels"].data = numpy.ones(
            shape=m31cube["pixels"].data.shape, dtype=float
        )
        for subimages in [16, 8, 2, 1]:
            image_list = image_scatter_channels(m31cube, subimages=subimages)
            m31cuberec = image_gather_channels(image_list)
            diff = m31cube["pixels"].data - m31cuberec["pixels"].data
            assert (
                numpy.max(numpy.abs(diff)) == 0.0
            ), f"Scatter gather failed for {subimages}"


def test_gather_channel(phase_centre):
    """Unit test for the image_gather_channels functions"""
    for nchan in [128, 16]:
        m31cube = create_image(
            npixel=512,
            cellsize=0.00015,
            phasecentre=phase_centre,
        )
        m31cube["pixels"].data = numpy.ones(
            shape=m31cube["pixels"].data.shape, dtype=float
        )
        image_list = image_scatter_channels(m31cube, subimages=nchan)
        m31cuberec = image_gather_channels(image_list)
        assert m31cube["pixels"].shape == m31cuberec["pixels"].shape
        diff = m31cube["pixels"].data - m31cuberec["pixels"].data
        assert (
            numpy.max(numpy.abs(diff)) == 0.0
        ), f"Scatter gather failed for {nchan}"
