# pylint: disable=invalid-name, too-many-arguments
# pylint: disable=attribute-defined-outside-init, unused-variable
# pylint: disable=too-many-instance-attributes, invalid-envvar-default
# pylint: disable=line-too-long, consider-using-f-string
# pylint: disable=missing-class-docstring, missing-function-docstring
# pylint: disable=import-error, no-name-in-module, import-outside-toplevel
"""Unit tests for image iteration


"""
import pytest

pytestmark = pytest.skip(
    allow_module_level=True
)  # Issues with create_image inputs to fix in gridding.py

import logging
import os
import tempfile

import numpy
from astropy import units
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.image.image_create import create_image
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)

from src.ska_sdp_func_python.image.gather_scatter import (
    image_gather_channels,
    image_gather_facets,
    image_scatter_channels,
    image_scatter_facets,
)

log = logging.getLogger("func-python-logger")

log.setLevel(logging.WARNING)


@pytest.fixture(scope="module", name="result_gather_scatter")
def gather_scatter_fixture():

    persist = os.getenv("FUNC_PYTHON_PERSIST", False)
    phase_centre = SkyCoord(
        ra=+180.0 * units.deg,
        dec=-35.0 * units.deg,
        frame="icrs",
        equinox="J2000",
    )
    params = {
        "persist": persist,
        "phase_centre": phase_centre,
    }
    return params


def test_scatter_gather_facet(result_gather_scatter):

    m31original = create_image(
        npixel=512,
        cellsize=0.00015,
        phasecentre=result_gather_scatter["phase_centre"],
    )
    assert numpy.max(
        numpy.abs(m31original["pixels"].data)
    ), "Original is empty"

    for nraster in [1, 4, 8]:
        m31model = create_image(
            npixel=512,
            cellsize=0.00015,
            phasecentre=result_gather_scatter["phase_centre"],
        )
        image_list = image_scatter_facets(m31model, facets=nraster)
        for patch in image_list:
            assert patch["pixels"].data.shape[3] == (
                m31model["pixels"].data.shape[3] // nraster
            ), "Number of pixels in each patch: %d not as expected: %d" % (
                patch["pixels"].data.shape[3],
                (m31model["pixels"].data.shape[3] // nraster),
            )
            assert patch["pixels"].data.shape[2] == (
                m31model["pixels"].data.shape[2] // nraster
            ), "Number of pixels in each patch: %d not as expected: %d" % (
                patch["pixels"].data.shape[2],
                (m31model["pixels"].data.shape[2] // nraster),
            )
            patch["pixels"].data[...] = 1.0
        m31reconstructed = create_image(
            npixel=512,
            cellsize=0.00015,
            phasecentre=result_gather_scatter["phase_centre"],
        )
        m31reconstructed = image_gather_facets(
            image_list, m31reconstructed, facets=nraster
        )
        flat = image_gather_facets(
            image_list, m31reconstructed, facets=nraster, return_flat=True
        )

        assert numpy.max(numpy.abs(flat["pixels"].data)), (
            "Flat is empty for %d" % nraster
        )
        assert numpy.max(numpy.abs(m31reconstructed["pixels"].data)), (
            "Raster is empty for %d" % nraster
        )


def test_scatter_gather_facet_overlap(result_gather_scatter):

    m31original = create_image(
        npixel=512,
        cellsize=0.00015,
        phasecentre=result_gather_scatter["phase_centre"],
    )
    assert numpy.max(
        numpy.abs(m31original["pixels"].data)
    ), "Original is empty"

    for nraster, overlap in [(1, 0), (4, 8), (8, 16)]:
        m31model = create_image(
            npixel=512,
            cellsize=0.00015,
            phasecentre=result_gather_scatter["phase_centre"],
        )
        image_list = image_scatter_facets(
            m31model, facets=nraster, overlap=overlap
        )
        for patch in image_list:
            assert patch["pixels"].data.shape[3] == (
                m31model["pixels"].data.shape[3] // nraster
            ), "Number of pixels in each patch: %d not as expected: %d" % (
                patch["pixels"].data.shape[3],
                (m31model["pixels"].data.shape[3] // nraster),
            )
            assert patch["pixels"].data.shape[2] == (
                m31model["pixels"].data.shape[2] // nraster
            ), "Number of pixels in each patch: %d not as expected: %d" % (
                patch["pixels"].data.shape[2],
                (m31model["pixels"].data.shape[2] // nraster),
            )
            patch["pixels"].data[...] = 1.0
        m31reconstructed = create_image(
            npixel=512,
            cellsize=0.00015,
            phasecentre=result_gather_scatter["phase_centre"],
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

        assert numpy.max(numpy.abs(flat["pixels"].data)), (
            "Flat is empty for %d" % nraster
        )
        assert numpy.max(numpy.abs(m31reconstructed["pixels"].data)), (
            "Raster is empty for %d" % nraster
        )


def test_scatter_gather_facet_overlap_taper(result_gather_scatter):

    m31original = create_image(
        npixel=512,
        cellsize=0.00015,
        phasecentre=result_gather_scatter["phase_centre"],
    )
    assert numpy.max(
        numpy.abs(m31original["pixels"].data)
    ), "Original is empty"

    for taper in ["linear", "tukey", None]:
        for nraster, overlap in [
            (1, 0),
            (2, 1),
            (2, 8),
            (4, 4),
            (4, 8),
            (8, 8),
            (8, 16),
        ]:
            m31model = create_image(
                npixel=512,
                cellsize=0.00015,
                phasecentre=result_gather_scatter["phase_centre"],
            )
            image_list = image_scatter_facets(
                m31model, facets=nraster, overlap=overlap, taper=taper
            )
            for patch in image_list:
                assert patch["pixels"].data.shape[3] == (
                    m31model["pixels"].data.shape[3] // nraster
                ), (
                    "Number of pixels in each patch: %d not as expected: %d"
                    % (
                        patch.data.shape[3],
                        (m31model["pixels"].data.shape[3] // nraster),
                    )
                )
                assert patch["pixels"].data.shape[2] == (
                    m31model["pixels"].data.shape[2] // nraster
                ), (
                    "Number of pixels in each patch: %d not as expected: %d"
                    % (
                        patch.data.shape[2],
                        (m31model["pixels"].data.shape[2] // nraster),
                    )
                )
            m31reconstructed = create_image(
                npixel=512,
                cellsize=0.00015,
                phasecentre=result_gather_scatter["phase_centre"],
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
            if result_gather_scatter["persist"]:
                with tempfile.TemporaryDirectory() as tempdir:
                    m31reconstructed.image_acc.export_to_fits(
                        f"{tempdir}/test_image_gather_scatter_{nraster}nraster_{overlap}overlap_{taper}_reconstructed.fits"
                    )
                    flat.image_acc.export_to_fits(
                        f"{tempdir}/test_image_gather_scatter_{nraster}nraster_{overlap}overlap_{taper}_flat.fits"
                    )

            assert numpy.max(numpy.abs(flat["pixels"].data)), (
                "Flat is empty for %d" % nraster
            )
            assert numpy.max(numpy.abs(m31reconstructed["pixels"].data)), (
                "Raster is empty for %d" % nraster
            )


def test_scatter_gather_channel(result_gather_scatter):
    for nchan in [128, 16]:
        m31cube = create_image(
            npixel=512,
            cellsize=0.00015,
            phasecentre=result_gather_scatter["phase_centre"],
            polarisation_frame=PolarisationFrame("stokesI"),
            frequency=numpy.linspace(1e8, 1.1e8, nchan),
        )

        for subimages in [16, 8, 2, 1]:
            image_list = image_scatter_channels(m31cube, subimages=subimages)
            m31cuberec = image_gather_channels(
                image_list, m31cube, subimages=subimages
            )
            diff = m31cube["pixels"].data - m31cuberec["pixels"].data
            assert numpy.max(numpy.abs(diff)) == 0.0, (
                "Scatter gather failed for %d" % subimages
            )


def test_gather_channel(result_gather_scatter):
    for nchan in [128, 16]:
        m31cube = create_image(
            npixel=512,
            cellsize=0.00015,
            phasecentre=result_gather_scatter["phase_centre"],
            polarisation_frame=PolarisationFrame("stokesI"),
            frequency=numpy.linspace(1e8, 1.1e8, nchan),
        )
        image_list = image_scatter_channels(m31cube, subimages=nchan)
        m31cuberec = image_gather_channels(image_list, None, subimages=nchan)
        assert m31cube["pixels"].shape == m31cuberec["pixels"].shape
        diff = m31cube["pixels"].data - m31cuberec["pixels"].data
        assert numpy.max(numpy.abs(diff)) == 0.0, (
            "Scatter gather failed for %d" % nchan
        )
