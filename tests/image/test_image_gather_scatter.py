"""
Unit tests for image iteration
"""
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
    """
    Unit test for the image_gather_facets function
    Original test uses an image that contains
    H-alpha region information in M31.
    Here we just create an empty image instead,
    and uses a single channel with frequency=1e8 Hz,
    and single StokesI polarisation.
    """
    original = create_image(
        10,
        0.00015,
        phase_centre,
        nchan=1,
    )
    original["pixels"].data = numpy.ones(
        shape=original["pixels"].data.shape, dtype=float
    )
    assert numpy.max(numpy.abs(original["pixels"].data)), "Original is empty"

    nraster = 1
    model = create_image(
        512,
        0.00015,
        phase_centre,
        nchan=1,
    )
    model["pixels"].data = numpy.ones(
        shape=model["pixels"].data.shape, dtype=float
    )
    image_list = image_scatter_facets(model, facets=nraster)
    for patch in image_list:
        assert patch["pixels"].data.shape[3] == (
            model["pixels"].data.shape[3] // nraster
        ), (
            f"Number of pixels in each patch: "
            f"{patch['pixels'].data.shape[3]} not as expected: "
            f"{(model['pixels'].data.shape[3] // nraster)}"
        )

        assert patch["pixels"].data.shape[2] == (
            model["pixels"].data.shape[2] // nraster
        ), (
            f"Number of pixels in each patch: "
            f"{patch['pixels'].data.shape[2]} not as expected: "
            f"{(model['pixels'].data.shape[2] // nraster)}"
        )

        # Check for frequency and polarisation
        assert patch["pixels"].data.shape[0] == model["pixels"].data.shape[0]
        assert patch["pixels"].data.shape[1] == model["pixels"].data.shape[1]

        patch["pixels"].data[...] = 1.0

    reconstructed = create_image(
        512,
        0.00015,
        phase_centre,
        nchan=1,
    )
    reconstructed["pixels"].data = numpy.ones(
        shape=reconstructed["pixels"].data.shape, dtype=float
    )
    reconstructed = image_gather_facets(
        image_list, reconstructed, facets=nraster
    )
    flat = image_gather_facets(
        image_list, reconstructed, facets=nraster, return_flat=True
    )

    assert numpy.max(
        numpy.abs(flat["pixels"].data)
    ), f"Flat is empty for {nraster}"
    assert numpy.max(
        numpy.abs(reconstructed["pixels"].data)
    ), f"Raster is empty for {nraster}"
    assert (
        flat["pixels"].data.shape[0] == reconstructed["pixels"].data.shape[0]
    )


def test_scatter_gather_facet_overlap(phase_centre):
    """
    Unit test for the image_gather_facets function with overlap
    Image information same as previous test
    """
    original = create_image(
        512,
        0.00015,
        phase_centre,
        nchan=1,
    )
    original["pixels"].data = numpy.ones(
        shape=original["pixels"].data.shape, dtype=float
    )
    assert numpy.max(numpy.abs(original["pixels"].data)), "Original is empty"

    for nraster, overlap in [(1, 0), (1, 8), (1, 16)]:
        model = create_image(
            512,
            0.00015,
            phase_centre,
            nchan=1,
        )
        model["pixels"].data = numpy.ones(
            shape=model["pixels"].data.shape, dtype=float
        )
        image_list = image_scatter_facets(
            model, facets=nraster, overlap=overlap
        )
        for patch in image_list:
            assert patch["pixels"].data.shape[3] == (
                model["pixels"].data.shape[3] // nraster
            ), (
                f"Number of pixels in each patch: "
                f"{patch['pixels'].data.shape[3]} not as expected: "
                f"{(model['pixels'].data.shape[3] // nraster)}"
            )

            assert patch["pixels"].data.shape[2] == (
                model["pixels"].data.shape[2] // nraster
            ), (
                f"Number of pixels in each patch: "
                f"{patch['pixels'].data.shape[2]} not as expected: "
                f"{(model['pixels'].data.shape[2] // nraster)}"
            )
            # Check for frequency and polarisation
            assert (
                patch["pixels"].data.shape[0] == model["pixels"].data.shape[0]
            )
            assert (
                patch["pixels"].data.shape[1] == model["pixels"].data.shape[1]
            )

            patch["pixels"].data[...] = 1.0

        reconstructed = create_image(
            512,
            0.00015,
            phase_centre,
            nchan=1,
        )
        reconstructed["pixels"].data = numpy.ones(
            shape=reconstructed["pixels"].data.shape, dtype=float
        )
        reconstructed = image_gather_facets(
            image_list, reconstructed, facets=nraster, overlap=overlap
        )
        flat = image_gather_facets(
            image_list,
            reconstructed,
            facets=nraster,
            overlap=overlap,
            return_flat=True,
        )

        assert (
            flat["pixels"].data.shape[0]
            == reconstructed["pixels"].data.shape[0]
        )
        assert numpy.max(
            numpy.abs(flat["pixels"].data)
        ), f"Flat is empty for {nraster}"
        assert numpy.max(
            numpy.abs(reconstructed["pixels"].data)
        ), f"Raster is empty for {nraster}"


def test_scatter_gather_facet_overlap_taper(phase_centre):
    """Unit test for the image_gather_facets function with overlap and taper"""
    original = create_image(
        512,
        0.00015,
        phase_centre,
        nchan=1,
    )
    original["pixels"].data = numpy.ones(
        shape=original["pixels"].data.shape, dtype=float
    )
    assert numpy.max(numpy.abs(original["pixels"].data)), "Original is empty"

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
            model = create_image(
                512,
                0.00015,
                phase_centre,
                nchan=1,
            )
            model["pixels"].data = numpy.ones(
                shape=model["pixels"].data.shape, dtype=float
            )
            image_list = image_scatter_facets(
                model, facets=nraster, overlap=overlap, taper=taper
            )
            for patch in image_list:
                assert patch["pixels"].data.shape[3] == (
                    model["pixels"].data.shape[3] // nraster
                ), (
                    f"Number of pixels in each patch: "
                    f"{patch.data.shape[3]} not as expected: "
                    f"{(model['pixels'].data.shape[3] // nraster)}"
                )
                assert patch["pixels"].data.shape[2] == (
                    model["pixels"].data.shape[2] // nraster
                ), (
                    f"Number of pixels in each patch: "
                    f"{patch.data.shape[2]} not as expected: "
                    f"{(model['pixels'].data.shape[2] // nraster)}"
                )
                # Check for frequency and polarisation
                assert (
                    patch["pixels"].data.shape[0]
                    == model["pixels"].data.shape[0]
                )
                assert (
                    patch["pixels"].data.shape[1]
                    == model["pixels"].data.shape[1]
                )

            reconstructed = create_image(
                512,
                0.00015,
                phase_centre,
                nchan=1,
            )
            reconstructed = image_gather_facets(
                image_list,
                reconstructed,
                facets=nraster,
                overlap=overlap,
                taper=taper,
            )
            flat = image_gather_facets(
                image_list,
                reconstructed,
                facets=nraster,
                overlap=overlap,
                taper=taper,
                return_flat=True,
            )

            assert numpy.max(
                numpy.abs(flat["pixels"].data)
            ), f"Flat is empty for {nraster}"
            assert numpy.max(
                numpy.abs(reconstructed["pixels"].data)
            ), f"Raster is empty for {nraster}"
            assert (
                flat["pixels"].data.shape[0]
                == reconstructed["pixels"].data.shape[0]
            )


def test_scatter_gather_channel(phase_centre):
    """Unit test for image_scatter_channels &
    image_gather_channels functions"""
    for _ in [128, 16]:
        cube = create_image(
            512,
            0.00015,
            phase_centre,
            nchan=1,
        )
        cube["pixels"].data = numpy.ones(
            shape=cube["pixels"].data.shape, dtype=float
        )
        for subimages in [16, 8, 2, 1]:
            image_list = image_scatter_channels(cube, subimages=subimages)
            cuberec = image_gather_channels(image_list)
            diff = cube["pixels"].data - cuberec["pixels"].data
            assert (
                numpy.max(numpy.abs(diff)) == 0.0
            ), f"Scatter gather failed for {subimages}"


def test_gather_channel(phase_centre):
    """Unit test for the image_gather_channels functions"""
    for nchan in [128, 16]:
        cube = create_image(
            512,
            0.00015,
            phase_centre,
            nchan=1,
        )
        cube["pixels"].data = numpy.ones(
            shape=cube["pixels"].data.shape, dtype=float
        )
        image_list = image_scatter_channels(cube, subimages=nchan)
        cuberec = image_gather_channels(image_list)
        assert cube["pixels"].shape == cuberec["pixels"].shape
        diff = cube["pixels"].data - cuberec["pixels"].data
        assert (
            numpy.max(numpy.abs(diff)) == 0.0
        ), f"Scatter gather failed for {nchan}"
