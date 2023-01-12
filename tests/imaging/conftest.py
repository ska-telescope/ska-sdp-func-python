"""
Fixtures for imaging tests
"""

import pytest
from astropy import units
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.image import create_image
from ska_sdp_datamodels.science_data_model import PolarisationFrame

from ska_sdp_func_python.imaging import create_image_from_visibility


@pytest.fixture(scope="package", name="image_phase_centre")
def image_phase_centre_fixt():
    """
    Phase centre for image object
    """
    return SkyCoord(
        ra=-180.0 * units.deg,
        dec=+35.0 * units.deg,
        frame="icrs",
        equinox="J2000",
    )


@pytest.fixture(scope="package", name="image")
def image_fixt(visibility, image_phase_centre):
    """
    Image fixture
    """
    npixels = 512
    cellsize = 0.00015
    im = create_image(
        npixels,
        cellsize,
        image_phase_centre,
        # pylint: disable=protected-access
        polarisation_frame=PolarisationFrame(visibility._polarisation_frame),
        frequency=visibility.frequency.data[0],
        channel_bandwidth=visibility.channel_bandwidth.data[0],
        nchan=visibility.visibility_acc.nchan,
    )
    return im


@pytest.fixture(scope="package", name="model")
def model_fixt(visibility):
    """
    Model image fixture
    """
    model = create_image_from_visibility(
        visibility,
        npixel=512,
        cellsize=0.0005,
        nchan=visibility.visibility_acc.nchan,
        polarisation_frame=PolarisationFrame("stokesIQUV"),
    )
    return model
