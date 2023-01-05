import pytest
from astropy import units
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.image import create_image
from ska_sdp_datamodels.science_data_model import PolarisationFrame


@pytest.fixture(scope="package")
def image_phase_centre():
    return SkyCoord(
        ra=-180.0 * units.deg,
        dec=+35.0 * units.deg,
        frame="icrs",
        equinox="J2000",
    )


@pytest.fixture(scope="package")
def image(visibility, image_phase_centre):
    npixels = 512
    cellsize = 0.00015
    im = create_image(
        npixels,
        cellsize,
        image_phase_centre,
        polarisation_frame=PolarisationFrame(visibility._polarisation_frame),
        frequency=visibility.frequency.data[0],
        channel_bandwidth=visibility.channel_bandwidth.data[0],
        nchan=visibility.visibility_acc.nchan,
    )
    return im
