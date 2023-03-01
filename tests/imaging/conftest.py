"""
Fixtures for imaging tests
"""

import pytest
from ska_sdp_datamodels.science_data_model import PolarisationFrame

from ska_sdp_func_python.imaging import create_image_from_visibility


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
