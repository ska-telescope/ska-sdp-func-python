"""
Unit tests for image operations
"""
import logging
import sys

import numpy
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.configuration.config_create import (
    create_named_configuration,
)
from ska_sdp_datamodels.gridded_visibility.grid_vis_create import (
    create_convolutionfunction_from_image,
    create_griddata_from_image,
)
from ska_sdp_datamodels.image.image_create import create_image
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)
from ska_sdp_datamodels.sky_model.sky_model import SkyComponent
from ska_sdp_datamodels.visibility.vis_create import create_visibility

from ska_sdp_func_python.grid_data.gridding import (
    degrid_visibility_from_griddata,
    fft_griddata_to_image,
    fft_image_to_griddata,
    grid_visibility_to_griddata,
    grid_visibility_weight_to_griddata,
    griddata_merge_weights,
    griddata_visibility_reweight,
)
from ska_sdp_func_python.sky_component.operations import insert_skycomponent

log = logging.getLogger("func-python-logger")

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))


@pytest.fixture(scope="module", name="input_params")
def input_for_gridding_fixture():
    """Pytest fixture for the gridding.py unit tests"""

    npixel = 256
    cellsize = 0.0009
    low = create_named_configuration("LOWBD2", rmax=750.0)
    ntimes = 3
    times = numpy.linspace(-2.0, +2.0, ntimes) * numpy.pi / 12.0
    frequency = numpy.array([1e8])
    channelwidth = numpy.array([4e7])
    vis_pol = PolarisationFrame("stokesI")
    flux = numpy.ones((1, 1))
    phase_centre = SkyCoord(
        ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
    )
    vis = create_visibility(
        low,
        times,
        frequency,
        phase_centre,
        channelwidth,
        polarisation_frame=vis_pol,
    )
    image = create_image(
        npixel,
        cellsize,
        phase_centre,
        frequency=frequency[0],
        channel_bandwidth=channelwidth[0],
        nchan=len(frequency),
    )
    components = SkyComponent(
        phase_centre,
        frequency,
        name="griddata_sc",
        flux=flux,
        polarisation_frame=vis_pol,
    )
    model = insert_skycomponent(image, components)

    grid_data = create_griddata_from_image(image, vis_pol)
    convolution_function = create_convolutionfunction_from_image(model)
    params = {
        "convolution_function": convolution_function,
        "grid_data": grid_data,
        "image": model,
        "visibility": vis,
    }

    return params


@pytest.mark.skip(reason="Convolution Function issues")
def test_grid_visibility_to_griddata(input_params):
    """Unit tests for grid_visibility_to_griddata function:
    check that grid_data is updated
    """
    conv_func = input_params["convolution_function"]
    grid_data = input_params["grid_data"]
    vis = input_params["visibility"]
    result = grid_visibility_to_griddata(vis, grid_data, conv_func)

    assert result != grid_data


def test_grid_visibility_weight_to_griddata(input_params):
    """Unit tests for grid_visibility_to_griddata function:
    check that grid_data is updated
    """
    grid_data = input_params["grid_data"]
    vis = input_params["visibility"]
    result_gd, _ = grid_visibility_weight_to_griddata(vis, grid_data)

    assert result_gd != grid_data


def test_griddata_merge_weights(input_params):
    """Unit tests for griddata_merge_weights function:
    check that grid_data is updated and sumwt is 3
    """
    grid_data = input_params["grid_data"]
    gd_list = [(grid_data, 1), (grid_data, 1), (grid_data, 1)]
    result_gd, result_sumwt = griddata_merge_weights(gd_list)

    assert result_gd != grid_data
    assert result_sumwt == 3


def test_griddata_visibility_reweight(input_params):
    """Unit tests for griddata_visibility_reweight function:
    check that vis is updated
    """
    grid_data = input_params["grid_data"]
    vis = input_params["visibility"]
    result = griddata_visibility_reweight(vis, grid_data)

    assert result != vis


@pytest.mark.skip(reason="Convolution Function issue")
def test_degrid_visibility_from_griddata(input_params):
    """Unit tests for degrid_visibility_from_griddata function:
    check that vis is updated
    """
    conv_func = input_params["convolution_function"]
    grid_data = input_params["grid_data"]
    vis = input_params["visibility"]
    result = degrid_visibility_from_griddata(vis, grid_data, conv_func)

    assert result == vis


def test_fft_griddata_to_image(input_params):
    """Unit tests for fft_griddata_to_image function:
    check that vis is updated
    """
    grid_data = input_params["grid_data"]
    image = input_params["image"]
    result = fft_griddata_to_image(grid_data, image)

    assert result != image


def test_fft_image_to_griddata(input_params):
    """Unit tests for fft_griddata_to_image function:
    check that vis is updated
    """
    grid_data = input_params["grid_data"]
    image = input_params["image"]
    result = fft_image_to_griddata(image, grid_data)

    assert result != grid_data
