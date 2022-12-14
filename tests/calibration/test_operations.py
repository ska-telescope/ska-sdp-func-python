"""
Unit tests for calibration operations
"""
import numpy
import pytest
from astropy import units
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.calibration.calibration_create import (
    create_gaintable_from_visibility,
)
from ska_sdp_datamodels.configuration.config_create import (
    create_named_configuration,
)
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)
from ska_sdp_datamodels.visibility.vis_create import create_visibility

from ska_sdp_func_python.calibration.operations import (
    apply_gaintable,
    concatenate_gaintables,
    multiply_gaintables,
)


@pytest.fixture(scope="module", name="input_params")
def input_vis_gt():
    """Fixture for the operations unit tests"""
    # Create a visibility object
    lowcore = create_named_configuration("LOWBD2-CORE")
    times = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 30.0)
    frequency = numpy.linspace(0.8e8, 1.2e8, 5)
    channel_bandwidth = numpy.array([1e7, 1e7, 1e7, 1e7, 1e7])
    polarisation_frame = PolarisationFrame("linear")
    phase_centre = SkyCoord(
        ra=+180.0 * units.deg,
        dec=-35.0 * units.deg,
        frame="icrs",
        equinox="J2000",
    )

    vis1 = create_visibility(
        lowcore,
        times,
        frequency,
        channel_bandwidth=channel_bandwidth,
        polarisation_frame=polarisation_frame,
        phasecentre=phase_centre,
        weight=1.0,
    )
    vis2 = create_visibility(
        lowcore,
        times,
        frequency=numpy.linspace(1.6e9, 2.4e8, 5),
        channel_bandwidth=numpy.array([2e7, 2e7, 2e7, 2e7, 2e7]),
        polarisation_frame=polarisation_frame,
        phasecentre=phase_centre,
        weight=2.0,
    )

    # Build GainTable object
    gain_table1 = create_gaintable_from_visibility(vis1)
    gain_table2 = create_gaintable_from_visibility(vis2)

    parameters = {
        "visibility1": vis1,
        "visibility2": vis2,
        "gaintable1": gain_table1,
        "gaintable2": gain_table2,
    }

    return parameters


def test_apply_gaintable(input_params):
    """
    Unit test for the apply_gaintable function, using flags
    TODO: The property of use_flags is not properly tested,
          need to revisit this test
    """

    vis = input_params["visibility2"]
    result = apply_gaintable(
        input_params["visibility1"],
        input_params["gaintable2"],
        use_flags=True,
    )
    assert (result["vis"].data == vis["vis"].data).all()
    assert (result["weight"].data == vis["weight"].data / 2).all()


def test_multiply_gaintables(input_params):
    """
    Unit test for the multiply_gaintable function
    """
    gt = create_gaintable_from_visibility(  # pylint: disable=invalid-name
        input_params["visibility1"]
    )
    dgt = create_gaintable_from_visibility(input_params["visibility2"])

    result = multiply_gaintables(gt, dgt)

    assert (
        result["gain"].data
        == numpy.einsum(
            "...ik,...ij->...kj", gt["gain"].data, dgt["gain"].data
        )
    ).all()
    assert (
        result["weight"].data == (gt["weight"].data * dgt["weight"].data)
    ).all()


def test_concatenate_gaintables(input_params):
    """
    Unit test for the multiply_gaintable function
    """

    gt_list = [
        input_params["gaintable1"],
        input_params["gaintable2"],
    ]

    result = concatenate_gaintables(gt_list)

    assert (len(gt_list[0]) + len(gt_list[1])) == (
        len(input_params["gaintable1"]) + len(input_params["gaintable2"])
    )
    assert len(result) == len(input_params["gaintable1"])
    assert len(result) == len(input_params["gaintable2"])
