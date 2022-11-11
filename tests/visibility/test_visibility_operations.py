# pylint: disable=duplicate-code, invalid-name, unsupported-assignment-operation
""" Unit tests for visibility operations


"""
import astropy.units as u
import numpy
import pytest
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.configuration.config_create import (
    create_named_configuration,
)
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)
from ska_sdp_datamodels.sky_model.sky_model import SkyComponent
from ska_sdp_datamodels.visibility.vis_create import create_visibility

from ska_sdp_func_python.visibility.operations import (
    concatenate_visibility,
    divide_visibility,
    subtract_visibility,
)


@pytest.fixture(scope="module", name="result_operations")
def visibility_operations_fixture():
    """ Fixture for operations.py unit tests
    """
    lowcore = create_named_configuration("LOWBD2-CORE")
    times = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 30.0)
    frequency = numpy.linspace(1.0e8, 1.1e8, 3)
    channel_bandwidth = numpy.array([1e7, 1e7, 1e7])
    f = numpy.array([100.0, 20.0, -10.0, 1.0])
    flux = numpy.array([f, 0.8 * f, 0.6 * f])
    phasecentre = SkyCoord(
        ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
    )
    compabsdirection = SkyCoord(
        ra=+181.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
    )
    pcof = phasecentre.skyoffset_frame()
    compreldirection = compabsdirection.transform_to(pcof)
    comp = SkyComponent(
        direction=compreldirection,
        frequency=frequency,
        flux=flux,
    )
    parameters = {
        "lowcore": lowcore,
        "times": times,
        "frequency": frequency,
        "channel_bandwidth": channel_bandwidth,
        "phasecentre": phasecentre,
        "compabsdirection": compabsdirection,
        "comp": comp,
    }
    return parameters


def test_concatenate_visibility(result_operations):
    """ Unit test for the concatenate_visibility function
    """
    vis = create_visibility(
        result_operations["lowcore"],
        result_operations["times"],
        result_operations["frequency"],
        channel_bandwidth=result_operations["channel_bandwidth"],
        phasecentre=result_operations["phasecentre"],
        weight=1.0,
    )
    othertimes = (numpy.pi / 43200.0) * numpy.arange(300.0, 600.0, 30.0)
    othervis = create_visibility(
        result_operations["lowcore"],
        othertimes,
        result_operations["frequency"],
        channel_bandwidth=result_operations["channel_bandwidth"],
        phasecentre=result_operations["phasecentre"],
        weight=1.0,
    )
    other_shape = list(othervis.vis.shape)
    this_shape = list(vis.vis.shape)
    newvis = concatenate_visibility([vis, othervis], dim="time")
    combined_shape = list(newvis.vis.shape)
    assert combined_shape[0] == this_shape[0] + other_shape[0]

    # Check that the input order is not important
    reverse_vis = concatenate_visibility([othervis, vis])
    assert reverse_vis.time.all() == newvis.time.all()
    assert combined_shape[0] == this_shape[0] + other_shape[0]
    newvis = newvis.dropna(dim="time", how="all")
    print(newvis)


def test_divide_visibility(result_operations):
    """ Unit test for the divide_visibility function with StokesI polarisation
    """
    vis = create_visibility(
        result_operations["lowcore"],
        result_operations["times"],
        result_operations["frequency"],
        channel_bandwidth=result_operations["channel_bandwidth"],
        phasecentre=result_operations["phasecentre"],
        weight=1.0,
        polarisation_frame=PolarisationFrame("stokesI"),
    )
    vis["vis"][..., :] = [2.0 + 0.0j]
    othervis = create_visibility(
        result_operations["lowcore"],
        result_operations["times"],
        result_operations["frequency"],
        channel_bandwidth=result_operations["channel_bandwidth"],
        phasecentre=result_operations["phasecentre"],
        weight=1.0,
        polarisation_frame=PolarisationFrame("stokesI"),
    )
    othervis["vis"][..., :] = [1.0 + 0.0j]
    ratiovis = divide_visibility(vis, othervis)
    assert ratiovis.visibility_acc.nvis == vis.visibility_acc.nvis
    assert numpy.max(numpy.abs(ratiovis.vis)) == 2.0, numpy.max(
        numpy.abs(ratiovis.vis)
    )


def test_divide_visibility_pol(result_operations):
    """ Unit test for the divide_visibility function with linear polarisation
    """
    vis = create_visibility(
        result_operations["lowcore"],
        result_operations["times"],
        result_operations["frequency"],
        channel_bandwidth=result_operations["channel_bandwidth"],
        phasecentre=result_operations["phasecentre"],
        weight=1.0,
        polarisation_frame=PolarisationFrame("linear"),
    )
    vis["vis"][..., :] = [2.0 + 0.0j, 0.0j, 0.0j, 2.0 + 0.0j]
    othervis = create_visibility(
        result_operations["lowcore"],
        result_operations["times"],
        result_operations["frequency"],
        channel_bandwidth=result_operations["channel_bandwidth"],
        phasecentre=result_operations["phasecentre"],
        weight=1.0,
        polarisation_frame=PolarisationFrame("linear"),
    )
    othervis["vis"][..., :] = [1.0 + 0.0j, 0.0j, 0.0j, 1.0 + 0.0j]
    ratiovis = divide_visibility(vis, othervis)
    assert ratiovis.visibility_acc.nvis == vis.visibility_acc.nvis
    assert numpy.max(numpy.abs(ratiovis.vis)) == 2.0, numpy.max(
        numpy.abs(ratiovis.vis)
    )


def test_divide_visibility_singular(result_operations):
    """ Unit test for the divide_visibility function with linear polarisation
    """
    vis = create_visibility(
        result_operations["lowcore"],
        result_operations["times"],
        result_operations["frequency"],
        channel_bandwidth=result_operations["channel_bandwidth"],
        phasecentre=result_operations["phasecentre"],
        weight=1.0,
        polarisation_frame=PolarisationFrame("linear"),
    )
    vis["vis"][..., :] = [
        2.0 + 0.0j,
        2.0 + 0.0j,
        2.0 + 0.0j,
        2.0 + 0.0j,
    ]
    othervis = create_visibility(
        result_operations["lowcore"],
        result_operations["times"],
        result_operations["frequency"],
        channel_bandwidth=result_operations["channel_bandwidth"],
        phasecentre=result_operations["phasecentre"],
        weight=1.0,
        polarisation_frame=PolarisationFrame("linear"),
    )
    othervis["vis"][..., :] = [
        1.0 + 0.0j,
        1.0 + 0.0j,
        1.0 + 0.0j,
        1.0 + 0.0j,
    ]
    ratiovis = divide_visibility(vis, othervis)
    assert ratiovis.visibility_acc.nvis == vis.visibility_acc.nvis
    assert numpy.max(numpy.abs(ratiovis.vis)) == 2.0, numpy.max(
        numpy.abs(ratiovis.vis)
    )


def test_subtract(result_operations):
    """ Unit test for the subtract_visibility function
    """
    vis1 = create_visibility(
        result_operations["lowcore"],
        result_operations["times"],
        result_operations["frequency"],
        channel_bandwidth=result_operations["channel_bandwidth"],
        phasecentre=result_operations["phasecentre"],
        weight=1.0,
        polarisation_frame=PolarisationFrame("stokesIQUV"),
    )
    vis1["vis"].data[...] = 1.0
    vis2 = create_visibility(
        result_operations["lowcore"],
        result_operations["times"],
        result_operations["frequency"],
        channel_bandwidth=result_operations["channel_bandwidth"],
        phasecentre=result_operations["phasecentre"],
        weight=1.0,
        polarisation_frame=PolarisationFrame("stokesIQUV"),
    )
    vis2["vis"].data[...] = 1.0
    zerovis = subtract_visibility(vis1, vis2)
    qa = zerovis.visibility_acc.qa_visibility(context="test_qa")
    assert qa.data["maxabs"] == pytest.approx(0.0, 7)
