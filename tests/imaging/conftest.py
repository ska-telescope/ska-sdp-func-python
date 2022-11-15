import numpy
import pytest
from astropy import units
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.configuration import create_named_configuration
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.sky_model import SkyComponent
from ska_sdp_datamodels.visibility import create_visibility

N_CHAN = 6


@pytest.fixture(scope="module", name="directions")
def directions_fixture():
    """
    Phase center and Component Absolute direction fixture
    """
    # The phase centre is absolute and the component
    # is specified relative (for now).
    # This means that the component should end up at
    # the position phase_centre+comp_redirection
    phase_centre = SkyCoord(
        ra=+180.0 * units.deg,
        dec=-35.0 * units.deg,
        frame="icrs",
        equinox="J2000",
    )
    comp_abs_direction = SkyCoord(
        ra=+181.0 * units.deg,
        dec=-35.0 * units.deg,
        frame="icrs",
        equinox="J2000",
    )
    return phase_centre, comp_abs_direction


@pytest.fixture(scope="module", name="comp")
def component_fixture(directions):
    """
    Sky component list fixture
    """
    n_comp = 20
    phase_centre = directions[0]
    comp_abs_direction = directions[1]
    phase_centre_offset = phase_centre.skyoffset_frame()
    comp_rel_direction = comp_abs_direction.transform_to(phase_centre_offset)

    flux = numpy.array(N_CHAN * [100.0, 20.0, -10.0, 1.0]).reshape([N_CHAN, 4])
    frequency = numpy.linspace(1.0e8, 1.1e8, N_CHAN)

    comp = n_comp * [
        SkyComponent(
            direction=comp_rel_direction,
            frequency=frequency,
            flux=flux,
        )
    ]
    return comp


@pytest.fixture(scope="module", name="vis")
def vis_fixture(directions):
    """
    Visibility fixture
    """
    n_times = 2
    low_core = create_named_configuration("LOW")
    times = (numpy.pi / 43200.0) * numpy.linspace(0.0, 300.0, n_times)

    frequency = numpy.linspace(1.0e8, 1.1e8, N_CHAN)
    channel_bandwidth = numpy.array(N_CHAN * [1e7 / N_CHAN])

    vis = create_visibility(
        low_core,
        times,
        frequency,
        channel_bandwidth=channel_bandwidth,
        phasecentre=directions[0],
        weight=1.0,
        polarisation_frame=PolarisationFrame("linear"),
    )
    return vis
