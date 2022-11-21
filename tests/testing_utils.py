"""
Util functions for testing.
"""

import logging

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.calibration import GainTable
from ska_sdp_datamodels.configuration import create_named_configuration
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.sky_model import SkyComponent
from ska_sdp_datamodels.visibility import create_visibility

from ska_sdp_func_python.imaging import dft_skycomponent_visibility

log = logging.getLogger("func-python-logger")


def simulate_gaintable(
    gain_table: GainTable,
    phase_error=0.1,
    amplitude_error=0.0,
    leakage=0.0,
) -> GainTable:
    """
    Simulate a gain table

    :type gain_table: GainTable
    :param phase_error: std of normal distribution, zero mean
    :param amplitude_error: std of log normal distribution
    :param leakage: std of cross hand leakage
    :return: updated GainTable
    """
    # pylint: disable=import-outside-toplevel
    from numpy.random import default_rng

    rng = default_rng(1805550721)

    log.debug(
        "simulate_gaintable: Simulating amplitude "
        "error = %.4f, phase error = %.4f",
        amplitude_error,
        phase_error,
    )
    amps = 1.0
    phases = 1.0
    nrec = gain_table["gain"].data.shape[3]

    if phase_error > 0.0:
        phases = rng.normal(0, phase_error, gain_table["gain"].data.shape)

    if amplitude_error > 0.0:
        amps = rng.lognormal(
            0.0, amplitude_error, gain_table["gain"].data.shape
        )

    gain_table["gain"].data = amps * numpy.exp(0 + 1j * phases)
    nrec = gain_table["gain"].data.shape[-1]
    if nrec > 1:
        if leakage > 0.0:
            leak = rng.normal(
                0, leakage, gain_table["gain"].data[..., 0, 0].shape
            ) + 1j * rng.normal(
                0, leakage, gain_table["gain"].data[..., 0, 0].shape
            )
            gain_table["gain"].data[..., 0, 1] = (
                gain_table["gain"].data[..., 0, 0] * leak
            )
            leak = rng.normal(
                0, leakage, gain_table["gain"].data[..., 1, 1].shape
            ) + 1j * rng.normal(
                0, leakage, gain_table["gain"].data[..., 1, 1].shape
            )
            gain_table["gain"].data[..., 1, 0] = (
                gain_table["gain"].data[..., 1, 1].data * leak
            )
        else:
            gain_table["gain"].data[..., 0, 1] = 0.0
            gain_table["gain"].data[..., 1, 0] = 0.0

    return gain_table


def vis_with_component_data(
    sky_pol_frame, data_pol_frame, flux_array, **kwargs
):
    """
    Generate Visibility data for testing.

    :param sky_pol_frame: PolarisationFrame of SkyComponents
    :param data_pol_frame: PolarisationFrame of Visibility data
    :param flux_array: Flux data for SkyComponents
    :param kwargs: includes:
            ntimes: number of time samples
            rmax: maximum distance of antenna from centre
                  when configuration is determined
            nchan: number of frequency channels
    """
    ntimes = kwargs.get("ntimes", 3)
    rmax = kwargs.get("rmax", 300)
    lowcore = create_named_configuration("LOWBD2", rmax=rmax)
    times = (numpy.pi / 43200.0) * numpy.linspace(0.0, 30.0, 1 + ntimes)

    nchan = kwargs.get("nchan", 1)
    if nchan > 1:
        frequency = numpy.linspace(1.0e8, 1.1e8, nchan)
        channel_bandwidth = numpy.array(nchan * [frequency[1] - frequency[0]])
    else:
        frequency = 1e8 * numpy.ones([1])
        channel_bandwidth = 1e7 * numpy.ones([1])

    # The phase centre is absolute and the component is specified relative
    # This means that the component should end up at the position
    # phasecentre+compredirection
    phasecentre = SkyCoord(
        ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
    )
    compabsdirection = SkyCoord(
        ra=+181.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
    )

    if sky_pol_frame == "stokesI":
        flux_array = [100.0]
    flux = numpy.outer(
        numpy.array([numpy.power(freq / 1e8, -0.7) for freq in frequency]),
        flux_array,
    )

    comp = SkyComponent(
        direction=compabsdirection,
        frequency=frequency,
        flux=flux,
        polarisation_frame=PolarisationFrame(sky_pol_frame),
    )
    vis = create_visibility(
        lowcore,
        times,
        frequency,
        phasecentre=phasecentre,
        channel_bandwidth=channel_bandwidth,
        weight=1.0,
        polarisation_frame=PolarisationFrame(data_pol_frame),
    )
    vis = dft_skycomponent_visibility(vis, comp)
    return vis
