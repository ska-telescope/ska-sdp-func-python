# pylint: disable=invalid-name
"""
Unit tests for ionospheric calibration and phase screen utils
"""
import numpy
import pytest
from ska_sdp_datamodels.configuration.config_create import (
    create_named_configuration,
)

from ska_sdp_func_python.calibration.ionosphere_utils import (
    decompose_phasescreen,
    zern,
    zern_array,
)


@pytest.fixture(scope="module", name="input_params")
def input_for_ionosphere_util():
    """Fixture for the ionosphere_util unit tests"""
    # Generate an array comprising the core stations and a handful of
    # six-station clusters
    low_config = create_named_configuration("LOWBD2", rmax=1200.0)

    parameters = {
        "configuration": low_config,
    }

    return parameters


def test_decompose_phasescreen(input_params):
    """Unit tests for decompose_phasescreen function:
    Generate 1000 random screens, fit the structure function parameters
    to the phase variance across the baselines, and make sure they are
    within a few percent of the input structure function parameters
    """
    # pylint: disable=too-many-locals
    low_config = input_params["configuration"]

    x = low_config.xyz.data[:, 0]
    y = low_config.xyz.data[:, 1]

    # Generate a mask to select cross products (baselines) from an
    # n_stations x n_stations covariance matrix
    n_stations = low_config.stations.shape[0]
    idx = numpy.arange(n_stations)
    mask = numpy.tile(idx[:, numpy.newaxis], (1, n_stations)) > numpy.tile(
        idx[numpy.newaxis, :], (n_stations, 1)
    )

    # Calculate the separation of station pairs. Pierce points towards a point
    # on the celestial sphere through a 2D horizontal ionosphere will have the
    # same relative positions
    u = (
        numpy.tile(x[:, numpy.newaxis], (1, n_stations))
        - numpy.tile(x[numpy.newaxis, :], (n_stations, 1))
    )[mask]
    v = (
        numpy.tile(y[:, numpy.newaxis], (1, n_stations))
        - numpy.tile(y[numpy.newaxis, :], (n_stations, 1))
    )[mask]
    r = numpy.sqrt(u * u + v * v)

    # Do the decomposition for the pierce points
    r_0 = 7e3
    beta = 5.0 / 3.0
    [evec_matrix, sqrt_evals] = decompose_phasescreen(x, y, r_0, beta)

    # Generate the random screens and accumulate the covariance estimates
    shift_variance = numpy.zeros(len(r))
    numpy.random.seed(int(1e8))
    for _ in range(1000):
        phase_shifts = evec_matrix @ (
            sqrt_evals * numpy.random.randn(n_stations)
        )
        relative_shifts = (
            numpy.tile(phase_shifts[:, numpy.newaxis], (1, n_stations))
            - numpy.tile(phase_shifts[numpy.newaxis, :], (n_stations, 1))
        )[mask]
        shift_variance += (relative_shifts**2) / 1000.0

    # Estimate structure function parameters from the covariance measurements
    # var(r) = (r/r_0)**beta
    # => log(var(r)) = beta*log(r) - beta*log(r_0)
    S = len(r)
    Sx = numpy.sum(numpy.log(r))
    Sxx = numpy.sum(numpy.log(r) ** 2)
    Sxy = numpy.sum(numpy.log(r) * numpy.log(shift_variance))
    Sy = numpy.sum(numpy.log(shift_variance))
    delta = S * Sxx - Sx * Sx
    beta_fit = (S * Sxy - Sx * Sy) / delta
    r_0_fit = numpy.exp(-(Sxx * Sy - Sx * Sxy) / delta / beta_fit)

    assert numpy.abs(beta_fit - beta) / beta < 2e-3
    assert numpy.abs(r_0_fit - r_0) / r_0 < 2e-2


def test_zern():
    """Unit tests for zern function:
    Very basic checks
    """

    # pick n points uniformly distributed across a unit circle
    numpy.random.seed(int(1e8))
    n = 1000
    rho = numpy.sqrt(numpy.random.rand(n))
    phi = 2.0 * numpy.pi * numpy.random.rand(n)

    count = 0
    for n in range(5):
        for m in range(-n, n + 1, 2):
            coeff = zern(m, n, rho, phi)
            assert numpy.amin(coeff) >= -1
            assert numpy.amax(coeff) <= 1
            count += 1


def test_zern_array():
    """Unit tests for zern_array function:
    Very basic checks
    """

    # pick n points uniformly distributed across a square
    numpy.random.seed(int(1e8))
    n = 1000
    x = 30 * numpy.random.rand(n)
    y = 30 * numpy.random.rand(n)

    assert len(zern_array(0, x, y, noll_order=True)) == len(x)
    assert len(zern_array(0, x, y, noll_order=True)[0]) == 1
    assert len(zern_array(1, x, y, noll_order=True)[0]) == 3
    assert len(zern_array(2, x, y, noll_order=True)[0]) == 6
    assert len(zern_array(3, x, y, noll_order=True)[0]) == 10
    assert len(zern_array(4, x, y, noll_order=True)[0]) == 15

    assert len(zern_array(0, x, y, noll_order=False)) == len(x)
    assert len(zern_array(0, x, y, noll_order=False)[0]) == 1
    assert len(zern_array(2, x, y, noll_order=False)[0]) == 4
    assert len(zern_array(4, x, y, noll_order=False)[0]) == 9
    assert len(zern_array(6, x, y, noll_order=False)[0]) == 16
