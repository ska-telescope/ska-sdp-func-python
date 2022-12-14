"""
Unit tests for calibration solution
"""
import pytest
from ska_sdp_datamodels.calibration.calibration_create import (
    create_gaintable_from_visibility,
)

from ska_sdp_func_python.calibration.operations import apply_gaintable
from ska_sdp_func_python.calibration.solvers import solve_gaintable
from tests.testing_utils import simulate_gaintable, vis_with_component_data


@pytest.mark.parametrize(
    "sky_pol_frame, data_pol_frame, flux_array, "
    "phase_error, expected_gain_sum",
    [
        (
            "stokesIQUV",
            "circular",
            [100.0, 0.0, 0.0, 50.0],
            10.0,
            (61.44194904161769, -0.028730599005608592),
        ),
        (
            "stokesIQUV",
            "linear",
            [100.0, 50.0, 0.0, 0.0],
            10.0,
            (61.44194904161769, -0.028730599005608592),
        ),
        (
            "stokesIV",
            "circularnp",
            [100.0, 50.0],
            0.1,
            (748.2141413044451, -2.679009413197875e-07),
        ),
        (
            "stokesIQ",
            "linearnp",
            [100.0, 50.0],
            0.1,
            (748.2141413044451, -2.679009413197875e-07),
        ),
        (
            "stokesIQUV",
            "circular",
            [100.0, 0.0, 0.0, 50.0],
            0.1,
            (748.2141413044451, -2.679009413197875e-07),
        ),
        (
            "stokesIQUV",
            "linear",
            [100.0, 50.0, 0.0, 0.0],
            0.1,
            (748.2141413044451, -2.679009413197875e-07),
        ),
        (
            "stokesI",
            "stokesI",
            [100.0, 0.0, 0.0, 0.0],
            0.1,
            (372.357810300829, 23.57909603997496),
        ),
    ],
)
def test_solve_gaintable_phase_only(
    sky_pol_frame, data_pol_frame, flux_array, phase_error, expected_gain_sum
):
    """
    Test solve_gaintable for phase solution only (with phase_errors),
    for different polarisation frames.
    """
    jones_type = "T"

    vis = vis_with_component_data(sky_pol_frame, data_pol_frame, flux_array)

    gain_table = create_gaintable_from_visibility(vis, jones_type=jones_type)
    gain_table = simulate_gaintable(
        gain_table,
        phase_error=phase_error,
        amplitude_error=0.0,
        leakage=0.0,
    )

    original = vis.copy(deep=True)
    vis = apply_gaintable(vis, gain_table)

    result_gain_table = solve_gaintable(
        vis,
        original,
        phase_only=True,
        niter=200,
        crosspol=False,
        tol=1e-6,
        normalise_gains=False,
        jones_type=jones_type,
    )

    assert result_gain_table["gain"].data.sum().real.round(10) == round(
        expected_gain_sum[0], 10
    )
    assert result_gain_table["gain"].data.sum().imag.round(10) == round(
        expected_gain_sum[1], 10
    )


@pytest.mark.parametrize(
    "sky_pol_frame, data_pol_frame, flux_array, "
    "amplitude_error, expected_gain_sum",
    [
        (
            "stokesIV",
            "circularnp",
            [100.0, 50.0],
            0.01,
            (748.3974636434654, 0.0004744256818732251),
        ),
        (
            "stokesIQUV",
            "circular",
            [100.0, 0.0, 0.0, 50.0],
            0.01,
            (748.3974636434654, 0.0004744256818732251),
        ),
        (
            "stokesIQUV",
            "linear",
            [100.0, 50.0, 0.0, 0.0],
            0.01,
            (748.3974636434654, 0.0004744256818732251),
        ),
        (
            "stokesI",
            "stokesI",
            [100.0, 0.0, 0.0, 0.0],
            0.1,
            (372.3091261026126, 23.89178596680593),
        ),
    ],
)
def test_solve_gaintable_phase_and_amplitude(
    sky_pol_frame,
    data_pol_frame,
    flux_array,
    amplitude_error,
    expected_gain_sum,
):
    """
    Test solve_gaintable with with phase and amplitude errors,
    for different polarisation frames.
    """
    jones_type = "G"

    vis = vis_with_component_data(sky_pol_frame, data_pol_frame, flux_array)

    gt = create_gaintable_from_visibility(vis, jones_type=jones_type)
    gt = simulate_gaintable(
        gt,
        phase_error=0.1,
        amplitude_error=amplitude_error,
        leakage=0.0,
    )
    original = vis.copy(deep=True)
    vis = apply_gaintable(vis, gt)

    result_gain_table = solve_gaintable(
        vis,
        original,
        phase_only=False,
        niter=200,
        crosspol=False,
        tol=1e-6,
        normalise_gains=False,
        jones_type=jones_type,
    )

    assert result_gain_table["gain"].data.sum().real.round(10) == round(
        expected_gain_sum[0], 10
    )
    assert result_gain_table["gain"].data.sum().imag.round(10) == round(
        expected_gain_sum[1], 10
    )


@pytest.mark.parametrize(
    "sky_pol_frame, data_pol_frame, flux_array",
    [
        ("stokesIQ", "linearnp", [100.0, 50.0]),
        ("stokesIQUV", "circular", [100.0, 10.0, -20.0, 50.0]),
        ("stokesIQUV", "circular", [100.0, 0.0, 0.0, 50.0]),
        ("stokesIQUV", "linear", [100.0, 50.0, 10.0, -20.0]),
        ("stokesIQUV", "linear", [100.0, 50.0, 0.0, 0.0]),
    ],
)
def test_solve_gaintable_crosspol(sky_pol_frame, data_pol_frame, flux_array):
    """
    Test solve_gaintable with crosspol=True, for different polarisation frames.
    """
    jones_type = "G"

    vis = vis_with_component_data(sky_pol_frame, data_pol_frame, flux_array)

    gt = create_gaintable_from_visibility(vis, jones_type=jones_type)
    gt = simulate_gaintable(
        gt,
        phase_error=0.1,
        amplitude_error=0.01,
        leakage=0.0,
    )
    original = vis.copy(deep=True)
    vis = apply_gaintable(vis, gt)

    result_gain_table = solve_gaintable(
        vis,
        original,
        phase_only=False,
        niter=200,
        crosspol=True,
        tol=1e-6,
        normalise_gains=False,
        jones_type=jones_type,
    )

    assert result_gain_table["gain"].data.sum().real.round(10) == round(
        748.3974636434654, 10
    )
    assert result_gain_table["gain"].data.sum().imag.round(10) == round(
        0.0004744256818732251, 10
    )


def test_solve_gaintable_timeslice():
    """
    Test solve_gaintable with timeslice set.
    """
    jones_type = "G"

    vis = vis_with_component_data("stokesI", "stokesI", [100.0, 0.0, 0.0, 0.0])

    gt = create_gaintable_from_visibility(vis, jones_type=jones_type)
    gt = simulate_gaintable(
        gt,
        phase_error=0.1,
        amplitude_error=0.1,
        leakage=0.0,
    )
    original = vis.copy(deep=True)
    vis = apply_gaintable(vis, gt)

    result_gain_table = solve_gaintable(
        vis,
        original,
        phase_only=False,
        niter=200,
        crosspol=False,
        tol=1e-6,
        normalise_gains=False,
        jones_type=jones_type,
        timeslice=120.0,
    )

    assert result_gain_table["gain"].data.sum().real == 94.0
    assert result_gain_table["gain"].data.sum().imag == 0.0j


def test_solve_gaintable_normalise():
    """
    Test solve_gaintable with normalise_gains=True.
    """
    jones_type = "G"

    vis = vis_with_component_data("stokesI", "stokesI", [100.0, 0.0, 0.0, 0.0])

    gt = create_gaintable_from_visibility(vis, jones_type=jones_type)
    gt = simulate_gaintable(
        gt,
        phase_error=0.1,
        amplitude_error=0.1,
        leakage=0.0,
    )
    original = vis.copy(deep=True)
    vis = apply_gaintable(vis, gt)

    result_gain_table = solve_gaintable(
        vis,
        original,
        phase_only=False,
        niter=200,
        crosspol=False,
        tol=1e-6,
        normalise_gains=True,
        jones_type=jones_type,
    )

    assert (
        result_gain_table["gain"].data.sum().real.round(10) == 372.3325412643
    )
    assert result_gain_table["gain"].data.sum().imag.round(10) == 23.7294030164


@pytest.mark.parametrize(
    "sky_pol_frame, data_pol_frame, flux_array, "
    "crosspol, nchan, expected_gain_sum",
    [
        (
            "stokesI",
            "stokesI",
            [100.0, 0.0, 0.0, 0.0],
            False,
            32,
            (11920.058493360062, 2.8005384554731485),
        ),
        (
            "stokesIQUV",
            "circular",
            [100.0, 0.0, 0.0, 50.0],
            True,
            4,
            (5986.719933148706, 0.04801930767159601),
        ),
        (
            "stokesIQUV",
            "linear",
            [100.0, 50.0, 0.0, 0.0],
            False,
            32,
            (47888.57884046984, 0.026622457006181932),
        ),
    ],
)
def test_solve_gaintable_bandpass(
    sky_pol_frame,
    data_pol_frame,
    flux_array,
    crosspol,
    nchan,
    expected_gain_sum,
):
    """
    Test solve_gaintable for bandpass solution of multiple channels,
    for different polarisation frames.
    """
    jones_type = "B"

    vis = vis_with_component_data(
        sky_pol_frame, data_pol_frame, flux_array, nchan=nchan
    )

    gt = create_gaintable_from_visibility(vis, jones_type=jones_type)
    gt = simulate_gaintable(
        gt,
        phase_error=0.1,
        amplitude_error=0.1,
        leakage=0.0,
    )
    original = vis.copy(deep=True)
    vis = apply_gaintable(vis, gt)

    result_gain_table = solve_gaintable(
        vis,
        original,
        phase_only=False,
        niter=200,
        crosspol=crosspol,
        tol=1e-6,
        normalise_gains=True,
        jones_type=jones_type,
    )

    assert result_gain_table["gain"].data.sum().real.round(10) == round(
        expected_gain_sum[0], 10
    )
    assert result_gain_table["gain"].data.sum().imag.round(10) == round(
        expected_gain_sum[1], 10
    )


def test_solve_gaintable_few_antennas_many_times():
    """
    Test solve_gaintable for different array size and time samples.
    (Small array, large number of time samples)
    """
    jones_type = "G"

    vis = vis_with_component_data(
        "stokesI", "stokesI", [100.0, 0.0, 0.0, 0.0], rmax=83, ntimes=400
    )

    gt = create_gaintable_from_visibility(vis, jones_type=jones_type)
    gt = simulate_gaintable(
        gt,
        phase_error=0.1,
        amplitude_error=0.1,
        leakage=0.0,
    )
    original = vis.copy(deep=True)
    vis = apply_gaintable(vis, gt)

    result_gain_table = solve_gaintable(
        vis,
        original,
        phase_only=False,
        niter=200,
        crosspol=False,
        tol=1e-6,
        normalise_gains=False,
        jones_type=jones_type,
    )

    assert (
        result_gain_table["gain"].data.sum().real.round(10) == 2393.9044551139
    )
    assert result_gain_table["gain"].data.sum().imag.round(10) == 24.2584023116
