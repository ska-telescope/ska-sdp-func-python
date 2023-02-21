# pylint: disable=invalid-name
"""
Unit tests for beamformer utils
"""
import numpy
from ska_sdp_datamodels.calibration.calibration_create import (
    create_gaintable_from_visibility,
)

from ska_sdp_func_python.calibration.beamformer_utils import (
    expand_delay_phase,
    multiply_gaintable_jones,
    resample_bandpass,
    set_beamformer_frequencies,
)
from tests.testing_utils import simulate_gaintable, vis_with_component_data


def test_expand_delay_phase():
    """
    Test expand_delay_phase
    """

    vis = vis_with_component_data(
        "stokesIQUV", "linear", [1.0, 0.0, 0.0, 0.0], nchan=5, ntimes=4
    )

    jones_type = "G"
    gt = create_gaintable_from_visibility(vis, jones_type=jones_type)
    assert gt.frequency.shape[0] == 1

    gtK = simulate_gaintable(
        gt,
        phase_error=0.1,
        amplitude_error=0.0,
        leakage=0.0,
    )

    # make the Jones type K and extrapolate across the full vis band
    gtK["jones_type"] = "K"
    gtB = expand_delay_phase(gtK, vis.frequency.data)
    assert gtB.frequency.shape == vis.frequency.shape

    time = 2
    ant = 5
    freq0 = gtK.frequency.data[0]
    freq = gtB.frequency.data - freq0
    phases = numpy.exp(
        1j * freq / freq0 * numpy.angle(gtK["gain"].data[time, ant, 0, 0, 0])
    )
    assert (
        numpy.abs(gtB["gain"].data[time, ant, :, 0, 0] - phases) < 1e-12
    ).all()
    phases = numpy.exp(
        1j * freq / freq0 * numpy.angle(gtK["gain"].data[time, ant, 0, 1, 1])
    )
    assert (
        numpy.abs(gtB["gain"].data[time, ant, :, 1, 1] - phases) < 1e-12
    ).all()


def test_multiply_gaintable_jones():
    """
    Test multiply_gaintable_jones
    """

    vis = vis_with_component_data(
        "stokesIQUV", "linear", [1.0, 0.0, 0.0, 0.0], nchan=5, ntimes=4
    )

    jones_type = "G"
    gt1 = create_gaintable_from_visibility(vis, jones_type=jones_type)
    assert gt1.frequency.shape[0] == 1

    jones_type = "B"
    gt2 = create_gaintable_from_visibility(vis, jones_type=jones_type)
    assert gt2.frequency.shape[0] == 5

    gt1 = simulate_gaintable(
        gt1,
        phase_error=0.1,
        amplitude_error=0.1,
        leakage=0.0,
    )

    gt2 = simulate_gaintable(
        gt2,
        phase_error=0.1,
        amplitude_error=0.1,
        leakage=0.0,
    )

    gt = multiply_gaintable_jones(gt1, gt2)
    assert gt.frequency.shape[0] == 5

    time = 2
    ant = 5
    chan = 3

    J1 = gt1.gain.data[time, ant, 0]
    J2 = gt2.gain.data[time, ant, chan]
    J = gt.gain.data[time, ant, chan]
    assert numpy.array_equal(J1 @ J2, J)

    jones_type = "B"
    gt1 = create_gaintable_from_visibility(vis, jones_type=jones_type)
    assert gt1.frequency.shape[0] == 5

    jones_type = "B"
    gt2 = create_gaintable_from_visibility(vis, jones_type=jones_type)
    assert gt2.frequency.shape[0] == 5
    # make the Jones type D
    gt2["jones_type"] = "D"

    gt1 = simulate_gaintable(
        gt1,
        phase_error=0.1,
        amplitude_error=0.1,
        leakage=0.0,
    )

    gt2 = simulate_gaintable(
        gt2,
        phase_error=1e-18,  # need to set this. Bug in simulate_gaintable
        amplitude_error=0.0,
        leakage=0.1,
    )

    gt = multiply_gaintable_jones(gt1, gt2)
    assert gt.frequency.shape[0] == 5

    time = 2
    ant = 5
    chan = 3

    J1 = gt1.gain.data[time, ant, chan]
    J2 = gt2.gain.data[time, ant, chan]
    J = gt.gain.data[time, ant, chan]
    assert numpy.array_equal(J1 @ J2, J)


def test_set_beamformer_frequencies():
    """
    Test set_beamformer_frequencies (SKA-Low)
    """

    vis = vis_with_component_data(
        "stokesIQUV", "linear", [1.0, 0.0, 0.0, 0.0], nchan=5, ntimes=4
    )

    jones_type = "B"
    gt = create_gaintable_from_visibility(vis, jones_type=jones_type)
    assert gt.frequency.shape[0] == 5

    f_out = set_beamformer_frequencies(gt)
    assert (set_beamformer_frequencies(gt) % 781.25e3 == 0).all()
    assert f_out[0] >= set_beamformer_frequencies(gt)[0]
    assert f_out[-1] <= set_beamformer_frequencies(gt)[-1]


def test_resample_bandpass():
    """
    Test resample_bandpass
    """

    vis = vis_with_component_data(
        "stokesIQUV", "linear", [1.0, 0.0, 0.0, 0.0], nchan=5, ntimes=4
    )

    jones_type = "B"
    gt = create_gaintable_from_visibility(vis, jones_type=jones_type)

    f_out = set_beamformer_frequencies(gt)

    # add a small gain change to one gain term and make sure the fits agree
    time = 2
    ant = 5
    gt["gain"].data[time, ant, :, 0, 0] = numpy.exp(
        1j * 2 * numpy.pi * 1e-8 * gt.frequency.data
    )

    gaintrue = numpy.exp(1j * 2 * numpy.pi * 1e-8 * f_out)
    gainfit1 = resample_bandpass(f_out, gt, alg="polyfit")
    gainfit2 = resample_bandpass(f_out, gt, alg="interp")
    gainfit3 = resample_bandpass(f_out, gt, alg="interp1d")
    gainfit4 = resample_bandpass(f_out, gt, alg="cubicspl")

    assert (numpy.abs(gainfit1[time, ant, :, 0, 0] - gaintrue) < 1e-4).all()
    assert (numpy.abs(gainfit2[time, ant, :, 0, 0] - gaintrue) < 1e-2).all()
    assert (numpy.abs(gainfit3[time, ant, :, 0, 0] - gaintrue) < 1e-2).all()
    assert (numpy.abs(gainfit4[time, ant, :, 0, 0] - gaintrue) < 1e-4).all()
