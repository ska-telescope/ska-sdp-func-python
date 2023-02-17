"""
Utilities to support rechannelisation of bandpass and delay solutions for CBF
beamformer calibration.
"""

import logging
import time as timer

import numpy
from numpy.polynomial import polynomial
from scipy import interpolate

log = logging.getLogger("func-python-logger")


def set_beamformer_frequencies(gt):
    """Generate a list of output frequencies

    SKA-Low beamformer:
     - Need Jones matrices for 384, 304 or 152 channels, per antenna, per beam
     - Station channels/beams are centred on integer multiples of 781.25 kHz
           781.25 kHz = 400 MHz / 512 channels
     - Station channels run from 50 MHz (64*df) to 350 MHz (448*df)
     - CBF beamformer calibration is done at the station channel resolution
     - Will assume that no padding is needed if input band < beamformer band

    MID beamformer
     - Need Jones matrices for 4096 channels, per antenna, per beam
     - Timing beam bandwidth : 200 MHz (channel width : 48.8281250 kHz?)
     - Search beam bandwidth : 300 MHz (channel width : 73.2421875 kHz?)

    :param gt: GainTable
    :return: numpy array of shape [nfreq,]
    """

    # determine array
    array_name = gt.configuration.name
    # determine beamformer type
    # bf_mode = ? get from function argument?
    log.info(f"Setting frequency for the {array_name} beamformer")

    # initial frequencies
    f_in = gt["frequency"].data
    nf_in = len(f_in)

    if nf_in <= 1:
        log.warning(f"Cannot rechannelise frequencies: {f_in}")
        return f_in

    if array_name.find("LOW") == 0:
        log.debug("Setting SKA-Low CBF beamformer frequencies")
        df_out = 781.25e3
        f0_out = df_out * numpy.round(numpy.amin(f_in) / df_out)
    elif array_name.find("MID") == 0:
        log.debug("Setting SKA-Mid CBF beamformer frequencies")
        df_out = 300e6 / 4096
        f0_out = numpy.amin(f_in)  # are there specific channel centres?
    else:
        log.warning(f"Unknown array: {array_name}. Frequencies unchanged")
        return f_in

    return numpy.arange(f0_out, numpy.amax(f_in), df_out)


def resample_bandpass(f_out, gt, alg="polyfit", edges=[]):
    """Re-channelise each spectrum of gain or leakage terms

    algorithms:
     - polyfit  numpy.polynomial.polyval [default]
     - interp   numpy.interp
     - interp1d scipy.interpolate.interp1d, lind=linear
     - cubicspl scipy.interpolate.CubicSpline

    :param f_out: numpy array of shape [nfreq_out,]
    :param gt: GainTable
    :param alg: algorithm type [default polyfit]
    :param edges: [default none]
    :return: numpy array of shape [nfreq_out,]
    """

    f_in = gt["frequency"].data

    if alg == "polyfit":
        sel = polyfit()
        if len(edges) > 0:
            sel.set_edges(edges)
    elif alg == "interp":
        sel = interp()
    elif alg == "interp1d":
        sel = interp1d()
    elif alg == "cubicspl":
        sel = cubicspl()

    ntime = len(gt["time"])
    nantenna = len(gt["antenna"])
    nreceptor1 = len(gt["receptor1"])
    nreceptor2 = len(gt["receptor2"])

    gain = gt["gain"].data
    gain_out = numpy.empty(
        (ntime, nantenna, len(f_out), nreceptor1, nreceptor2), "complex128"
    )
    t0 = timer.perf_counter()
    for t in range(0, ntime):
        for ant in range(0, nantenna):
            for r1 in range(0, nreceptor1):
                for r2 in range(0, nreceptor2):
                    gain_out[t, ant, :, r1, r2] = sel.interp(
                        f_out, f_in, gain[t, ant, :, r1, r2]
                    )
    log.info(
        "{:<11} took {:.1f} seconds".format(alg, timer.perf_counter() - t0)
    )

    return gain_out


class polyfit:
    """fit the data using polynomials

    algorithms:
     - polyfit  numpy.polynomial.polyval [default]
     - interp   numpy.interp
     - interp1d scipy.interpolate.interp1d, lind=linear
     - cubicspl scipy.interpolate.CubicSpline

    :param f_out: numpy array of shape [nfreq_out,]
    :param gt: GainTable
    :param alg: algorithm type [default polyfit]
    :param edges: [default none]
    :return: numpy array of shape [nfreq_out,]
    """

    def __init__(self):
        self.edges = []
        self.polydeg = 3

    def set_edges(self, edges):
        self.edges = edges

    def interp(self, f_out, f_in, gain):
        if len(self.edges) == 0:
            self.edges = numpy.array([-1, len(f_in) - 1])
            log.debug(f"set edges to {self.edges}")

        idx_out = numpy.arange(0, len(f_out)).astype("int")
        gain_out = numpy.empty(len(f_out), "complex128")

        df_in = f_in[1] - f_in[0]
        for k in range(1, len(self.edges)):
            ch_in = numpy.arange(
                self.edges[k - 1] + 1, self.edges[k] + 1
            ).astype("int")
            ch_out = idx_out[
                (f_out >= f_in[self.edges[k - 1] + 1] - df_in / 2)
                * (f_out < f_in[self.edges[k]] + df_in / 2)
            ]
            # fit the data using polynomials
            coef_re = polynomial.polyfit(
                f_in[ch_in], numpy.real(gain[ch_in]), self.polydeg
            )
            coef_im = polynomial.polyfit(
                f_in[ch_in], numpy.imag(gain[ch_in]), self.polydeg
            )
            # evaluated the fits at the output frequencies
            fit_re = polynomial.polyval(f_out[ch_out], coef_re)
            fit_im = polynomial.polyval(f_out[ch_out], coef_im)
            gain_out[ch_out] = fit_re + 1j * fit_im

        return gain_out


class interp:
    def interp(self, f_out, f_in, gain):
        return numpy.interp(f_out, f_in, gain)


class interp1d:
    def interp(self, f_out, f_in, gain):
        fn = interpolate.interp1d(f_in, gain, kind="linear")
        return fn(f_out)


class cubicspl:
    def interp(self, f_out, f_in, gain):
        fn = interpolate.CubicSpline(f_in, gain)
        return fn(f_out)
