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


def set_beamformer_frequencies(gaintable):
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

    :param gaintable: GainTable
    :return: numpy array of shape [nfreq,]
    """

    # determine array
    array_name = gaintable.configuration.name
    # determine beamformer type
    # bf_mode = ? get from function argument?
    log.info("Setting frequency for the %s beamformer", array_name)

    # initial frequencies
    f_in = gaintable["frequency"].data
    nf_in = len(f_in)

    if nf_in <= 1:
        log.warning("Cannot rechannelise %d channel[s]", nf_in)
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
        log.warning("Unknown array: %s. Frequencies unchanged", array_name)
        return f_in

    return numpy.arange(f0_out, numpy.amax(f_in), df_out)


def resample_bandpass(f_out, gaintable, alg="polyfit", edges=None):
    """Re-channelise each spectrum of gain or leakage terms

    algorithms:
     - polyfit  numpy.polynomial.polyval [default]
     - interp   numpy.interp
     - interp1d scipy.interpolate.interp1d, lind=linear
     - cubicspl scipy.interpolate.CubicSpline

    :param f_out: numpy array of shape [nfreq_out,]
    :param gaintable: GainTable
    :param alg: algorithm type [default polyfit]
    :param edges: list of edges (polyfit only) [default none]
    :return: numpy array of shape [nfreq_out,]
    """

    f_in = gaintable["frequency"].data

    if alg == "polyfit":
        sel = PolynomialInterpolator()
        if edges is not None:
            sel.set_edges(edges)
    elif alg == "interp":
        sel = NumpyLinearInterpolator()
    elif alg == "interp1d":
        sel = ScipyLinearInterpolator()
    elif alg == "cubicspl":
        sel = ScipySplineInterpolator()

    gain = gaintable["gain"].data
    shape_out = numpy.array(gain.shape)
    shape_out[2] = len(f_out)
    gain_out = numpy.empty(shape_out, "complex128")
    timer0 = timer.perf_counter()
    for time in range(0, shape_out[0]):
        for ant in range(0, shape_out[1]):
            for rec1 in range(0, shape_out[3]):
                for rec2 in range(0, shape_out[4]):
                    gain_out[time, ant, :, rec1, rec2] = sel.interp(
                        f_out, f_in, gain[time, ant, :, rec1, rec2]
                    )
    log.warning("%11s took %.1f seconds", alg, timer.perf_counter() - timer0)

    return gain_out


class PolynomialInterpolator:
    """fit the data using the numpy polynomial polyfit function

    Attributes
    ----------
    edges : numpy array
        A vector containing the starting channels of any band intervals
        requiring separate fits. Defaults to the full band.
        Internally, full-band edge channels are appended: [0, ..., nchan].
    polydeg : int [default 3]
        Order of the polynomial fit

    Methods
    -------
    set_edges(edges):
        Provide the start channels of any sub-bands requiring separate fits

    set_polydeg(polydeg):
        Update the order of the polynomial fit

    interp(self, f_out, f_in, gain):
        Do the interpolation for the gain in "gain"

    """

    def __init__(self):
        self.edges = None
        self.polydeg = 3

    def set_edges(self, edges):
        """Provide the start channels of any sub-bands requiring separate fits

        :param edges: list of edges (starting channel indices)

        """
        self.edges = edges

    def set_polydeg(self, polydeg):
        """Update the order of the polynomial fit

        :param polydeg: Order of the polynomial fit

        """
        self.polydeg = polydeg

    def interp(self, f_out, f_in, gain):
        """Do the interpolation for the complex data in "gain"

        :param f_out: numpy array of shape [len(f_out)]
            final frequency values
        :param f_in: numpy array of shape [len(f_in)]
            initial frequency values
        :param gain: numpy array of shape [len(f_in)]
            complex sequence to interpolate
        :return: numpy array of shape [len(f_out)]
            interpolated complex sequence

        """
        if self.edges is None or self.edges == []:
            self.edges = numpy.array([0, len(f_in)])
            fstr = f"set edges to {self.edges}"
            log.debug("set edges to %s", fstr)
        # ensure that the channel before the first discontinuity are included
        if self.edges[0] > 0:
            self.edges = numpy.concatenate(([0], self.edges))
        # ensure that the channel after the last discontinuity are included
        if self.edges[-1] < len(f_in):
            self.edges = numpy.concatenate((self.edges, [len(f_in)]))

        idx_out = numpy.arange(0, len(f_out)).astype("int")
        gain_out = numpy.empty(len(f_out), "complex128")

        df_in = f_in[1] - f_in[0]
        edges = self.edges
        for k in range(0, len(edges) - 1):
            ch_in = numpy.arange(edges[k], edges[k + 1]).astype("int")
            ch_out = idx_out[
                (f_out >= f_in[edges[k]] - df_in / 2)
                * (f_out < f_in[edges[k + 1] - 1] + df_in / 2)
            ]

            # fit the data using polynomials
            coef_re = polynomial.polyfit(
                f_in[ch_in], numpy.real(gain[ch_in]), self.polydeg
            )
            coef_im = polynomial.polyfit(
                f_in[ch_in], numpy.imag(gain[ch_in]), self.polydeg
            )
            # evaluated the fits at the output frequencies
            gain_out[ch_out] = (
                polynomial.polyval(f_out[ch_out], coef_re)
                + polynomial.polyval(f_out[ch_out], coef_im) * 1j
            )

        return gain_out


# could add the extrapolation options instead of disabling the pylint
# error, or just remove this interpolator. It is pretty simple and fast.
class NumpyLinearInterpolator:  # pylint: disable=too-few-public-methods
    """fit the data using the numpy interp function

    Methods
    -------

    interp(self, f_out, f_in, gain):
        Do the interpolation for the gain in "gain"

    """

    def interp(self, f_out, f_in, gain):
        """Do the interpolation for the complex data in "gain"

        :param f_out: numpy array of shape [len(f_out)]
            final frequency values
        :param f_in: numpy array of shape [len(f_in)]
            initial frequency values
        :param gain: numpy array of shape [len(f_in)]
            complex sequence to interpolate
        :return: numpy array of shape [len(f_out)]
            interpolated complex sequence

        """
        return numpy.interp(f_out, f_in, gain)


class ScipyLinearInterpolator:
    """fit the data using the scipy interpolate interp1d function

    Attributes
    ----------
    kind : str [default "linear"]
        The kind of interpolation. Any supported by interp1d.

    Methods
    -------
    set_kind(kind):
        Update the kind of interpolation

    interp(self, f_out, f_in, gain):
        Do the interpolation for the gain in "gain"

    """

    def __init__(self):
        self.kind = "linear"

    def set_kind(self, kind):
        """Update the kind of interpolation

        :param kind: str [default "linear"]
            The kind of interpolation. Any supported by interp1d

        """
        self.kind = kind

    def interp(self, f_out, f_in, gain):
        """Do the interpolation for the complex data in "gain"

        :param f_out: numpy array of shape [len(f_out)]
            final frequency values
        :param f_in: numpy array of shape [len(f_in)]
            initial frequency values
        :param gain: numpy array of shape [len(f_in)]
            complex sequence to interpolate
        :return: numpy array of shape [len(f_out)]
            interpolated complex sequence

        """
        func = interpolate.interp1d(f_in, gain, kind=self.kind)
        return func(f_out)


# could add the extrapolation options instead of disabling the pylint
class ScipySplineInterpolator:  # pylint: disable=too-few-public-methods
    """fit the data using the scipy interpolate CubicSpline function

    Methods
    -------
    interp(self, f_out, f_in, gain):
        Do the interpolation for the gain in "gain"

    """

    def interp(self, f_out, f_in, gain):
        """Do the interpolation for the complex data in "gain"

        :param f_out: numpy array of shape [len(f_out)]
            final frequency values
        :param f_in: numpy array of shape [len(f_in)]
            initial frequency values
        :param gain: numpy array of shape [len(f_in)]
            complex sequence to interpolate
        :return: numpy array of shape [len(f_out)]
            interpolated complex sequence

        """
        func = interpolate.CubicSpline(f_in, gain)
        return func(f_out)
