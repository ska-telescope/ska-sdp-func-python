"""
Utilities to support rechannelisation of bandpass and delay solutions for CBF
beamformer calibration.
"""

import logging
import time as timer

import numpy
from numpy.polynomial import polynomial
from scipy import interpolate
from ska_sdp_datamodels.calibration.calibration_model import GainTable

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
    f_in = gaintable.frequency.data
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


def expand_delay_phase(delaygaintable, frequency):
    """CASA delay calibration tables with type K or Kcross are currently stored
    in GainTable Jones matrices as phase shifts at a single reference
    frequency. These are expanded to other frequencies assuming
    phase = 2 * pi * t_delay * frequency.
    Note that this only works if the delay is less than half a wavelength at
    the reference frequency. In the future it is likely that the time delay
    will be stored in such GainTables and used directly.

    :param delaygaintable: GainTable with single phase values derived from
        delays. Must have jones_type "K".
    :return: GainTable array with len(frequency) phase values
    """
    if delaygaintable.jones_type != "K":
        raise ValueError(f"Wrong Jones type: {delaygaintable.jones_type} != K")
    # after extrapolating to other frequencies the Jones type will be set to B

    if delaygaintable.frequency.shape[0] != 1:
        raise ValueError("Expect a single frequency")
    frequency0 = delaygaintable.frequency.data[0]

    shape = numpy.array(delaygaintable.gain.shape)
    shape[2] = len(frequency)

    gain = numpy.empty(shape, "complex128")

    # Set the gain weight to one and residual to zero
    weight = numpy.ones(shape)
    residual = numpy.zeros((shape[0], shape[2], shape[3], shape[4]))

    # only works if the delay at ref freq is less than half a wavelength
    phase0 = numpy.angle(delaygaintable.gain.data)
    for chan, freq in enumerate(frequency):
        gain[:, :, chan, :, :] = numpy.exp(
            1j * freq / frequency0 * phase0[:, :, 0, :, :]
        )

    return GainTable.constructor(
        gain=gain,
        time=delaygaintable.time,
        interval=delaygaintable.interval,
        weight=weight,
        residual=residual,
        frequency=frequency,
        receptor_frame=delaygaintable.receptor_frame1,
        phasecentre=delaygaintable.phasecentre,
        configuration=delaygaintable.configuration,
        jones_type="B",
    )


def _set_gaintable_product_shape(gaintable1, gaintable2):
    """Determine the shape of the product of two GainTables

    :param gaintable1: GainTable containing left-hand side Jones matrices
    :param gaintable2: GainTable containing right-hand side Jones matrices
    :return: Shape of the combined GainTable
    """
    gain1 = gaintable1.gain.data
    gain2 = gaintable2.gain.data

    if gain1.shape[0] != gain2.shape[0]:
        raise ValueError("time error {gain1.shape[0]} != {gain2.shape[0]}")
    if gain1.shape[1] != gain2.shape[1]:
        raise ValueError("antenna error {gain1.shape[1]} != {gain2.shape[1]}")
    # Tables must have the same number of channels, unless one set is constant
    # with a single Jones matrix per time and antenna
    if (
        gain1.shape[2] != gain2.shape[2]
        and gain1.shape[2] != 1
        and gain2.shape[2] != 1
    ):
        raise ValueError("shape error {gain1.shape} != {gain2.shape}")
    if gain1.shape[3] != gain2.shape[3]:
        raise ValueError("shape error {gain1.shape[3]} != {gain2.shape[3]}")
    if gain1.shape[4] != gain2.shape[4]:
        raise ValueError("shape error {gain1.shape[4]} != {gain2.shape[4]}")
    # Make sure that ncol of matrix 1 equals nrow of matrix 2
    if gaintable1.receptor2.shape != gaintable2.receptor1.shape:
        raise ValueError("Matrices not compatible for multiplication")

    return (
        gain1.shape[0],
        gain1.shape[1],
        max(gain1.shape[2], gain2.shape[2]),
        gain1.shape[3],
        gain2.shape[4],
    )


def multiply_gaintable_jones(gaintable1, gaintable2):
    """Multiply the Jones matrices for all times, antennas and frequencies
    of two GainTables.

    :param gaintable1: GainTable containing left-hand side Jones matrices
    :param gaintable2: GainTable containing right-hand side Jones matrices
    :return: GainTable containing gaintable1 Jones * gaintable2 Jones
    """
    if gaintable1.jones_type == "K" or gaintable2.jones_type == "K":
        raise ValueError("Cannot multiply delays. Use expand_delay_phase")

    shape = _set_gaintable_product_shape(gaintable1, gaintable2)

    gain = numpy.empty(shape, "complex128")

    gain1 = gaintable1.gain.data
    gain2 = gaintable2.gain.data

    # Map output channel indices to input channel indices
    chan1 = numpy.arange(shape[2]).astype("int")
    chan2 = numpy.arange(shape[2]).astype("int")
    if gain1.shape[2] == 1:
        chan1 *= 0
    if gain2.shape[2] == 1:
        chan2 *= 0

    for time in range(0, shape[0]):
        for ant in range(0, shape[1]):
            for chan in range(0, shape[2]):
                gain[time, ant, chan] = (
                    gain1[time, ant, chan1[chan]]
                    @ gain2[time, ant, chan2[chan]]
                )

    # Get the frequencies, noting that one set may be of length 1
    if gain1.shape[2] > 1:
        frequency = gaintable1.frequency.data
        weight = gaintable1.weight
        residual = gaintable1.residual
    else:
        frequency = gaintable2.frequency.data
        weight = gaintable2.weight
        residual = gaintable2.residual

    # If the two tables have the same jones_type use that, otherwise use B.
    if gaintable1.jones_type == gaintable2.jones_type:
        jones_type = gaintable1.jones_type
    else:
        jones_type = "B"

    return GainTable.constructor(
        gain=gain,
        time=gaintable1.time,
        interval=gaintable1.interval,
        weight=weight,
        residual=residual,
        frequency=frequency,
        receptor_frame=gaintable1.receptor_frame1,
        phasecentre=gaintable1.phasecentre,
        configuration=gaintable1.configuration,
        jones_type=jones_type,
    )


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

    f_in = gaintable.frequency.data

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

    gain = gaintable.gain.data
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
