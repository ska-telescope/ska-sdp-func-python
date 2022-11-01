""" Unit tests for calibration solution


"""
import logging
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame
from ska_sdp_datamodels.sky_model.sky_model import SkyComponent

from src.ska_sdp_func_python.calibration.operations import (
    apply_gaintable,
    create_gaintable_from_visibility,
)
from src.ska_sdp_func_python.calibration.solvers import solve_gaintable
from src.ska_sdp_func_python.imaging import dft_skycomponent_visibility
from src.ska_sdp_func_python.simulation import create_named_configuration
from src.ska_sdp_func_python.simulation import simulate_gaintable
from src.ska_sdp_func_python.visibility.base import (
    create_visibility,
)

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)


class TestCalibrationSolvers(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(180555)

    def actualSetup(
        self,
        sky_pol_frame="stokesIQUV",
        data_pol_frame="linear",
        f=None,
        vnchan=3,
        ntimes=3,
        rmax=300.0,
    ):
        self.lowcore = create_named_configuration("LOWBD2", rmax=rmax)
        self.times = (numpy.pi / 43200.0) * numpy.linspace(0.0, 30.0, 1 + ntimes)
        if vnchan > 1:
            self.frequency = numpy.linspace(1.0e8, 1.1e8, vnchan)
            self.channel_bandwidth = numpy.array(
                vnchan * [self.frequency[1] - self.frequency[0]]
            )
        else:
            self.frequency = 1e8 * numpy.ones([1])
            self.channel_bandwidth = 1e7 * numpy.ones([1])

        if f is None:
            f = [100.0, 50.0, -10.0, 40.0]

        if sky_pol_frame == "stokesI":
            f = [100.0]

        self.flux = numpy.outer(
            numpy.array([numpy.power(freq / 1e8, -0.7) for freq in self.frequency]), f
        )

        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre = SkyCoord(
            ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.compabsdirection = SkyCoord(
            ra=+181.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.comp = SkyComponent(
            direction=self.compabsdirection,
            frequency=self.frequency,
            flux=self.flux,
            polarisation_frame=PolarisationFrame(sky_pol_frame),
        )
        self.vis = create_visibility(
            self.lowcore,
            self.times,
            self.frequency,
            phasecentre=self.phasecentre,
            channel_bandwidth=self.channel_bandwidth,
            weight=1.0,
            polarisation_frame=PolarisationFrame(data_pol_frame),
        )
        self.vis = dft_skycomponent_visibility(self.vis, self.comp)

    def test_solve_gaintable_stokesI_phaseonly(self):
        self.core_solve(
            "stokesI",
            "stokesI",
            phase_error=0.1,
            phase_only=True,
            leakage=0.0,
            f=[100.0, 0.0, 0.0, 0.0],
        )

    def test_solve_gaintable_stokesI(self):
        self.core_solve(
            "stokesI",
            "stokesI",
            amplitude_error=0.1,
            phase_error=0.1,
            phase_only=False,
            leakage=0.0,
            f=[100.0, 0.0, 0.0, 0.0],
        )

    def test_solve_gaintable_stokesI_small_n_large_t(self):
        # Select only 6 stations
        self.core_solve(
            "stokesI",
            "stokesI",
            amplitude_error=0.1,
            phase_error=0.1,
            phase_only=False,
            leakage=0.0,
            f=[100.0, 0.0, 0.0, 0.0],
            ntimes=400,
            rmax=83,
        )

    def test_solve_gaintable_stokesI_timeslice(self):
        self.core_solve(
            "stokesI",
            "stokesI",
            amplitude_error=0.1,
            phase_error=0.1,
            phase_only=False,
            leakage=0.0,
            f=[100.0, 0.0, 0.0, 0.0],
            timeslice=120.0,
        )

    def test_solve_gaintable_stokesI_normalise(self):
        self.core_solve(
            "stokesI",
            "stokesI",
            amplitude_error=0.1,
            phase_error=0.1,
            phase_only=False,
            leakage=0.0,
            f=[100.0, 0.0, 0.0, 0.0],
            normalise_gains=True,
        )

    def test_solve_gaintable_stokesI_bandpass(self):
        self.core_solve(
            "stokesI",
            "stokesI",
            amplitude_error=0.1,
            phase_error=0.1,
            phase_only=False,
            leakage=0.0,
            f=[100.0, 0.0, 0.0, 0.0],
            vnchan=32,
        )

    def test_solve_gaintable_stokesIQUV_bandpass(self):
        self.core_solve(
            "stokesIQUV",
            "linear",
            amplitude_error=0.1,
            phase_error=0.1,
            phase_only=False,
            leakage=0.0,
            f=[100.0, 50.0, 0.0, 0.0],
            vnchan=32,
        )

    def core_solve(
        self,
        spf,
        dpf,
        phase_error=0.1,
        amplitude_error=0.0,
        leakage=0.01,
        phase_only=True,
        niter=200,
        crosspol=False,
        residual_tol=1e-6,
        f=None,
        vnchan=1,
        timeslice=None,
        ntimes=3,
        rmax=300,
        normalise_gains=False,
    ):
        if vnchan > 1:
            jones_type = "B"
        else:
            if amplitude_error > 0.0:
                jones_type = "G"
            else:
                jones_type = "T"

        if f is None:
            f = [100.0, 50.0, -10.0, 40.0]
        self.actualSetup(spf, dpf, f=f, vnchan=vnchan, ntimes=ntimes, rmax=rmax)
        gt = create_gaintable_from_visibility(
            self.vis, timeslice=timeslice, jones_type=jones_type
        )
        log.info("Created gain table: %.3f GB" % (gt.gaintable_acc.size()))
        gt = simulate_gaintable(
            gt,
            phase_error=phase_error,
            amplitude_error=amplitude_error,
            leakage=leakage,
        )
        original = self.vis.copy(deep=True)
        vis = apply_gaintable(self.vis, gt)
        gtsol = solve_gaintable(
            self.vis,
            original,
            phase_only=phase_only,
            niter=niter,
            crosspol=crosspol,
            tol=1e-6,
            normalise_gains=normalise_gains,
            jones_type=jones_type,
        )
        cal_vis = apply_gaintable(vis, gtsol, inverse=True)
        assert numpy.max(numpy.abs(vis.vis - cal_vis.vis)) < residual_tol

    def test_solve_gaintable_stokesIQUV_phase_only_linear(self):
        self.core_solve(
            "stokesIQUV",
            "linear",
            phase_error=0.1,
            phase_only=True,
            leakage=0.0,
            f=[100.0, 50.0, 0.0, 0.0],
        )

    def test_solve_gaintable_stokesIQUV_phase_only_circular(self):
        self.core_solve(
            "stokesIQUV",
            "circular",
            phase_error=0.1,
            phase_only=True,
            leakage=0.0,
            f=[100.0, 0.0, 0.0, 50.0],
        )

    def test_solve_gaintable_stokesIQUV_phase_only_linearnp(self):
        self.core_solve(
            "stokesIQ",
            "linearnp",
            phase_error=0.1,
            phase_only=True,
            leakage=0.0,
            f=[100.0, 50.0],
        )

    def test_solve_gaintable_stokesIQUV_phase_only_circularnp(self):
        self.core_solve(
            "stokesIV",
            "circularnp",
            phase_error=0.1,
            phase_only=True,
            leakage=0.0,
            f=[100.0, 50.0],
        )

    def test_solve_gaintable_stokesIQUV_large_phase_only_linear(self):
        self.core_solve(
            "stokesIQUV",
            "linear",
            phase_error=10.0,
            phase_only=True,
            leakage=0.0,
            f=[100.0, 50.0, 0.0, 0.0],
        )

    def test_solve_gaintable_stokesIQUV_large_phase_only_circular(self):
        self.core_solve(
            "stokesIQUV",
            "circular",
            phase_error=10.0,
            leakage=0.0,
            phase_only=True,
            f=[100.0, 0.0, 0.0, 50.0],
        )

    def test_solve_gaintable_stokesIQUV_both_linear(self):
        self.core_solve(
            "stokesIQUV",
            "linear",
            phase_error=0.1,
            amplitude_error=0.01,
            leakage=0.0,
            phase_only=False,
            f=[100.0, 50.0, 0.0, 0.0],
        )

    def test_solve_gaintable_stokesIQUV_both_circular(self):
        self.core_solve(
            "stokesIQUV",
            "circular",
            phase_error=0.1,
            amplitude_error=0.01,
            leakage=0.0,
            phase_only=False,
            f=[100.0, 0.0, 0.0, 50.0],
        )

    def test_solve_gaintable_stokesIV_both_circularnp(self):
        self.core_solve(
            "stokesIV",
            "circularnp",
            phase_error=0.1,
            amplitude_error=0.01,
            leakage=0.0,
            phase_only=False,
            f=[100.0, 50.0],
        )

    def test_solve_gaintable_stokesIQUV_crosspol_both_linear(self):
        self.core_solve(
            "stokesIQUV",
            "linear",
            phase_error=0.1,
            amplitude_error=0.01,
            leakage=0.0,
            residual_tol=1e-6,
            crosspol=True,
            phase_only=False,
            f=[100.0, 50.0, 0.0, 0.0],
        )

    def test_solve_gaintable_stokesIQ_crosspol_both_linearnp(self):
        self.core_solve(
            "stokesIQ",
            "linearnp",
            phase_error=0.1,
            amplitude_error=0.01,
            leakage=0.0,
            residual_tol=1e-6,
            crosspol=True,
            phase_only=False,
            f=[100.0, 50.0],
        )

    def test_solve_gaintable_stokesIQUV_crosspol_both_linear_cross(self):
        self.core_solve(
            "stokesIQUV",
            "linear",
            phase_error=0.1,
            amplitude_error=0.01,
            leakage=0.1,
            residual_tol=1e-6,
            crosspol=True,
            phase_only=False,
            f=[100.0, 50.0, 10.0, -20.0],
        )

    def test_solve_gaintable_stokesIQUV_crosspol_both_circular(self):
        self.core_solve(
            "stokesIQUV",
            "circular",
            phase_error=0.1,
            amplitude_error=0.01,
            leakage=0.0,
            residual_tol=1e-6,
            crosspol=True,
            phase_only=False,
            f=[100.0, 0.0, 0.0, 50.0],
        )

    def test_solve_gaintable_stokesIQUV_crosspol_both_circular_cross(self):
        self.core_solve(
            "stokesIQUV",
            "circular",
            phase_error=0.1,
            amplitude_error=0.01,
            leakage=0.1,
            residual_tol=1e-6,
            crosspol=True,
            phase_only=False,
            f=[100.0, 10.0, -20.0, 50.0],
        )

    def test_solve_gaintable_stokesIQUV_crosspol_both_circular_channel(self):
        self.core_solve(
            "stokesIQUV",
            "circular",
            phase_error=0.1,
            amplitude_error=0.01,
            leakage=0.0,
            residual_tol=1e-6,
            crosspol=True,
            vnchan=4,
            phase_only=False,
            f=[100.0, 0.0, 0.0, 50.0],
        )


if __name__ == "__main__":
    unittest.main()
