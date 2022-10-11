"""Unit tests for visibility selectors

"""
import logging
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation_data_models import PolarisationFrame
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.visibility.base import create_visibility
from rascil.processing_components.visibility.visibility_selection import (
    visibility_select_r_range,
    visibility_select_uv_range,
)

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)


class TestVisibilitySelectors(unittest.TestCase):
    def setUp(self):
        self.lowcore = create_named_configuration("LOWBD2-CORE")
        self.times = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 30.0)
        self.frequency = numpy.linspace(0.8e8, 1.2e8, 5)
        self.channel_bandwidth = numpy.array([1e7, 1e7, 1e7, 1e7, 1e7])
        # Define the component and give it some spectral behaviour
        f = numpy.array([100.0, 20.0, -10.0, 1.0])
        self.flux = numpy.array([f, 0.8 * f, 0.6 * f])
        self.polarisation_frame = PolarisationFrame("linear")

        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre = SkyCoord(
            ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
        )

    def test_visibility_groupby_time(self):
        bvis = create_visibility(
            self.lowcore,
            self.times,
            self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            polarisation_frame=self.polarisation_frame,
            phasecentre=self.phasecentre,
            weight=1.0,
        )
        times = numpy.array([result[0] for result in bvis.groupby("time")])
        assert times.all() == bvis.time.all()

    def test_visibility_groupby_bins_time(self):
        bvis = create_visibility(
            self.lowcore,
            self.times,
            self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            polarisation_frame=self.polarisation_frame,
            phasecentre=self.phasecentre,
            weight=1.0,
        )
        for result in bvis.groupby_bins("time", 3):
            log.info(result[0])

    def test_visibility_iselect_time(self):
        bvis = create_visibility(
            self.lowcore,
            self.times,
            self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            polarisation_frame=self.polarisation_frame,
            phasecentre=self.phasecentre,
            weight=1.0,
        )
        selected_bvis = bvis.isel({"time": slice(5, 7)})
        log.info(selected_bvis)
        assert len(selected_bvis.time) == 2
        assert len(selected_bvis.channel_bandwidth.shape) == 1
        assert len(selected_bvis.integration_time.shape) == 1

    def test_visibility_select_time(self):
        bvis = create_visibility(
            self.lowcore,
            self.times,
            self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            polarisation_frame=self.polarisation_frame,
            phasecentre=self.phasecentre,
            weight=1.0,
        )
        time_slice = bvis.time.data[1:4]
        selected_bvis = bvis.sel({"time": time_slice})
        assert len(selected_bvis.time) == 3
        assert len(selected_bvis.channel_bandwidth.shape) == 1
        assert len(selected_bvis.integration_time.shape) == 1

    def test_visibility_select_frequency(self):
        bvis = create_visibility(
            self.lowcore,
            self.times,
            self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            polarisation_frame=self.polarisation_frame,
            phasecentre=self.phasecentre,
            weight=1.0,
        )
        selected_bvis = bvis.sel({"frequency": slice(0.9e8, 1.2e8)})
        log.info(selected_bvis)
        assert len(selected_bvis.frequency) == 4
        assert len(selected_bvis.channel_bandwidth.shape) == 1
        assert len(selected_bvis.integration_time.shape) == 1

    def test_visibility_select_frequency_polarisation(self):
        bvis = create_visibility(
            self.lowcore,
            self.times,
            self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            polarisation_frame=self.polarisation_frame,
            phasecentre=self.phasecentre,
            weight=1.0,
        )
        selected_bvis = bvis.sel(
            {"frequency": slice(0.9e8, 1.2e8), "polarisation": ["XX", "YY"]}
        ).dropna(dim="frequency", how="all")
        log.info(selected_bvis)
        assert len(selected_bvis.frequency) == 4
        assert len(selected_bvis.polarisation) == 2
        assert len(selected_bvis.channel_bandwidth.shape) == 1
        assert len(selected_bvis.integration_time.shape) == 1

    def test_visibility_iselect_channel(self):
        bvis = create_visibility(
            self.lowcore,
            self.times,
            self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            polarisation_frame=self.polarisation_frame,
            phasecentre=self.phasecentre,
            weight=1.0,
        )
        selected_bvis = bvis.isel({"frequency": slice(1, 3)})
        log.info(selected_bvis)
        assert len(selected_bvis.frequency) == 2

    def test_visibility_flag_auto(self):
        bvis = create_visibility(
            self.lowcore,
            self.times,
            self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            polarisation_frame=self.polarisation_frame,
            phasecentre=self.phasecentre,
            weight=1.0,
        )
        flags_shape = bvis.flags.shape
        before = bvis["flags"].sum()
        uvdist_lambda = numpy.hypot(
            bvis.visibility_acc.uvw_lambda[..., 0],
            bvis.visibility_acc.uvw_lambda[..., 1],
        )
        bvis["flags"].data[numpy.where(uvdist_lambda <= 20000.0)] = 1
        assert bvis.flags.shape == flags_shape
        after = bvis["flags"].sum()
        assert after > before
        log.info(bvis)

    def test_visibility_select_uvrange(self):
        bvis = create_visibility(
            self.lowcore,
            self.times,
            self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            polarisation_frame=self.polarisation_frame,
            phasecentre=self.phasecentre,
            weight=1.0,
        )
        before = bvis["flags"].sum()
        uvmin = 100.0
        uvmax = 20000.0

        assert bvis["flags"].sum() == 0
        bvis = visibility_select_uv_range(bvis, uvmin, uvmax)
        assert bvis["flags"].sum() == 1185464
        assert bvis.frequency.shape == (5,)

    def test_visibility_select_r_range(self):
        """Expected number of baselines was calculated from a manual inspection
        of the configuration file.
        """
        bvis = create_visibility(
            self.lowcore,
            self.times,
            self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            polarisation_frame=self.polarisation_frame,
            phasecentre=self.phasecentre,
            weight=1.0,
        )
        rmin = 100.0
        rmax = 20000.0

        sub_bvis = visibility_select_r_range(bvis, rmin, rmax)
        assert len(sub_bvis.baselines) == 11781
        assert len(sub_bvis.configuration.names) == 166
        assert sub_bvis.frequency.shape == (5,)
        assert sub_bvis.integration_time.shape == bvis.integration_time.shape
