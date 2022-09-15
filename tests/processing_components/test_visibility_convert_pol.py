""" Unit tests for visibility operations


"""

import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from rascil.data_models.memory_data_models import SkyComponent
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.visibility.base import create_visibility
from rascil.processing_components.visibility.operations import (
    convert_visibility_to_stokesI,
    convert_visibility_to_stokes,
    convert_visibility_stokesI_to_polframe,
)


class TestVisibilityConvertPol(unittest.TestCase):
    def setUp(self):
        self.lowcore = create_named_configuration("LOWBD2-CORE")
        self.times = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 150.0)
        self.frequency = numpy.linspace(1.0e8, 1.1e8, 3)
        self.channel_bandwidth = numpy.array([1e7, 1e7, 1e7])
        # Define the component and give it some spectral behaviour
        f = numpy.array([100.0, 20.0, -10.0, 1.0])
        self.flux = numpy.array([f, 0.8 * f, 0.6 * f])

        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre = SkyCoord(
            ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.compabsdirection = SkyCoord(
            ra=+181.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
        )
        pcof = self.phasecentre.skyoffset_frame()
        self.compreldirection = self.compabsdirection.transform_to(pcof)
        self.comp = SkyComponent(
            direction=self.compreldirection, frequency=self.frequency, flux=self.flux
        )

    def test_convert_visibility_I(self):
        for pol in ["linear", "circular"]:
            vis = create_visibility(
                self.lowcore,
                self.times,
                self.frequency,
                channel_bandwidth=self.channel_bandwidth,
                phasecentre=self.phasecentre,
                weight=1.0,
                polarisation_frame=PolarisationFrame(pol),
            )
            visi = convert_visibility_to_stokesI(vis)
            assert visi.visibility_acc.polarisation_frame.type == "stokesI"
            assert visi.visibility_acc.npol == 1

    def test_convert_visibility_I_polframe(self):
        for pol in ["linear", "circular"]:
            vis = create_visibility(
                self.lowcore,
                self.times,
                self.frequency,
                channel_bandwidth=self.channel_bandwidth,
                phasecentre=self.phasecentre,
                weight=1.0,
                polarisation_frame=PolarisationFrame("stokesI"),
            )
            assert vis.visibility_acc.npol == 1
            visp = convert_visibility_stokesI_to_polframe(
                vis, poldef=PolarisationFrame(pol)
            )
            assert visp.visibility_acc.polarisation_frame.type == pol
            assert visp.visibility_acc.npol == 4

    def test_convert_visibility_stokes(self):
        for pol in ["linear", "circular"]:
            vis = create_visibility(
                self.lowcore,
                self.times,
                self.frequency,
                channel_bandwidth=self.channel_bandwidth,
                phasecentre=self.phasecentre,
                weight=1.0,
                polarisation_frame=PolarisationFrame(pol),
            )
            visi = convert_visibility_to_stokes(vis)
            assert visi.polarisation_frame.type == "stokesIQUV"
            assert visi.visibility_acc.npol == 4


if __name__ == "__main__":
    unittest.main()
