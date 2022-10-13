""" Unit tests for image operations

    For class methods in the Image class

"""
import logging
import os
import unittest

import astropy.units as u
import matplotlib.pyplot as plt
import numpy
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation_data_models import PolarisationFrame
from rascil.processing_components import (
    create_test_image,
    show_image,
)

log = logging.getLogger("rascil-logger")
log.setLevel(logging.WARNING)


class TestImageClassMethods(unittest.TestCase):
    def setUp(self):

        from rascil.processing_components.parameters import rascil_path

        self.results_dir = rascil_path("test_results")

        self.image = create_test_image(
            cellsize=0.0001,
            phasecentre=SkyCoord(
                ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
            ),
            frequency=numpy.linspace(0.8e9, 1.2e9, 5),
            polarisation_frame=PolarisationFrame("stokesIQUV"),
            channel_bandwidth=1e7 * numpy.ones([5]),
        )

        self.cellsize = 180.0 * 0.0001 / numpy.pi
        self.persist = os.getenv("RASCIL_PERSIST", False)

    def test_image_where_radius_radec(self):
        self.image = self.image.add_ra_dec()
        secd = 1.0 / numpy.cos(numpy.deg2rad(self.image.dec))
        r = numpy.hypot(
            (self.image.ra_grid - self.image.ra) * secd,
            self.image.dec_grid - self.image.dec,
        )
        show_image(self.image.where(r < 0.3, 0.0))
        plt.show()


if __name__ == "__main__":
    unittest.main()
