""" Unit tests for visibility operations
    
    
"""
import logging
import os
import sys
import unittest

import numpy

from rascil.data_models.polarisation_data_models import PolarisationFrame
from rascil.processing_components import (
    invert_visibility,
    create_image_from_visibility,
)
from rascil.processing_components.parameters import rascil_path, rascil_data_path
from rascil.processing_components.visibility.base import create_visibility_from_uvfits
from rascil.processing_components.visibility.operations import (
    integrate_visibility_by_channel,
)

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))


class TestCreateMS(unittest.TestCase):
    def setUp(self):
        self.results_dir = rascil_path("test_results")

        self.persist = os.getenv("RASCIL_PERSIST", False)

        self.doplot = False

        return

    def test_create_list_spectral(self):

        uvfitsfile = rascil_data_path("vis/ASKAP_example.fits")

        vis_by_channel = list()
        nchan_ave = 16
        nchan = 192
        for schan in range(0, nchan, nchan_ave):
            max_chan = min(nchan, schan + nchan_ave)
            v = create_visibility_from_uvfits(uvfitsfile, range(schan, max_chan))
            vis_by_channel.append(v[0])

        assert len(vis_by_channel) == 12
        for v in vis_by_channel:
            assert v.vis.data.shape[-1] == 4
            assert v.visibility_acc.polarisation_frame.type == "linear"

    def test_create_list_spectral_average(self):

        uvfitsfile = rascil_data_path("vis/ASKAP_example.fits")

        vis_by_channel = list()
        nchan_ave = 16
        nchan = 192
        for schan in range(0, nchan, nchan_ave):
            max_chan = min(nchan, schan + nchan_ave)
            v = create_visibility_from_uvfits(uvfitsfile, range(schan, max_chan))
            vis_by_channel.append(integrate_visibility_by_channel(v[0]))

        assert len(vis_by_channel) == 12
        for v in vis_by_channel:
            assert v.vis.data.shape[-1] == 4
            assert v.vis.data.shape[-2] == 1
            assert v.visibility_acc.polarisation_frame.type == "linear"

    def test_invert(self):

        uvfitsfile = rascil_data_path("vis/ASKAP_example.fits")

        nchan_ave = 32
        nchan = 192
        for schan in range(0, nchan, nchan_ave):
            max_chan = min(nchan, schan + nchan_ave)
            vis = create_visibility_from_uvfits(uvfitsfile, range(schan, max_chan))[0]
            model = create_image_from_visibility(
                vis, npixel=256, polarisation_frame=PolarisationFrame("stokesI")
            )
            dirty, sumwt = invert_visibility(vis, model, context="2d")
            assert (numpy.max(numpy.abs(dirty["pixels"].data))) > 0.0
            assert dirty["pixels"].shape == (nchan_ave, 1, 256, 256)
            if self.doplot:
                import matplotlib.pyplot as plt
                from rascil.processing_components.image.operations import show_image

                show_image(dirty)
                plt.show(block=False)
            if self.persist:
                dirty.export_to_fits(
                    "%s/test_visibility_uvfits_dirty.fits" % self.results_dir
                )

            if schan == 0:
                qa = dirty.qa_image()
                numpy.testing.assert_allclose(
                    qa.data["max"], 1.0668020958044764, atol=1e-7, err_msg=f"{qa}"
                )
                numpy.testing.assert_allclose(
                    qa.data["min"], -0.6730247688717795, atol=1e-7, err_msg=f"{qa}"
                )


if __name__ == "__main__":
    unittest.main()
