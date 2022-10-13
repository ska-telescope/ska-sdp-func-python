"""Unit tests for testing support

"""

import logging
import unittest

from rascil.processing_components.image.gradients import image_gradients
from rascil.processing_components.image.operations import (
    show_image,
    import_image_from_fits,
)
from rascil.processing_components.parameters import rascil_data_path

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)


class TestGradients(unittest.TestCase):
    def setUp(self):
        from rascil.processing_components.parameters import (
            rascil_path,
        )

        self.results_dir = rascil_path("test_results")

        self.show = False
        self.persist = False

    def test_create_gradient(self):
        real_vp = import_image_from_fits(
            rascil_data_path("models/MID_FEKO_VP_B2_45_1360_real.fits")
        )
        gradx, grady = image_gradients(real_vp)

        gradxx, gradxy = image_gradients(gradx)
        gradyx, gradyy = image_gradients(grady)

        gradx["pixels"].data *= real_vp["pixels"].data
        grady["pixels"].data *= real_vp["pixels"].data
        gradxx["pixels"].data *= real_vp["pixels"].data
        gradxy["pixels"].data *= real_vp["pixels"].data
        gradyx["pixels"].data *= real_vp["pixels"].data
        gradyy["pixels"].data *= real_vp["pixels"].data

        if self.show:
            import matplotlib.pyplot as plt

            plt.clf()
            show_image(gradx, title="gradx")
            plt.show(block=False)
            plt.clf()
            show_image(grady, title="grady")
            plt.show(block=False)
        if self.persist:
            gradx.export_to_fits(
                "%s/test_image_gradients_gradx.fits" % (self.results_dir)
            )
            grady.export_to_fits(
                "%s/test_image_gradients_grady.fits" % (self.results_dir)
            )

        if self.show:
            import matplotlib.pyplot as plt

            plt.clf()
            show_image(gradxx, title="gradxx")
            plt.show(block=False)
            plt.clf()
            show_image(gradxy, title="gradxy")
            plt.show(block=False)
            plt.clf()
            show_image(gradyx, title="gradyx")
            plt.show(block=False)
            plt.clf()
            show_image(gradyy, title="gradyy")
            plt.show(block=False)
        if self.persist:
            gradxx.export_to_fits(
                "%s/test_image_gradients_gradxx.fits" % (self.results_dir)
            )
            gradxy.export_to_fits(
                "%s/test_image_gradients_gradxy.fits" % (self.results_dir)
            )
            gradyx.export_to_fits(
                "%s/test_image_gradients_gradyx.fits" % (self.results_dir)
            )
            gradyy.export_to_fits(
                "%s/test_image_gradients_gradyy.fits" % (self.results_dir)
            )


if __name__ == "__main__":
    unittest.main()
