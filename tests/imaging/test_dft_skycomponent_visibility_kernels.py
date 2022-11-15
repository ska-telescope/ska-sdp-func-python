"""
Unit tests for DFT kernels
"""
import numpy
import pytest

from ska_sdp_func_python.imaging.dft import dft_skycomponent_visibility


@pytest.mark.parametrize(
    "compute_kernel", ["cpu_looped", "gpu_cupy_raw", "proc_func"]
)
def test_dft_stokesiquv_visibility(compute_kernel, vis, comp):
    """
    The various DFT kernels return the same results
    """
    if compute_kernel == "gpu_cupy_raw":
        try:
            import cupy  # noqa: F401
        except ModuleNotFoundError:
            return

    new_vis = vis.copy(deep=True)
    result = dft_skycomponent_visibility(
        new_vis,
        comp,
        dft_compute_kernel=compute_kernel,
    )
    qa = result.visibility_acc.qa_visibility()
    numpy.testing.assert_almost_equal(qa.data["maxabs"], 2400.0)
    numpy.testing.assert_almost_equal(qa.data["minabs"], 200.9975124)
    numpy.testing.assert_almost_equal(qa.data["rms"], 942.9223125)

    numpy.testing.assert_almost_equal(
        result.vis.data.sum(), 16396270.264258286 - 219031.4899462515j
    )
