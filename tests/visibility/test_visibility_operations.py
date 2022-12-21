"""
Unit tests for visibility operations
"""
import numpy
from ska_sdp_datamodels.visibility.vis_create import create_visibility

from ska_sdp_func_python.visibility.operations import (
    concatenate_visibility,
    divide_visibility,
    subtract_visibility,
)


def test_concatenate_visibility(visibility):
    """Unit test for the concatenate_visibility function"""
    new_times = (numpy.pi / 43200.0) * numpy.arange(300.0, 600.0, 30.0)
    new_vis = create_visibility(
        visibility.configuration,
        new_times,
        visibility.frequency.data,
        channel_bandwidth=visibility.channel_bandwidth.data,
        phasecentre=visibility.phasecentre,
        weight=1.0,
    )
    new_ntimes = new_vis.vis.shape[0]
    original_ntimes = visibility.vis.shape[0]
    result = concatenate_visibility([visibility, new_vis], dim="time")
    result_ntimes = result.vis.shape[0]
    assert result_ntimes == new_ntimes + original_ntimes

    reverse_result = concatenate_visibility([new_vis, visibility], dim="time")
    assert reverse_result.time.all() == result.time.all()


def test_divide_visibility(visibility):
    """Unit test for the divide_visibility function with linear polarisation"""
    vis = visibility.copy(deep=True)
    vis["vis"].data[..., :] = [2.0 + 0.0j, 0.0j, 0.0j, 2.0 + 0.0j]
    new_vis = visibility.copy(deep=True)
    new_vis["vis"].data[..., :] = [1.0 + 0.0j, 0.0j, 0.0j, 1.0 + 0.0j]

    result = divide_visibility(vis, new_vis)
    assert result.visibility_acc.nvis == vis.visibility_acc.nvis
    assert numpy.max(numpy.abs(result.vis)) == 2.0, numpy.max(
        numpy.abs(result.vis)
    )


def test_divide_visibility_singular(visibility):
    """Unit test for the divide_visibility function with linear polarisation"""
    vis = visibility.copy(deep=True)
    vis["vis"].data[..., :] = [
        2.0 + 0.0j,
        2.0 + 0.0j,
        2.0 + 0.0j,
        2.0 + 0.0j,
    ]
    new_vis = visibility.copy(deep=True)
    new_vis["vis"].data[..., :] = [
        1.0 + 0.0j,
        1.0 + 0.0j,
        1.0 + 0.0j,
        1.0 + 0.0j,
    ]

    result = divide_visibility(vis, new_vis)
    assert result.visibility_acc.nvis == vis.visibility_acc.nvis
    assert numpy.max(numpy.abs(result.vis)) == 2.0, numpy.max(
        numpy.abs(result.vis)
    )


def test_subtract(visibility):
    """Unit test for the subtract_visibility function"""
    vis1 = visibility.copy(deep=True)
    vis1["vis"].data[...] = 10.0
    vis2 = visibility.copy(deep=True)
    vis2["vis"].data[...] = 1.0

    result = subtract_visibility(vis1, vis2)
    qa = result.visibility_acc.qa_visibility(context="test_qa")
    assert qa.data["maxabs"] == 9.0
