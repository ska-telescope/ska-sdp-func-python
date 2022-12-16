"""
Unit tests for convolutional Gridding
"""
import numpy
import pytest

from ska_sdp_func_python.fourier_transforms.fft_coordinates import (
    coordinateBounds,
    coordinates,
    coordinates2,
    coordinates2Offset,
    w_beam,
)


@pytest.mark.parametrize("N", [4, 5, 6, 7, 8, 9, 1000, 1001, 1002, 1003])
def test_coordinates(N):
    """Unit tests for the coordinates function"""
    low, high = coordinateBounds(N)
    c = coordinates(N)
    assert numpy.min(c) == low
    assert numpy.max(c) == high
    assert c[N // 2] == 0


@pytest.mark.parametrize("N", [4, 5, 6, 7, 8, 9, 1000, 1001, 1002, 1003])
def test_coordinates2(N):
    """Unit tests for the coordinates2 function"""
    low, high = coordinateBounds(N)
    cx, cy = coordinates2(N)
    assert numpy.min(cx) == low
    assert numpy.max(cx) == high
    assert numpy.min(cy) == low
    assert numpy.max(cy) == high
    assert (cx[N // 2, :] == 0).all()
    assert (cy[:, N // 2] == 0).all()


@pytest.mark.parametrize("N", [4, 5, 6, 7, 8, 9, 1000, 1001, 1002, 1003])
def test_coordinates2offset(N):
    """Unit tests for the coordinates2_offset function"""
    low, high = coordinateBounds(N)
    cx_off, cy_off = coordinates2Offset(N, None, None)
    assert numpy.min(cx_off) == low
    assert numpy.max(cx_off) == high
    assert numpy.min(cy_off) == low
    assert numpy.max(cy_off) == high
    assert (cx_off[N // 2, :] == 0).all()
    assert (cy_off[:, N // 2] == 0).all()


def test_w_kernel_beam():
    """Unit tests for the w_beam function"""
    assert (numpy.real(w_beam(5, 0.1, 0))[0, 0] == 1.0).all()
    assert (w_beam(5, 0.1, 100)[2, 2] == 1).all()
    assert (w_beam(10, 0.1, 100)[5, 5] == 1).all()
    assert (w_beam(11, 0.1, 1000)[5, 5] == 1).all()
