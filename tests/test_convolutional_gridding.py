""" Unit tests for convolutional Gridding


"""
import numpy
import itertools
import random
import unittest

from numpy.testing import assert_allclose

from arl.fourier_transforms.convolutional_gridding import w_kernel, w_beam, kernel_oversample, coordinates, \
    coordinates2, coordinateBounds, anti_aliasing_calculate, anti_aliasing_transform, \
    convolutional_degrid, convolutional_grid


class TestConvolutionalGridding(unittest.TestCase):
    
    @staticmethod
    def assertAlmostEqualScalar(a, result=1.0):
        w = result * numpy.ones_like(result)

    def test_coordinates(self):
        for N in [4,5,6,7,8,9,1000,1001,1002,1003]:
            low, high = coordinateBounds(N)
            c = coordinates(N)
            cx, cy = coordinates2(N)
            self.assertAlmostEqual(numpy.min(c), low)
            self.assertAlmostEqual(numpy.max(c), high)
            self.assertAlmostEqual(numpy.min(cx), low)
            self.assertAlmostEqual(numpy.max(cx), high)
            self.assertAlmostEqual(numpy.min(cy), low)
            self.assertAlmostEqual(numpy.max(cy), high)
            assert c[N//2] == 0
            assert (cx[N//2,:] == 0).all()
            assert (cy[:,N//2] == 0).all()

    @staticmethod
    def _test_pattern(npixel):
        return coordinates2(npixel)[0] + coordinates2(npixel)[1] * 1j

    
    def test_anti_aliasing_transform(self):
        for shape in [(64, 64), (128, 128), (256, 256)]:
            _, aaf = anti_aliasing_transform(shape, 8)
            self.assertAlmostEqual(numpy.max(aaf[..., aaf.shape[1] // 2, aaf.shape[0] // 2]), 1.0)


    def test_anti_aliasing_calculate(self):
        for shape in [(64, 64), (128, 128), (256, 256)]:
            _, aaf = anti_aliasing_calculate(shape, 8)
            self.assertAlmostEqual(numpy.max(aaf[..., aaf.shape[1] // 2, aaf.shape[0] // 2]), 1.0)

    def test_w_kernel_function(self):
        assert_allclose(numpy.real(w_beam(5, 0.1, 0))[0,0], 1.0)
        self.assertAlmostEqualScalar(w_beam(5, 0.1, 100)[2, 2], 1)
        self.assertAlmostEqualScalar(w_beam(10, 0.1, 100)[5, 5], 1)
        self.assertAlmostEqualScalar(w_beam(11, 0.1, 1000)[5, 5], 1)
    
    def test_kernel_oversampled_subgrid(self):
        # Oversampling should produce the same values where sub-grids overlap
        for npixel in range(4, 30, 2):
            pat = self._test_pattern(npixel)
            kern = kernel_oversample(pat, npixel, 1, npixel - 2)
            kern2 = kernel_oversample(pat, npixel, 2, npixel - 2)
            assert_allclose(kern[0, 0], kern2[0, 0], atol=1e-15)
            kern3 = kernel_oversample(pat, npixel, 3, npixel - 2)
            assert_allclose(kern[0, 0], kern3[0, 0], atol=1e-15)
            kern4 = kernel_oversample(pat, npixel, 4, npixel - 2)
            for ux, uy in itertools.product(range(2), range(2)):
                assert_allclose(kern2[uy, ux], kern4[2 * uy, 2 * ux], atol=1e-15)
            kern8 = kernel_oversample(pat, npixel, 8, npixel - 2)
            for ux, uy in itertools.product(range(3), range(3)):
                assert_allclose(kern4[uy, ux], kern8[2 * uy, 2 * ux], atol=1e-15)
    
    def test_kernel_scale(self):
        # Scaling the grid should not make a difference
        npixel = 10
        wff = numpy.zeros((npixel, npixel))
        wff[npixel // 2, npixel // 2] = 1  # Not the most interesting kernel...
        k = kernel_oversample(wff, npixel, 1, npixel)
        k2 = kernel_oversample(4 * wff, npixel * 2, 1, npixel)
        assert_allclose(k, k2)
    
    def test_w_kernel_normalisation(self):
        # Test w-kernel normalisation.
        for kernel_oversampling in [4, 5, 6]:
            for npixel in [16, 20, 24, 32, 64]:
                k = kernel_oversample(w_beam(npixel + 2, 0.1, npixel * 10), npixel + 2,
                                      kernel_oversampling, npixel)
                assert_allclose(numpy.sum(k), kernel_oversampling ** 2, rtol=0.07)

    def test_convolutional_grid(self):
        npixel = 256
        nvis = 1000
        nchan = 1
        npol = 4
        uvgrid = numpy.zeros([nchan, npol, npixel, npixel], dtype='complex')
        gcf, kernel = anti_aliasing_transform((npixel, npixel), 8)
        # kernel has shape [kernel_oversampling, kernel_oversampling, npixel, npixel] The fractional
        # part of the coordinate maps onto the first two axes.
        # Make some uv coordinates with random locations
        uvcoords = numpy.array([[random.uniform(-0.25, 0.25), random.uniform(-0.25, 0.25)] for ivis in range(nvis)])
        # Make some visibilities, all complex unity
        vis = numpy.ones([nvis, nchan, npol], dtype='complex')
        visweights = numpy.ones([nvis, nchan, npol])
        kernels = (numpy.zeros([nvis], dtype='int'), [kernel])
        frequencymap=numpy.zeros([nvis], dtype='int')
        # On my laptop the following takes about 14 seconds for 4M points so about 3.5 us per sample
        uvgrid, sumwt=convolutional_grid(kernels, uvgrid, vis, visweights, uvcoords, frequencymap)
        assert numpy.sum(sumwt) > 0.0
        assert uvgrid.shape[0] == nchan
        assert uvgrid.shape[1] == npol
        assert uvgrid.shape[2] == npixel
        assert uvgrid.shape[3] == npixel

    def test_convolutional_degrid(self):
        npixel = 256
        nvis = 1000
        nchan = 1
        npol = 4
        uvgrid = numpy.ones([nchan, npol, npixel, npixel], dtype='complex')
        gcf, kernel = anti_aliasing_transform((npixel, npixel), 8)
        # kernel has shape [kernel_oversampling, kernel_oversampling, npixel, npixel] The fractional
        # part of the coordinate maps onto the first two axes.
        # Make some uv coordinates with random locations
        uvcoords = numpy.array([[random.uniform(-0.25, 0.25), random.uniform(-0.25, 0.25)] for ivis in range(nvis)])
        vshape=[nvis, npol]
        kernels = (numpy.zeros([nvis], dtype='int'), [kernel])
        frequencymap=numpy.zeros([nvis], dtype='int')

        vis = convolutional_degrid(kernels, vshape, uvgrid, uvcoords, frequencymap)
        assert vis.shape[0] == nvis
        assert vis.shape[1] == npol


if __name__ == '__main__':
    unittest.main()
