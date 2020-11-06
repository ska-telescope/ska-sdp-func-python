.. _Fourier_processing:

Fourier processing
******************

For wide field imaging with w term correction, the Nifty Gridder is supported and has the best performance.
It is installed via pip as part of the RASCIL install process.See more information on nifty-gridder at:

    https://gitlab.mpcdf.mpg.de/ift/nifty_gridder

Nifty Gridder is supported at the processing component level via:

 * :py:func:`rascil.processing_components.imaging.ng.invert_ng`
 * :py:func:`rascil.processing_components.imaging.ng.predict_ng`

At the workflow level, such as imaging and pipeline workflows, use context='ng'.

RASCIL model
------------

If only wterm needs to be corrected, the Nifty Gridder is the best option. However RASCIL has another approach.
There are many algorithms for imaging, using different approaches to correct for various effects:

+ Simple 2D transforms
+ AW projection
+ MFS variants

Since the scale of SKA is so much larger than previous telescopes, it is not clear which scaling strategies and
algorithms are going to offer the best performance. For this reason, it is important the synthesis framework not be
restrictive.

All the above functions are linear in the visibilities and image. The 2D transform is correct for sufficiently
restricted context. Hence we layer all algorithms on top of the 2D transform. This means that a suitable
framework decomposes the overall transform into suitable linear combinations of invocations of 2D transforms.

The full layering is:

+ The core imaging functions are defined in :py:mod:`processing_components.imaging.base`. Nifty Gridder provides
optimised wstack/wprojection imaging and should be the default. AW projection is also possible using the
2d gridder and a suitable gridding convolution function.

+ Other algorithms (e.g. timeslice and wstack) are implemented as workflows using the core 2d imaging functions. These are defined in :py:mod:`workflows.serial.imaging.imaging_serial` and :py:mod:`workflows.rsexecute.imaging.imaging_rsexecute`

The style of first approach is::

        m31model=create_test_image()
        for ipatch in image_raster_iter(m31model, facets=4):
            # each image patch can be used to add to the visibility data
            vis.data['vis'] += predict_2d(vis, ipatch).data['vis'].data

This relies upon the data objects (model and vis) possessing sufficient meta data to enable operations such as phase
rotation from one frame to another.

The second approach is based on the same underlying functions, predict_2d and invert_2d but encapsulates the looping
of the above example::

        m31model=create_test_image()
        vis = predict_list_serial_workflow(vis, m31model, context='facets', nfacets=4)

The third approach implements imaging via rsexecute::

        m31model_component = rsexecute.execute(create_test_image)()
        vis_component = rsexecute(vis_scatter_time)(vis, timeslice='auto')
        vis_component = predict_component(vis_component, m31component_component, facets=4)

This form may be executed immediately::

        rsexecute.set_client(use_dask=False)
        m31model_component = rsexecute.execute(create_test_image)()
        vis_component = rsexecute(vis_scatter_time)(vis, timeslice='auto')
        vis_component = predict_component(vis_component, m31component_component, facets=4)

Or delayed::

        rsexecute.set_client(use_dask=True)
        m31model_component = rsexecute.execute(create_test_image)()
        vis_component = rsexecute(vis_scatter_time)(vis, timeslice='auto')
        vis_component = predict_component(vis_component, m31component_component, facets=4)
        vis_component = rsexecute.compute(vis_component, sync=True)

