"""Unit tests for pipelines expressed via dask.delayed

The variety of Dask-based pipelines are tested here including
continuum imaging pipelines and ICAL pipelines.

Each dask worker can use only one GPU. To prevent hardware race it is required to
use only one thread per worker. It is allowed to use the same GPU by several workers
but in this case one has to care about GPU memory usage.

"""
import logging
import os
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models.data_convert_persist import export_gaintable_to_hdf5
from rascil.data_models.memory_data_models import SkyModel
from rascil.data_models.polarisation_data_models import PolarisationFrame
from rascil.processing_components.calibration.chain_calibration import (
    create_calibration_controls,
)
from rascil.processing_components.calibration.operations import (
    create_gaintable_from_visibility,
    apply_gaintable,
)
from rascil.processing_components.image.gather_scatter import image_gather_channels
from rascil.processing_components.image.operations import (
    smooth_image,
)
from rascil.processing_components.imaging import dft_skycomponent_visibility
from rascil.processing_components.simulation import (
    create_named_configuration,
    decimate_configuration,
)
from rascil.processing_components.simulation import (
    ingest_unittest_visibility,
    create_unittest_model,
    create_unittest_components,
)
from rascil.processing_components.simulation import simulate_gaintable
from rascil.processing_components.skycomponent import (
    insert_skycomponent,
    copy_skycomponent,
)
from rascil.workflows.rsexecute.execution_support.rsexecute import (
    rsexecute,
    get_dask_client,
)
from rascil.workflows.rsexecute.pipelines.pipeline_skymodel_rsexecute import (
    ical_skymodel_list_rsexecute_workflow,
    continuum_imaging_skymodel_list_rsexecute_workflow,
)

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)


class TestPipelineGraphs(unittest.TestCase):
    def setUp(self):

        # We always want the same numbers
        from numpy.random import default_rng

        self.rng = default_rng(1805550721)

        rsexecute.set_client(
            client=get_dask_client(n_workers=4, threads_per_worker=1),
            use_dask=True,
            verbose=True,
        )
        from rascil.processing_components.parameters import rascil_path

        self.results_dir = rascil_path("test_results")
        self.persist = os.getenv("RASCIL_PERSIST", False)

    def tearDown(self):
        rsexecute.close()

    def actualSetUp(
        self, add_errors=False, nfreqwin=5, dopol=False, zerow=False, vnchan=1
    ):
        """Setup the vis, model images, and components for use in the tests

        :param add_errors: Do we want to add errors?
        :param nfreqwin: Number of frequency windows
        :param vnchan: Number of frequencies in each window
        :param dopol: Do polarisation?
        :param zerow: Zero the w coordinates?
        :return:
        """
        self.npixel = 512
        self.low = create_named_configuration("LOW", rmax=750.0)
        self.low = decimate_configuration(self.low, skip=3)
        self.ntimes = 3
        self.times = numpy.linspace(-3.0, +3.0, self.ntimes) * numpy.pi / 12.0

        self.freqwin = nfreqwin
        block_channel_width = (1.2e8 - 0.8e8) / self.freqwin
        channel_width = (1.2e8 - 0.8e8) / (self.freqwin * vnchan)
        self.frequency = numpy.array(
            [
                [
                    0.8e8 + channel_width * vchan + freqwin * block_channel_width
                    for vchan in range(vnchan)
                ]
                for freqwin in range(nfreqwin)
            ]
        )

        self.channelwidth = numpy.array(
            [[channel_width for vchan in range(vnchan)] for freqwin in range(nfreqwin)]
        )

        if dopol:
            self.vis_pol = PolarisationFrame("linear")
            self.image_pol = PolarisationFrame("stokesIQUV")
            f = numpy.array([100.0, 20.0, 0.0, 0.0])
        else:
            self.vis_pol = PolarisationFrame("stokesI")
            self.image_pol = PolarisationFrame("stokesI")
            f = numpy.array([100.0])

        flux = numpy.array(
            [f * numpy.power(freq / 1e8, -0.7) for freq in self.frequency]
        )

        self.phasecentre = SkyCoord(
            ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.bvis_list = [
            rsexecute.execute(ingest_unittest_visibility, nout=1)(
                self.low,
                self.frequency[i],
                self.channelwidth[i],
                self.times,
                self.vis_pol,
                self.phasecentre,
                zerow=zerow,
            )
            for i in range(nfreqwin)
        ]
        self.bvis_list = rsexecute.persist(self.bvis_list)

        self.model_imagelist = [
            rsexecute.execute(create_unittest_model, nout=1)(
                self.bvis_list[i],
                self.image_pol,
                npixel=self.npixel,
                cellsize=0.001,
                nchan=vnchan,
            )
            for i in range(nfreqwin)
        ]
        self.model_imagelist = rsexecute.persist(self.model_imagelist)

        self.components_list = [
            rsexecute.execute(create_unittest_components)(
                self.model_imagelist[freqwin], flux[freqwin][:, numpy.newaxis]
            )
            for freqwin, m in enumerate(self.model_imagelist)
        ]
        self.components_list = rsexecute.persist(self.components_list)

        self.bvis_list = [
            rsexecute.execute(dft_skycomponent_visibility)(
                self.bvis_list[freqwin], self.components_list[freqwin]
            )
            for freqwin, _ in enumerate(self.bvis_list)
        ]
        self.bvis_list = rsexecute.persist(self.bvis_list)

        self.model_imagelist = [
            rsexecute.execute(insert_skycomponent, nout=1)(
                self.model_imagelist[freqwin], self.components_list[freqwin]
            )
            for freqwin in range(nfreqwin)
        ]
        self.model_imagelist = rsexecute.persist(self.model_imagelist)

        if self.persist:
            model = rsexecute.compute(self.model_imagelist[0], sync=True)
            self.cmodel = smooth_image(model)

            model.export_to_fits(
                "%s/test_pipelines_rsexecute_model.fits" % self.results_dir
            )
            self.cmodel.export_to_fits(
                "%s/test_pipelines_rsexecute_cmodel.fits" % self.results_dir,
            )

        if add_errors:
            seeds = [
                self.rng.integers(low=1, high=2**32 - 1) for i in range(nfreqwin)
            ]
            if nfreqwin == 5:
                assert seeds == [
                    3822708302,
                    2154889844,
                    3073218956,
                    3754981936,
                    3778183766,
                ], seeds

            def sim_and_apply(vis, seed):
                gt = create_gaintable_from_visibility(vis, jones_type="G")
                gt = simulate_gaintable(
                    gt,
                    phase_error=0.1,
                    amplitude_error=0.0,
                    smooth_channels=1,
                    leakage=0.0,
                    seed=seed,
                )
                return apply_gaintable(vis, gt)

            # Do this without Dask since the random number generation seems to go wrong
            self.bvis_list = [
                rsexecute.execute(sim_and_apply)(self.bvis_list[i], seeds[i])
                for i in range(self.freqwin)
            ]
            self.bvis_list = rsexecute.compute(self.bvis_list, sync=True)
            self.bvis_list = rsexecute.scatter(self.bvis_list)

        self.model_imagelist = [
            rsexecute.execute(create_unittest_model, nout=1)(
                self.bvis_list[i], self.image_pol, npixel=self.npixel, cellsize=0.001
            )
            for i in range(nfreqwin)
        ]
        self.model_imagelist = rsexecute.persist(self.model_imagelist, sync=True)

    def save_and_check(
        self, tag, clean, residual, restored, flux_max, flux_min, taylor=False
    ):

        if not taylor:
            clean = image_gather_channels(clean)
            residual = image_gather_channels([r[0] for r in residual])
            restored = image_gather_channels(restored)
            if self.persist:
                clean.export_to_fits(
                    f"{self.results_dir}/test_pipelines_{tag}_rsexecute_deconvolved.fits",
                )
                residual.export_to_fits(
                    f"{self.results_dir}/test_pipelines_{tag}_rsexecute_residual.fits",
                )
                restored.export_to_fits(
                    f"{self.results_dir}/test_pipelines_{tag}_rsexecute_restored.fits",
                )
            qa = restored.qa_image()
            assert numpy.abs(qa.data["max"] - flux_max) < 1.0e-5, str(qa)
            assert numpy.abs(qa.data["min"] - flux_min) < 1.0e-5, str(qa)
        else:
            if self.persist:
                for moment, _ in enumerate(clean):
                    clean[moment].export_to_fits(
                        f"{self.results_dir}/test_pipelines_{tag}_rsexecute_deconvolved_taylor{moment}.fits",
                    )
                for moment, _ in enumerate(clean):
                    residual[moment][0].export_to_fits(
                        f"{self.results_dir}/test_pipelines_{tag}_rsexecute_residual_taylor{moment}.fits",
                    )
                for moment, _ in enumerate(clean):
                    restored[moment].export_to_fits(
                        f"{self.results_dir}/test_pipelines_{tag}_rsexecute_restored_taylor{moment}.fits",
                    )
            qa = restored[0].qa_image()
            assert numpy.abs(qa.data["max"] - flux_max) < 1.0e-5, str(qa)
            assert numpy.abs(qa.data["min"] - flux_min) < 1.0e-5, str(qa)

    def test_ical_skymodel_pipeline_empty(self):
        # Run the ICAL pipeline starting with an empty model
        self.actualSetUp(add_errors=True)
        controls = create_calibration_controls()
        controls["T"]["first_selfcal"] = 1
        controls["T"]["timeslice"] = "auto"

        skymodel_list = [
            rsexecute.execute(SkyModel)(image=im) for im in self.model_imagelist
        ]
        skymodel_list = rsexecute.persist(skymodel_list)

        ical_list = ical_skymodel_list_rsexecute_workflow(
            self.bvis_list,
            model_imagelist=self.model_imagelist,
            skymodel_list=skymodel_list,
            context="wg",
            algorithm="mmclean",
            facets=1,
            scales=[0],
            niter=100,
            fractional_threshold=0.1,
            threshold=0.01,
            nmoment=2,
            nmajor=3,
            gain=0.7,
            deconvolve_facets=2,
            deconvolve_overlap=32,
            deconvolve_taper="tukey",
            psf_support=64,
            restore_facets=1,
            calibration_context="T",
            controls=controls,
            do_selfcal=True,
            global_solution=False,
        )
        residual, restored, sky_model_list, gt_list = rsexecute.compute(
            ical_list, sync=True
        )
        clean = [sm.image for sm in sky_model_list]

        for freqwin in range(self.freqwin):
            qa = gt_list[freqwin]["T"].qa_gaintable(
                context=f"Frequency window {freqwin}"
            )
            assert qa.data["residual"] < 2.1e-2, str(qa)

        if self.persist:
            export_gaintable_to_hdf5(
                gt_list[0]["T"],
                "%s/test_pipelines_ical_skymodel_pipeline_rsexecute_gaintable.hdf5"
                % self.results_dir,
            )
        self.save_and_check(
            "ical_skymodel_pipeline_empty",
            clean,
            residual,
            restored,
            116.80924213494968,
            -0.12700338113215193,
        )

    def test_ical_skymodel_pipeline_empty_4chan_T(self):
        # Run the ICAL pipeline starting with an empty model
        self.actualSetUp(add_errors=True, vnchan=4)
        controls = create_calibration_controls()
        controls["T"]["first_selfcal"] = 1
        controls["T"]["timeslice"] = "auto"

        skymodel_list = [
            rsexecute.execute(SkyModel)(image=im) for im in self.model_imagelist
        ]
        skymodel_list = rsexecute.persist(skymodel_list)

        ical_list = ical_skymodel_list_rsexecute_workflow(
            self.bvis_list,
            model_imagelist=self.model_imagelist,
            skymodel_list=skymodel_list,
            context="wg",
            algorithm="mmclean",
            facets=1,
            scales=[0],
            niter=100,
            fractional_threshold=0.1,
            threshold=0.01,
            nmoment=2,
            nmajor=3,
            gain=0.7,
            deconvolve_facets=2,
            deconvolve_overlap=32,
            deconvolve_taper="tukey",
            psf_support=64,
            restore_facets=1,
            calibration_context="T",
            controls=controls,
            do_selfcal=True,
            global_solution=False,
        )
        residual, restored, sky_model_list, gt_list = rsexecute.compute(
            ical_list, sync=True
        )
        clean = [sm.image for sm in sky_model_list]

        for freqwin in range(self.freqwin):
            qa = gt_list[freqwin]["T"].qa_gaintable(
                context=f"Frequency window {freqwin}"
            )
            assert qa.data["residual"] < 4e-2, str(qa)

        if self.persist:
            export_gaintable_to_hdf5(
                gt_list[0]["T"],
                "%s/test_pipelines_ical_skymodel_pipeline_empty_4chan_T_gaintable.hdf5"
                % self.results_dir,
            )
        self.save_and_check(
            "ical_skymodel_pipeline_empty_4chan_T",
            clean,
            residual,
            restored,
            113.81012471759779,
            -0.3991829011868599,
        )

    def test_ical_skymodel_pipeline_empty_4chan_B(self):
        # Run the ICAL pipeline starting with an empty model
        self.actualSetUp(add_errors=True, vnchan=4)
        controls = create_calibration_controls()
        controls["T"]["first_selfcal"] = 10
        controls["T"]["timeslice"] = "auto"
        controls["B"]["first_selfcal"] = 1
        controls["B"]["timeslice"] = "auto"

        skymodel_list = [
            rsexecute.execute(SkyModel)(image=im) for im in self.model_imagelist
        ]
        skymodel_list = rsexecute.persist(skymodel_list)

        ical_list = ical_skymodel_list_rsexecute_workflow(
            self.bvis_list,
            model_imagelist=self.model_imagelist,
            skymodel_list=skymodel_list,
            context="wg",
            algorithm="mmclean",
            facets=1,
            scales=[0],
            niter=100,
            fractional_threshold=0.1,
            threshold=0.01,
            nmoment=2,
            nmajor=3,
            gain=0.7,
            deconvolve_facets=2,
            deconvolve_overlap=32,
            deconvolve_taper="tukey",
            psf_support=64,
            restore_facets=1,
            calibration_context="T",
            controls=controls,
            do_selfcal=True,
            global_solution=False,
        )
        residual, restored, sky_model_list, gt_list = rsexecute.compute(
            ical_list, sync=True
        )
        clean = [sm.image for sm in sky_model_list]

        for freqwin in range(self.freqwin):
            qa = gt_list[freqwin]["T"].qa_gaintable(
                context=f"Frequency window {freqwin}"
            )
            assert qa.data["residual"] < 2.1e-2, str(qa)

        if self.persist:
            export_gaintable_to_hdf5(
                gt_list[0]["T"],
                "%s/test_pipelines_ical_skymodel_pipeline_empty_4chan_B_gaintable.hdf5"
                % self.results_dir,
            )
        self.save_and_check(
            "ical_skymodel_pipeline_empty_4chan_B",
            clean,
            residual,
            restored,
            113.69590541101483,
            -0.5268029813213463,
        )

    def test_ical_skymodel_pipeline_empty_4chan_TB(self):
        # Run the ICAL pipeline starting with an empty model
        self.actualSetUp(add_errors=True, vnchan=4)
        controls = create_calibration_controls()
        controls["T"]["first_selfcal"] = 1
        controls["T"]["phase_only"] = True
        controls["T"]["timeslice"] = "auto"
        controls["B"]["first_selfcal"] = 2
        controls["B"]["phase_only"] = False
        controls["B"]["timeslice"] = "auto"

        skymodel_list = [
            rsexecute.execute(SkyModel)(image=im) for im in self.model_imagelist
        ]
        skymodel_list = rsexecute.persist(skymodel_list)

        ical_list = ical_skymodel_list_rsexecute_workflow(
            self.bvis_list,
            model_imagelist=self.model_imagelist,
            skymodel_list=skymodel_list,
            context="wg",
            algorithm="mmclean",
            facets=1,
            scales=[0],
            niter=100,
            fractional_threshold=0.1,
            threshold=0.01,
            nmoment=2,
            nmajor=5,
            gain=0.7,
            deconvolve_facets=2,
            deconvolve_overlap=32,
            deconvolve_taper="tukey",
            psf_support=64,
            restore_facets=1,
            calibration_context="TB",
            controls=controls,
            do_selfcal=True,
            global_solution=False,
        )
        residual, restored, sky_model_list, gt_list = rsexecute.compute(
            ical_list, sync=True
        )
        clean = [sm.image for sm in sky_model_list]

        for freqwin in range(self.freqwin):
            qa = gt_list[freqwin]["T"].qa_gaintable(
                context=f"Frequency window {freqwin}"
            )
            assert qa.data["residual"] < 3.3e-2, str(qa)
            qa = gt_list[freqwin]["B"].qa_gaintable(
                context=f"Frequency window {freqwin}"
            )
            assert qa.data["residual"] < 5e-3, str(qa)

        if self.persist:
            export_gaintable_to_hdf5(
                gt_list[0]["T"],
                "%s/test_pipelines_ical_skymodel_pipeline_empty_4chan_TB_gaintable.hdf5"
                % self.results_dir,
            )
        self.save_and_check(
            "ical_skymodel_pipeline_empty_4chan_TB",
            clean,
            residual,
            restored,
            113.91148746256246,
            -0.26329612618417786,
        )

    def test_ical_skymodel_pipeline_empty_4chan_TGB(self):
        # Run the ICAL pipeline starting with an empty model
        self.actualSetUp(add_errors=True, vnchan=4)
        controls = create_calibration_controls()
        controls["T"]["first_selfcal"] = 1
        controls["T"]["phase_only"] = True
        controls["T"]["timeslice"] = "auto"
        controls["G"]["first_selfcal"] = 2
        controls["G"]["phase_only"] = False
        controls["G"]["timeslice"] = "auto"
        controls["B"]["first_selfcal"] = 3
        controls["B"]["phase_only"] = False
        controls["B"]["timeslice"] = "auto"

        skymodel_list = [
            rsexecute.execute(SkyModel)(image=im) for im in self.model_imagelist
        ]
        skymodel_list = rsexecute.persist(skymodel_list)

        ical_list = ical_skymodel_list_rsexecute_workflow(
            self.bvis_list,
            model_imagelist=self.model_imagelist,
            skymodel_list=skymodel_list,
            context="wg",
            algorithm="mmclean",
            facets=1,
            scales=[0],
            niter=100,
            fractional_threshold=0.1,
            threshold=0.01,
            nmoment=2,
            nmajor=5,
            gain=0.7,
            deconvolve_facets=2,
            deconvolve_overlap=32,
            deconvolve_taper="tukey",
            psf_support=64,
            restore_facets=1,
            calibration_context="TGB",
            controls=controls,
            do_selfcal=True,
            global_solution=False,
        )
        residual, restored, sky_model_list, gt_list = rsexecute.compute(
            ical_list, sync=True
        )
        clean = [sm.image for sm in sky_model_list]

        for freqwin in range(self.freqwin):
            qa = gt_list[freqwin]["T"].qa_gaintable(
                context=f"Frequency window {freqwin}"
            )
            assert qa.data["residual"] < 3.3e-2, str(qa)
            qa = gt_list[freqwin]["G"].qa_gaintable(
                context=f"Frequency window {freqwin}"
            )
            assert qa.data["residual"] < 3.3e-2, str(qa)
            qa = gt_list[freqwin]["B"].qa_gaintable(
                context=f"Frequency window {freqwin}"
            )
            assert qa.data["residual"] < 5e-3, str(qa)

        if self.persist:
            export_gaintable_to_hdf5(
                gt_list[0]["T"],
                "%s/test_pipelines_ical_skymodel_pipeline_empty_4chan_TGB_gaintable.hdf5"
                % self.results_dir,
            )
        self.save_and_check(
            "ical_skymodel_pipeline_empty_4chan_TGB",
            clean,
            residual,
            restored,
            113.8947579307751,
            -0.3540250885871582,
        )

    def test_ical_skymodel_pipeline_empty_4chan_global(self):
        # Run the ICAL pipeline starting with an empty model
        self.actualSetUp(add_errors=True, vnchan=4)
        controls = create_calibration_controls()
        controls["T"]["first_selfcal"] = 1
        controls["T"]["timeslice"] = "auto"

        skymodel_list = [
            rsexecute.execute(SkyModel)(image=im) for im in self.model_imagelist
        ]
        skymodel_list = rsexecute.persist(skymodel_list)

        ical_list = ical_skymodel_list_rsexecute_workflow(
            self.bvis_list,
            model_imagelist=self.model_imagelist,
            skymodel_list=skymodel_list,
            context="wg",
            algorithm="mmclean",
            facets=1,
            scales=[0],
            niter=100,
            fractional_threshold=0.1,
            threshold=0.01,
            nmoment=2,
            nmajor=3,
            gain=0.7,
            deconvolve_facets=2,
            deconvolve_overlap=32,
            deconvolve_taper="tukey",
            psf_support=64,
            restore_facets=1,
            calibration_context="T",
            controls=controls,
            do_selfcal=True,
            global_solution=True,
        )
        residual, restored, sky_model_list, gt_list = rsexecute.compute(
            ical_list, sync=True
        )
        clean = [sm.image for sm in sky_model_list]

        qa = gt_list[0]["T"].qa_gaintable(context=f"Entire frequency window")
        assert qa.data["residual"] < 3.2e-2, str(qa)

        if self.persist:
            export_gaintable_to_hdf5(
                gt_list[0]["T"],
                "%s/test_pipelines_ical_skymodel_pipeline_empty_4chan_global_gaintable.hdf5"
                % self.results_dir,
            )
        self.save_and_check(
            "ical_skymodel_pipeline_empty_4chan_global",
            clean,
            residual,
            restored,
            113.71719855672642,
            -0.4912817839706385,
        )

    def test_ical_skymodel_pipeline_empty_threshold(self):
        # Run the ICAL pipeline starting with an empty model and a component_threshold to
        # find the brightest sources
        self.actualSetUp(add_errors=True)
        controls = create_calibration_controls()
        controls["T"]["first_selfcal"] = 1
        controls["T"]["timeslice"] = "auto"

        skymodel_list = [
            rsexecute.execute(SkyModel)(image=im) for im in self.model_imagelist
        ]
        skymodel_list = rsexecute.persist(skymodel_list)

        ical_list = ical_skymodel_list_rsexecute_workflow(
            self.bvis_list,
            model_imagelist=self.model_imagelist,
            skymodel_list=skymodel_list,
            context="wg",
            algorithm="mmclean",
            facets=1,
            scales=[0],
            niter=100,
            fractional_threshold=0.1,
            threshold=0.01,
            nmoment=2,
            nmajor=3,
            gain=0.7,
            deconvolve_facets=2,
            deconvolve_overlap=32,
            deconvolve_taper="tukey",
            psf_support=64,
            restore_facets=1,
            calibration_context="T",
            controls=controls,
            do_selfcal=True,
            global_solution=False,
            component_threshold=10.0,
        )
        residual, restored, sky_model_list, gt_list = rsexecute.compute(
            ical_list, sync=True
        )
        clean = [sm.image for sm in sky_model_list]

        if self.persist:
            export_gaintable_to_hdf5(
                gt_list[0]["T"],
                "%s/test_pipelines_ical_skymodel_pipeline_empty_threshold_rsexecute_gaintable_T.hdf5"
                % self.results_dir,
            )
        self.save_and_check(
            "ical_skymodel_pipeline_empty_threshold",
            clean,
            residual,
            restored,
            116.8092421349497,
            -0.12700338113215057,
        )

    def test_ical_skymodel_pipeline_exact(self):
        # Run the ICAL pipeline starting with an exactly correct model

        self.actualSetUp(add_errors=True)
        controls = create_calibration_controls()
        controls["T"]["first_selfcal"] = 0
        controls["T"]["timeslice"] = "auto"

        skymodel_list = [
            rsexecute.execute(SkyModel)(
                components=comp_list, image=self.model_imagelist[icomp]
            )
            for icomp, comp_list in enumerate(self.components_list)
        ]
        skymodel_list = rsexecute.persist(skymodel_list)

        ical_list = ical_skymodel_list_rsexecute_workflow(
            self.bvis_list,
            model_imagelist=self.model_imagelist,
            skymodel_list=skymodel_list,
            context="wg",
            algorithm="mmclean",
            facets=1,
            scales=[0],
            niter=100,
            fractional_threshold=0.1,
            threshold=0.01,
            nmoment=2,
            nmajor=3,
            gain=0.7,
            deconvolve_facets=2,
            deconvolve_overlap=32,
            deconvolve_taper="tukey",
            psf_support=64,
            restore_facets=1,
            calibration_context="T",
            controls=controls,
            do_selfcal=True,
            global_solution=False,
            reset_skymodel=True,
        )
        residual, restored, sky_model_list, gt_list = rsexecute.compute(
            ical_list, sync=True
        )
        clean = [sm.image for sm in sky_model_list]

        if self.persist:
            export_gaintable_to_hdf5(
                gt_list[0]["T"],
                "%s/test_pipelines_ical_skymodel_pipeline_exact_rsexecute_gaintable_T.hdf5"
                % self.results_dir,
            )
        # Check that the residuals are small
        assert numpy.max(gt_list[0]["T"]["residual"].data) < 9e-3

        self.save_and_check(
            "ical_skymodel_pipeline_exact",
            clean,
            residual,
            restored,
            116.83666909212971,
            -0.1272805973991534,
        )

    def test_ical_skymodel_pipeline_exact_dont_reset(self):
        # Run the ICAL pipeline starting with an exactly correct model, keep skymodel after initial calibration

        self.actualSetUp(add_errors=True)
        controls = create_calibration_controls()
        controls["T"]["first_selfcal"] = 0
        controls["T"]["timeslice"] = "auto"

        skymodel_list = [
            rsexecute.execute(SkyModel)(
                components=comp_list, image=self.model_imagelist[icomp]
            )
            for icomp, comp_list in enumerate(self.components_list)
        ]
        skymodel_list = rsexecute.persist(skymodel_list)

        ical_list = ical_skymodel_list_rsexecute_workflow(
            self.bvis_list,
            model_imagelist=self.model_imagelist,
            skymodel_list=skymodel_list,
            context="wg",
            algorithm="mmclean",
            facets=1,
            scales=[0],
            niter=100,
            fractional_threshold=0.1,
            threshold=0.01,
            nmoment=2,
            nmajor=3,
            gain=0.7,
            deconvolve_facets=2,
            deconvolve_overlap=32,
            deconvolve_taper="tukey",
            psf_support=64,
            restore_facets=1,
            calibration_context="T",
            controls=controls,
            do_selfcal=True,
            global_solution=False,
            reset_skymodel=False,
        )
        residual, restored, sky_model_list, gt_list = rsexecute.compute(
            ical_list, sync=True
        )
        clean = [sm.image for sm in sky_model_list]

        if self.persist:
            export_gaintable_to_hdf5(
                gt_list[0]["T"],
                "%s/test_pipelines_ical_skymodel_pipeline_exact_rsexecute_gaintable_T.hdf5"
                % self.results_dir,
            )
        # Check that the residuals are very small
        assert numpy.max(gt_list[0]["T"]["residual"].data) < 1.4e-7

        self.save_and_check(
            "ical_skymodel_pipeline_exact",
            clean,
            residual,
            restored,
            116.90605687884043,
            -2.287644886998181e-06,
        )

    def test_ical_skymodel_pipeline_partial(self):
        # Run the ICAL pipeline starting with a model which is half the true model
        self.actualSetUp(add_errors=True)
        controls = create_calibration_controls()
        controls["T"]["first_selfcal"] = 0
        controls["T"]["timeslice"] = "auto"

        def downscale(comp):
            comp.flux *= 0.5
            return comp

        def downscale_list(cl):
            return [downscale(comp) for comp in cl]

        scaled_components_list = [
            rsexecute.execute(downscale_list)(comp_list)
            for comp_list in self.components_list
        ]
        skymodel_list = [
            rsexecute.execute(SkyModel)(
                components=comp_list, image=self.model_imagelist[icomp]
            )
            for icomp, comp_list in enumerate(scaled_components_list)
        ]
        skymodel_list = rsexecute.persist(skymodel_list)

        ical_list = ical_skymodel_list_rsexecute_workflow(
            self.bvis_list,
            model_imagelist=self.model_imagelist,
            skymodel_list=skymodel_list,
            context="wg",
            algorithm="mmclean",
            facets=1,
            scales=[0],
            niter=100,
            fractional_threshold=0.1,
            threshold=0.01,
            nmoment=2,
            nmajor=3,
            gain=0.7,
            deconvolve_facets=2,
            deconvolve_overlap=32,
            deconvolve_taper="tukey",
            psf_support=64,
            restore_facets=1,
            calibration_context="T",
            controls=controls,
            do_selfcal=True,
            global_solution=False,
            reset_skymodel=False,
        )
        residual, restored, sky_model_list, gt_list = rsexecute.compute(
            ical_list, sync=True
        )
        clean = [sm.image for sm in sky_model_list]

        if self.persist:
            export_gaintable_to_hdf5(
                gt_list[0]["T"],
                "%s/test_pipelines_ical_skymodel_pipeline_partial_rsexecute_gaintable_T.hdf5"
                % self.results_dir,
            )
        self.save_and_check(
            "ical_skymodel_pipeline_partial",
            clean,
            residual,
            restored,
            116.87136533279478,
            -0.06364171175002449,
        )

    def test_continuum_imaging_skymodel_pipeline_empty(self):
        self.actualSetUp()

        continuum_imaging_list = continuum_imaging_skymodel_list_rsexecute_workflow(
            self.bvis_list,
            model_imagelist=self.model_imagelist,
            skymodel_list=None,
            context="wg",
            algorithm="mmclean",
            facets=1,
            scales=[0],
            niter=100,
            fractional_threshold=0.1,
            threshold=0.01,
            nmoment=2,
            nmajor=3,
            gain=0.7,
            deconvolve_facets=2,
            deconvolve_overlap=32,
            deconvolve_taper="tukey",
            psf_support=64,
            restore_facets=1,
        )
        residual, restored, sky_model_list = rsexecute.compute(
            continuum_imaging_list, sync=True
        )
        clean = [sm.image for sm in sky_model_list]

        self.save_and_check(
            "cip_skymodel_pipeline_empty",
            clean,
            residual,
            restored,
            116.83913792559628,
            -0.12738898297535658,
        )

    def test_continuum_imaging_skymodel_pipeline_empty_taylor_terms(self):
        self.actualSetUp()

        continuum_imaging_list = continuum_imaging_skymodel_list_rsexecute_workflow(
            self.bvis_list,
            model_imagelist=self.model_imagelist,
            skymodel_list=None,
            context="wg",
            algorithm="mmclean",
            facets=1,
            scales=[0],
            niter=100,
            fractional_threshold=0.1,
            threshold=0.01,
            nmoment=2,
            nmajor=3,
            gain=0.7,
            deconvolve_facets=2,
            deconvolve_overlap=32,
            deconvolve_taper="tukey",
            psf_support=64,
            restore_facets=1,
            restored_output="taylor",
        )
        residual, restored, sky_model_list = rsexecute.compute(
            continuum_imaging_list, sync=True
        )
        clean = [sm.image for sm in sky_model_list]

        self.save_and_check(
            "cip_skymodel_pipeline_empty_taylor_terms",
            clean,
            residual,
            restored,
            103.75563532871399,
            -0.040258320105007656,
            taylor=True,
        )

    def test_continuum_imaging_skymodel_pipeline_empty_threshold(self):
        self.actualSetUp()

        continuum_imaging_list = continuum_imaging_skymodel_list_rsexecute_workflow(
            self.bvis_list,
            model_imagelist=self.model_imagelist,
            skymodel_list=None,
            context="wg",
            algorithm="mmclean",
            facets=1,
            scales=[0],
            niter=100,
            fractional_threshold=0.1,
            threshold=0.01,
            nmoment=2,
            nmajor=3,
            gain=0.7,
            deconvolve_facets=1,
            deconvolve_overlap=32,
            deconvolve_taper="tukey",
            psf_support=64,
            restore_facets=1,
            component_threshold=10.0,
        )
        residual, restored, sky_model_list = rsexecute.compute(
            continuum_imaging_list, sync=True
        )
        clean = [sm.image for sm in sky_model_list]

        self.save_and_check(
            "cip_skymodel_pipeline_empty_threshold_taylor",
            clean,
            residual,
            restored,
            116.8621697606808,
            -0.12560801720818232,
        )

    def test_continuum_imaging_skymodel_pipeline_partial(self):
        self.actualSetUp()

        def downscale(comp):
            newcomp = copy_skycomponent(comp)
            newcomp.flux *= 0.5
            return newcomp

        def downscale_list(cl):
            return [downscale(comp) for comp in cl]

        scaled_components_list = [
            rsexecute.execute(downscale_list)(comp_list)
            for comp_list in self.components_list
        ]
        skymodel_list = [
            rsexecute.execute(SkyModel)(
                components=comp_list, image=self.model_imagelist[i]
            )
            for i, comp_list in enumerate(scaled_components_list)
        ]
        skymodel_list = rsexecute.persist(skymodel_list)

        continuum_imaging_list = continuum_imaging_skymodel_list_rsexecute_workflow(
            self.bvis_list,
            model_imagelist=self.model_imagelist,
            skymodel_list=skymodel_list,
            context="wg",
            algorithm="mmclean",
            facets=1,
            scales=[0],
            niter=100,
            fractional_threshold=0.1,
            threshold=0.01,
            nmoment=2,
            nmajor=3,
            gain=0.7,
            deconvolve_facets=1,
            deconvolve_overlap=32,
            deconvolve_taper="tukey",
            psf_support=64,
            restore_facets=1,
        )
        residual, restored, sky_model_list = rsexecute.compute(
            continuum_imaging_list, sync=True
        )
        clean = [sm.image for sm in sky_model_list]

        self.save_and_check(
            "cip_skymodel_pipeline_partial",
            clean,
            residual,
            restored,
            116.88411286925424,
            -0.06280400860409169,
        )

    def test_continuum_imaging_skymodel_pipeline_exact(self):
        self.actualSetUp()

        skymodel_list = [
            rsexecute.execute(SkyModel)(
                components=comp_list, image=self.model_imagelist[i]
            )
            for i, comp_list in enumerate(self.components_list)
        ]
        skymodel_list = rsexecute.persist(skymodel_list)

        continuum_imaging_list = continuum_imaging_skymodel_list_rsexecute_workflow(
            self.bvis_list,
            model_imagelist=self.model_imagelist,
            skymodel_list=skymodel_list,
            context="wg",
            algorithm="mmclean",
            facets=1,
            scales=[0],
            niter=100,
            fractional_threshold=0.1,
            threshold=0.01,
            nmoment=2,
            nmajor=3,
            gain=0.7,
            deconvolve_facets=1,
            deconvolve_overlap=32,
            deconvolve_taper="tukey",
            psf_support=64,
            restore_facets=1,
        )
        residual, restored, sky_model_list = rsexecute.compute(
            continuum_imaging_list, sync=True
        )
        clean = [sm.image for sm in sky_model_list]

        self.save_and_check(
            "cip_skymodel_pipeline_exact",
            clean,
            residual,
            restored,
            116.90605597782766,
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
