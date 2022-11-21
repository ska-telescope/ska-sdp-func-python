## Description

Files in this directory are used for testing the code. Not meant to be used for any other purpose.

## Generating these files:

### 1. low_test_skymodel_from_gream.hdf

The file was generated with rascil_main commit #71ca45e21c6f04dc558bf14bd847c5cf0a4d2f4c

Use rascil.processing_components.simulation.testing_support.create_low_test_skymodel_from_gleam

    from ska_sdp_datamodels.science_data_model import PolarisationFrame
    skymodel = create_low_test_skymodel_from_gleam(
        npixel=512,
        cellsize=0.001,
        frequency=numpy.array([1.0e8]),
        radius=(512 * 0.001 / 2.0),
        phasecentre=phase_centre,
        polarisation_frame=PolarisationFrame("stokesI"),
        flux_limit=0.3,
        flux_threshold=1.0,
        flux_max=5.0,
    )

where phase_centre:

    from astropy import units
    from astropy.coordinates import SkyCoord    
    phase_centre = SkyCoord(
            ra=+30.0 * units.deg, dec=-60.0 * units.deg, frame="icrs", equinox="J2000"
        )

Then export it with `ska_sdp_datamodels.sky_model.export_skymodel_to_hdf5`