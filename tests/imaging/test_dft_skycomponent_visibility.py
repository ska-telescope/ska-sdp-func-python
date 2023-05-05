"""
Unit tests for DFT-related functions
"""
import astropy.units as u
import numpy
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)
from ska_sdp_datamodels.sky_model.sky_model import SkyComponent
from ska_sdp_datamodels.visibility import create_visibility

pytest.importorskip(
    modname="ska_sdp_func", reason="ska-sdp-func is an optional dependency"
)
from ska_sdp_func_python.imaging.dft import (
    dft_skycomponent_visibility,
    extract_direction_and_flux,
    idft_visibility_skycomponent,
)
from ska_sdp_func_python.visibility.base import phaserotate_visibility

# length: nchan of visibility fixture
ONED_FLUX = [
    100.0,
    0.9 * 100.0,
    0.8 * 100.0,
    0.7 * 100.0,
    0.6 * 100.0,
    0.5 * 100.0,
]
# times for visibility fixture on HA
VIS_TIMES_HA = (numpy.pi / 43200.0) * numpy.linspace(0.0, 300.0, 2)


def _generate_sky_component(pol_frame, vis, comp_dir):
    # Define the component and give it some spectral behaviour
    flux = numpy.array([ONED_FLUX] * len(pol_frame.names)).transpose()

    phase_centre = vis.attrs["phasecentre"]
    phase_centre_offset = phase_centre.skyoffset_frame()

    comp_rel_direction = comp_dir.transform_to(phase_centre_offset)
    comp = SkyComponent(
        direction=comp_rel_direction,
        frequency=vis.frequency.data,
        flux=flux,
        polarisation_frame=pol_frame,
    )
    return comp


@pytest.fixture(scope="module", name="component")
def component_fixt(visibility, comp_direction):
    """
    SkyComponent for visibility fixture
    """
    comp = _generate_sky_component(
        PolarisationFrame("linear"),  # visibility is linear
        visibility,
        comp_direction,
    )

    return comp


@pytest.mark.parametrize(
    "polarisation_frame",
    [
        PolarisationFrame("linear"),
        PolarisationFrame("stokesI"),
        PolarisationFrame("stokesIQUV"),
    ],
)
def test_phaserotate_visibility(
    polarisation_frame, visibility, comp_direction
):
    """
    Test phase-rotating visibility with different polarisation frames
    """
    phase_centre = visibility.attrs["phasecentre"]

    comp = _generate_sky_component(
        polarisation_frame, visibility, comp_direction
    )

    vis = create_visibility(
        visibility.configuration,
        VIS_TIMES_HA,
        visibility.frequency.data,
        channel_bandwidth=visibility.channel_bandwidth.data,
        phasecentre=phase_centre,
        weight=1.0,
        polarisation_frame=polarisation_frame,
    )
    vis_model = dft_skycomponent_visibility(vis, comp)

    # Predict visibilities with new phase centre independently
    ha_diff = -(comp_direction.ra - phase_centre.ra).to(u.rad).value
    vis_new_phase = create_visibility(
        vis.configuration,
        VIS_TIMES_HA + ha_diff,
        vis.frequency.data,
        channel_bandwidth=vis.channel_bandwidth.data,
        phasecentre=comp_direction,
        weight=1.0,
        polarisation_frame=polarisation_frame,
    )
    vis_new_phase_model = dft_skycomponent_visibility(vis_new_phase, comp)

    # Should yield the same results as rotation
    # Tested function
    rotated_vis = phaserotate_visibility(
        vis_model, newphasecentre=comp_direction, tangent=False
    )
    assert_allclose(rotated_vis.vis, vis_new_phase_model.vis, rtol=3e-6)
    assert_allclose(rotated_vis.uvw, vis_new_phase_model.uvw, rtol=3e-6)


@pytest.mark.parametrize(
    "polarisation_frame",
    [
        PolarisationFrame("linear"),
        PolarisationFrame("circular"),
        PolarisationFrame("stokesIQUV"),
    ],
)
def test_idft_visibility_skycomponent(
    polarisation_frame, visibility, comp_direction
):
    """
    Test iDFT returns component
    """
    phase_centre = visibility.attrs["phasecentre"]

    comp = _generate_sky_component(
        polarisation_frame, visibility, comp_direction
    )

    vis = create_visibility(
        visibility.configuration,
        VIS_TIMES_HA,
        visibility.frequency.data,
        channel_bandwidth=visibility.channel_bandwidth.data,
        phasecentre=phase_centre,
        weight=1.0,
        polarisation_frame=polarisation_frame,
    )
    # run DFT on visibility first
    vis_model = dft_skycomponent_visibility(vis, comp)

    # then run iDFT to get the component back
    result_component, _ = idft_visibility_skycomponent(vis_model, comp)
    assert_allclose(
        comp.flux, numpy.real(result_component[0].flux), rtol=1e-10
    )


def test_extract_direction_and_flux(visibility, component):
    """
    vis and comp frequency and polarisation are the same
    --> expected flux is same as comp flux (except complex)
    """
    expected_direction = numpy.array(
        [[1.42961744e-02, -7.15598688e-05, -1.02198084e-04]]
    )
    result_direction, result_flux = extract_direction_and_flux(
        component, visibility
    )

    assert_array_almost_equal(result_direction, expected_direction)
    assert (result_flux == component.flux.astype(complex)).all()


def test_extract_direction_and_flux_diff_pol(visibility, component):
    """
    vis and comp frequency match, but polarisation frame is
    different (vis = stokesI, comp = linear).
    Expected flux contains the data for the polarisation of visibility.
    """
    vis = create_visibility(
        visibility.configuration,
        VIS_TIMES_HA,
        visibility.frequency.data,
        channel_bandwidth=visibility.channel_bandwidth.data,
        phasecentre=visibility.attrs["phasecentre"],
        weight=1.0,
        polarisation_frame=PolarisationFrame("stokesI"),
    )

    expected_direction = numpy.array(
        [[1.42961744e-02, -7.15598688e-05, -1.02198084e-04]]
    )
    flux = numpy.array([ONED_FLUX] * 4).transpose()
    expected_flux = flux[:, 0].astype(complex).reshape((flux.shape[0], 1))
    result_direction, result_flux = extract_direction_and_flux(component, vis)

    assert_array_almost_equal(result_direction, expected_direction)
    assert (result_flux == expected_flux).all()
