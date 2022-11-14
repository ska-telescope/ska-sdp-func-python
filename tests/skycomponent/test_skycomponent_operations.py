# pylint: disable= unused-variable
""" Unit tests for skycomponent operations

"""
import logging

import astropy.units as u
import numpy
import pytest
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.image.image_create import create_image
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)
from ska_sdp_datamodels.sky_model.sky_model import SkyComponent

from ska_sdp_func_python.skycomponent.operations import (
    filter_skycomponents_by_flux,
    find_nearest_skycomponent,
    find_nearest_skycomponent_index,
    find_separation_skycomponents,
    find_skycomponent_matches,
    find_skycomponent_matches_atomic,
    fit_skycomponent_spectral_index,
    insert_skycomponent,
    partition_skycomponent_neighbours,
    remove_neighbouring_components,
    restore_skycomponent,
    select_components_by_separation,
    select_neighbouring_components,
    voronoi_decomposition,
)

log = logging.getLogger("func-python-logger")

log.setLevel(logging.WARNING)


@pytest.fixture(scope="module", name="result_operations")
def operations_fixture():
    """Fixture for operations.py unit tests"""
    home_coords = SkyCoord(
        ra=+10.0 * u.deg, dec=-40.0 * u.deg, frame="icrs", equinox="J2000"
    )
    phase_centre = SkyCoord(
        ra=+180.0 * u.deg, dec=-40.0 * u.deg, frame="icrs", equinox="J2000"
    )
    frequency = numpy.array([1e8])
    name = "test_sc"
    flux = numpy.ones((1, 1))
    shape = "Point"
    polarisation_frame = PolarisationFrame("stokesI")

    sky_component1 = SkyComponent(
        phase_centre,
        frequency,
        name,
        flux,
        shape,
        polarisation_frame,
    )

    sky_component2 = SkyComponent(
        home_coords,
        frequency,
        name,
        flux=numpy.array([[10]]),
        shape=shape,
        polarisation_frame=polarisation_frame,
    )

    ref_sky_component1 = SkyComponent(
        home_coords,
        frequency,
        name,
        flux,
        shape,
        polarisation_frame,
    )

    ref_sky_component2 = SkyComponent(
        SkyCoord(
            ra=+100.0 * u.deg, dec=-40.0 * u.deg, frame="icrs", equinox="J2000"
        ),
        frequency,
        name,
        flux,
        shape,
        polarisation_frame,
    )
    params = {
        "home": home_coords,
        "ref_skycomponents_list": [
            ref_sky_component1,
            ref_sky_component2,
        ],
        "skycomponents_list": [
            sky_component1,
            sky_component2,
        ],
    }
    return params


def test_find_nearest_skycomponent_index(result_operations):
    """Check the index is 1"""
    home = result_operations["home"]
    components = result_operations["skycomponents_list"]

    index = find_nearest_skycomponent_index(home, components)

    assert index == 1


def test_find_nearest_skycomponent(result_operations):
    """Check the matching component is at index 1 and the seperation is 0"""
    home = result_operations["home"]
    components = result_operations["skycomponents_list"]

    best_sc, best_sc_seperation = find_nearest_skycomponent(home, components)

    assert best_sc == components[1]
    assert best_sc_seperation == 0


def test_find_separation_skycomponents(result_operations):
    """Check the seperation"""
    components = result_operations["skycomponents_list"]
    seperations = find_separation_skycomponents(components, components)
    expected_seperations = numpy.array([[0, 1.73628363], [1.73628363, 0]])

    assert seperations == pytest.approx(expected_seperations, 1e-7)


def test_find_skycomponent_matches_atomic(result_operations):
    """Check the matches"""
    components = result_operations["skycomponents_list"]
    ref_components = result_operations["ref_skycomponents_list"]
    matches = find_skycomponent_matches_atomic(components, ref_components)

    assert len(matches) == 1


def test_find_skycomponent_matches(result_operations):
    """Check matches"""
    components = result_operations["skycomponents_list"]
    ref_components = result_operations["ref_skycomponents_list"]
    matches = find_skycomponent_matches(components, ref_components)

    assert len(matches) == 1


def test_select_components_by_separation(result_operations):
    """Check correct component is selected"""
    home = result_operations["home"]
    ref_components = result_operations["ref_skycomponents_list"]
    selected = select_components_by_separation(home, ref_components, rmax=1)

    assert selected == [ref_components[0]]


def test_select_neighbouring_components(result_operations):
    """Check correct component is selected"""
    components = result_operations["skycomponents_list"]
    target = result_operations["ref_skycomponents_list"]
    target_index, target_sep = select_neighbouring_components(
        components, target
    )

    assert (target_index == [1, 0]).all()
    assert (target_sep.deg == [pytest.approx(58.99740846, 1e-7), 0.0]).all()


def test_remove_neighbouring_components(result_operations):
    """Check correct component is compressed (only index 1 should remain)"""
    components = result_operations["skycomponents_list"]
    distance = 1  # in radians

    compressed_targets = remove_neighbouring_components(components, distance)

    assert compressed_targets[0] == [0, 1]


def test_filter_skycomponents_by_flux(result_operations):
    """Check a skycomponent is filtered out"""

    components = result_operations["skycomponents_list"]
    new_components = filter_skycomponents_by_flux(components, flux_min=5)

    assert len(components) == 2
    assert len(new_components) == 1


def test_insert_skycomponent(result_operations):
    """Check a skycomponent is inserted to the image"""
    image = create_image(
        npixel=512, cellsize=0.0001, phasecentre=result_operations["home"]
    )
    component = result_operations["skycomponents_list"][0]

    new_image = insert_skycomponent(image, component)

    assert new_image != image


def test_restore_skycomponent(result_operations):
    """Check a skycomponent is restored to the image"""
    image = create_image(
        npixel=512, cellsize=0.0001, phasecentre=result_operations["home"]
    )
    component = result_operations["skycomponents_list"][0]
    clean_beam = {"bmaj": 0.1, "bmin": 0.05, "bpa": -60.0}
    new_image = restore_skycomponent(image, component, clean_beam=clean_beam)

    assert new_image != image


@pytest.mark.skip(
    reason="Better understanding of Vornoi needed to make a useful unit test"
)
def test_voronoi_decomposition(result_operations):
    """Check Vornoi decompostion"""
    image = create_image(
        npixel=512, cellsize=0.0001, phasecentre=result_operations["home"]
    )
    image_component = result_operations["ref_skycomponents_list"][1]
    # Get an image that isn't empty (insert skycomonent)
    image = insert_skycomponent(image, image_component)
    components = result_operations["ref_skycomponents_list"]
    v_structure, v_image = voronoi_decomposition(image, components)


@pytest.mark.skip(
    reason="Need better understanding of function to build meaningful test"
)
def test_partition_skycomponent_neighbours(result_operations):
    """Check skycompoentns are partitioned correctly"""

    components = result_operations["ref_skycomponents_list"]
    target = [result_operations["ref_skycomponents_list"][0]]

    partitioned_comps = partition_skycomponent_neighbours(components, target)

    assert partitioned_comps == [components]


@pytest.mark.skip(
    reason="Unable to set frequency and flux correctly for "
    "a multi-frequency skycomponent"
)
def test_fit_skycomponent_spectral_index(result_operations):
    """Check fits multi-frequency skycomponents"""
    single_freq_comp = result_operations["skycomponents_list"][0]
    sf_spec_indx = fit_skycomponent_spectral_index(single_freq_comp)

    frequency = numpy.linspace(0.9e8, 1.1e8, 3)
    flux = numpy.ones(shape=(3, 1), dtype=float)
    multi_freq_comp = SkyComponent(
        direction=result_operations["home"],
        frequency=frequency,
        name="multi_freq_sc",
        flux=flux,
        shape="Point",
        polarisation_frame=PolarisationFrame("stokesI"),
    )

    mf_spec_indx = fit_skycomponent_spectral_index(multi_freq_comp)

    assert sf_spec_indx == 0.0
    assert mf_spec_indx == 1
