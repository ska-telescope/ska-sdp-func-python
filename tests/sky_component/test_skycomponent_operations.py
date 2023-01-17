"""
Unit tests for SkyComponent operations
"""
import astropy.units as u
import numpy
import pytest
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)
from ska_sdp_datamodels.sky_model.sky_model import SkyComponent

from ska_sdp_func_python.sky_component.operations import (
    filter_skycomponents_by_flux,
    find_nearest_skycomponent,
    find_nearest_skycomponent_index,
    find_separation_skycomponents,
    find_skycomponent_matches,
    find_skycomponent_matches_atomic,
    fit_skycomponent_spectral_index,
    image_voronoi_iter,
    insert_skycomponent,
    partition_skycomponent_neighbours,
    remove_neighbouring_components,
    restore_skycomponent,
    select_components_by_separation,
    select_neighbouring_components,
    voronoi_decomposition,
)

HOME = SkyCoord(
    ra=+179.0 * u.deg, dec=-40.0 * u.deg, frame="icrs", equinox="J2000"
)

FREQ = numpy.array([1e8])
FLUX = numpy.ones((1, 1))
POL_FRAME = PolarisationFrame("stokesI")


@pytest.fixture(scope="module", name="sky_comp_one")
def sky_comp_one_fixt():
    """
    SkyComponent fixture at RA=181, DEC=-33
    """
    phase_centre = SkyCoord(
        ra=+181.0 * u.deg, dec=-33.0 * u.deg, frame="icrs", equinox="J2000"
    )
    return SkyComponent(
        direction=phase_centre,
        frequency=FREQ,
        flux=FLUX,
        polarisation_frame=POL_FRAME,
    )


@pytest.fixture(scope="module", name="sky_comp_two")
def sky_comp_two_fixt():
    """
    SkyComponent fixture at RA=181, DEC=-35
    """
    phase_centre = SkyCoord(
        ra=+181.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
    )
    return SkyComponent(
        direction=phase_centre,
        frequency=FREQ,
        flux=FLUX,
        polarisation_frame=POL_FRAME,
    )


@pytest.fixture(scope="module", name="sky_comp_three")
def sky_comp_three_fixt():
    """
    SkyComponent fixture at RA=182, DEC=-36
    """
    phase_centre = SkyCoord(
        ra=+182.0 * u.deg, dec=-36.0 * u.deg, frame="icrs", equinox="J2000"
    )
    return SkyComponent(
        direction=phase_centre,
        frequency=FREQ,
        flux=FLUX,
        polarisation_frame=POL_FRAME,
    )


@pytest.fixture(scope="module", name="sky_comp_four")
def sky_comp_four_fixt():
    """
    SkyComponent fixture at RA=179, DEC=-40
    """
    return SkyComponent(
        direction=HOME, frequency=FREQ, flux=FLUX, polarisation_frame=POL_FRAME
    )


@pytest.fixture(scope="module", name="sky_comp_five")
def sky_comp_five_fixt():
    """
    SkyComponent fixture at RA=179, DEC=-39
    """
    phase_centre = SkyCoord(
        ra=+179.0 * u.deg, dec=-39.0 * u.deg, frame="icrs", equinox="J2000"
    )
    return SkyComponent(
        direction=phase_centre,
        frequency=FREQ,
        flux=FLUX,
        polarisation_frame=POL_FRAME,
    )


def test_find_nearest_skycomponent_index(
    sky_comp_one, sky_comp_two, sky_comp_five
):
    """
    The index of the SkyComponent nearest to a given direction
    (HOME) is found.
    """
    components = [sky_comp_one, sky_comp_two, sky_comp_five]
    result_index = find_nearest_skycomponent_index(HOME, components)
    assert result_index == 2


def test_find_nearest_skycomponent_zero_sep(
    sky_comp_one, sky_comp_two, sky_comp_four
):
    """
    Nearest SkyComponent is at the same direction as "home"
    -> separation is 0.
    """
    components = [sky_comp_one, sky_comp_two, sky_comp_four]
    result_sc, result_separation = find_nearest_skycomponent(HOME, components)

    assert result_sc == sky_comp_four
    assert result_separation == 0


def test_find_nearest_skycomponent_smallest_sep(sky_comp_one, sky_comp_five):
    """
    Nearest SkyComponent is not exactly at "home",
    so there is a small separation
    """
    components = [sky_comp_one, sky_comp_five]
    result_sc, result_separation = find_nearest_skycomponent(HOME, components)

    assert result_sc == sky_comp_five
    assert result_separation.round(3) == 0.017


def test_find_separation_skycomponents(
    sky_comp_one, sky_comp_two, sky_comp_three
):
    """
    Check separation of list of components compared to reference comps.

    test_comp1  test_comp2  test_comp3
    x           x           x           ref_comp1
    x           x           x           ref_comp2
    """
    comps_to_test = [sky_comp_one, sky_comp_two, sky_comp_three]
    ref_comps = [sky_comp_two, sky_comp_three]
    separations = find_separation_skycomponents(comps_to_test, ref_comps)

    expected_separations = numpy.array(
        [[0.03490659, 0.0, 0.02250552], [0.05429855, 0.02250552, 0.0]]
    )

    numpy.testing.assert_allclose(separations, expected_separations, atol=1e-7)


def test_find_skycomponent_matches_atomic(
    sky_comp_one, sky_comp_two, sky_comp_three
):
    """Check the matches"""
    components = [sky_comp_one, sky_comp_two, sky_comp_three]
    ref_components = [sky_comp_one, sky_comp_three]
    matches = find_skycomponent_matches_atomic(components, ref_components)

    assert len(matches) == 2


def test_find_skycomponent_matches(sky_comp_one, sky_comp_two, sky_comp_three):
    """Check matches"""
    components = [sky_comp_one, sky_comp_one, sky_comp_two, sky_comp_three]
    ref_components = [sky_comp_one, sky_comp_three]
    matches = find_skycomponent_matches(components, ref_components)

    assert len(matches) == 3


def test_select_components_by_separation(
    sky_comp_one, sky_comp_four, sky_comp_five
):
    """Check correct component is selected"""
    comps = [sky_comp_one, sky_comp_five, sky_comp_four]
    selected = select_components_by_separation(HOME, comps, rmax=0.1)

    assert selected == [sky_comp_five, sky_comp_four]


def test_select_neighbouring_components(sky_comp_one, sky_comp_three):
    """
    Check correct component is selected for each object
    in "components" list from the "target" list
    """
    components = [sky_comp_one]
    target = [sky_comp_three, sky_comp_one]
    result_idx, result_sep = select_neighbouring_components(components, target)

    assert (result_idx == [1]).all()
    assert (result_sep.deg == [0.0]).all()


def test_select_neighbouring_components_multi(
    sky_comp_one, sky_comp_two, sky_comp_three
):
    """
    Check correct component is selected for each object
    in "components" list from the "target" list
    """
    components = [sky_comp_one, sky_comp_three]
    target = [sky_comp_three, sky_comp_two, sky_comp_one]
    result_idx, result_sep = select_neighbouring_components(components, target)

    assert (result_idx == [2, 0]).all()
    assert (result_sep.deg == [0.0, 0.0]).all()


def test_remove_neighbouring_components(sky_comp_two):
    """
    From multiple components within a given distance,
    remove the faintest one.
    """
    new_comp = sky_comp_two.copy()
    new_comp.flux = sky_comp_two.flux + 10.0
    new_comp_2 = sky_comp_two.copy()
    new_comp_2.flux = sky_comp_two.flux + 5.0
    components = [new_comp_2, sky_comp_two, new_comp]
    distance = 1  # in radians

    result_idx, result_kept_comp = remove_neighbouring_components(
        components, distance
    )

    assert result_idx == [0, 2]
    assert result_kept_comp == [new_comp_2, new_comp]


def test_filter_skycomponents_by_flux(sky_comp_one):
    """
    SkyComponents with flux >= flux_min are kept
    """
    new_comp = sky_comp_one.copy()
    new_comp.flux = sky_comp_one.flux + 10.0
    new_comp_2 = sky_comp_one.copy()
    new_comp_2.flux = sky_comp_one.flux + 5.0
    components = [sky_comp_one, new_comp, new_comp_2]
    result_components = filter_skycomponents_by_flux(components, flux_min=6)

    assert result_components == [new_comp]


def test_insert_skycomponent(image, sky_comp_one):
    """Check a skycomponent is inserted to the image"""
    img = image.copy(deep=True)
    new_image = insert_skycomponent(img, sky_comp_one)
    assert new_image != img


def test_restore_skycomponent(image, sky_comp_two):
    """Check a skycomponent is restored into the image
    FIXME: what is the difference between restore_skycomponent
     and insert_skycomponent?
    """
    img = image.copy(deep=True)
    clean_beam = {"bmaj": 0.1, "bmin": 0.05, "bpa": -60.0}
    new_image = restore_skycomponent(img, sky_comp_two, clean_beam=clean_beam)

    assert new_image != image


def test_voronoi_decomposition(
    image, sky_comp_one, sky_comp_two, sky_comp_three
):
    """
    Check Voronoi decomposition

    - all components have to fit within the image coordinates
            if a component doesn't, its pixel coordinates
            are nan and that breaks the Voronoi object
            (scipy.spatial._qhull.Voronoi)

    Asserted values:
    - vor_structure.points: pixel coordinates of sky components on image
        e.g. sky_comp_two has the same direction as the
        phase_centre of the image, so its coordinates are
        in the centre (257, 257)
    - vor_structure.point_region: which Voronoi region
        each components falls within
        this gives the index of the regions
    - vor_structure.vertices: coordinates of Voronoi vertices
        which fall within the image

    # TODO: need to understand what vor_image is and add asserts for that
    """
    # pylint: disable=unused-variable
    # TODO: remove pylint disable once test for vor_image is in place

    # all three components fit within the image
    components = [sky_comp_one, sky_comp_two, sky_comp_three]
    vor_structure, vor_image = voronoi_decomposition(image, components)

    assert (
        vor_structure.points.round(6)
        == numpy.array(
            [[257.0, 489.663311], [257.0, 257.0], [162.871377, 140.179461]]
        )
    ).all()
    assert (vor_structure.point_region == numpy.array([1, 2, 3])).all()
    assert (
        vor_structure.vertices.round(6)
        == numpy.array([[-6.931885, 373.331656]])
    ).all()


def test_image_voronoi_iter(image, sky_comp_one, sky_comp_two, sky_comp_three):
    """
    Unit test for image_voronoi_iter.

    The yielded images will have 1s where the points belong to a
    certain voronoi region and zeroes everywhere else.
    """
    model = image.copy(deep=True)
    model["pixels"].data[...] = 1.0

    components = [sky_comp_one, sky_comp_two, sky_comp_three]

    num_images = 0
    for result_img in image_voronoi_iter(model, components):
        # there are non-zero data points in voronoi image
        assert numpy.sum(result_img["pixels"].data) > 1
        # not all of the points in the voronoi image are non-zero
        # only the ones that match the mask given by the decomposition
        assert numpy.sum(result_img["pixels"].data) < numpy.sum(
            model["pixels"].data
        )
        num_images += 1

    # there are three images to yield
    assert num_images == 3


def test_partition_skycomponent_neighbours(
    sky_comp_one, sky_comp_two, sky_comp_three, sky_comp_four, sky_comp_five
):
    """
    Check SkyComponents are partitioned correctly

    Output is a list of lists:
        len(outer-list) == num_targets
        len(inner-list): this will vary depending on how many
        components are closest to that specific target component
        from the component list.
    """
    components = [sky_comp_one, sky_comp_three, sky_comp_five]
    target = [sky_comp_two, sky_comp_four]

    partitioned_comps = partition_skycomponent_neighbours(components, target)

    assert len(partitioned_comps) == 2  # two targets
    # sky_comp_one and sky_comp_three are nearest
    # to sky_comp_two from target list
    assert partitioned_comps[0] == [sky_comp_one, sky_comp_three]
    # sky_comp_five is nearest to sky_comp_four from target list
    assert partitioned_comps[1] == [sky_comp_five]


def test_fit_skycomponent_spectral_index_single_freq(sky_comp_one):
    """
    Spectral index of single frequency SkyComponent
    """
    result_spec_index = fit_skycomponent_spectral_index(sky_comp_one)
    assert result_spec_index == 0.0


def test_fit_skycomponent_spectral_index_multi_freq():
    """
    Spectral index of multi frequency SkyComponent
    """
    frequency = numpy.linspace(0.9e8, 1.1e8, 3)
    flux = numpy.power(frequency / 1e8, -0.7).reshape((3, 1))
    multi_freq_comp = SkyComponent(
        direction=HOME,
        frequency=frequency,
        flux=flux,
        polarisation_frame=PolarisationFrame("stokesI"),
    )

    result_spec_index = fit_skycomponent_spectral_index(multi_freq_comp)

    numpy.testing.assert_almost_equal(result_spec_index, -0.7, decimal=7)
