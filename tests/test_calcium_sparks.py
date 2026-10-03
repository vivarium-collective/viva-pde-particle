"""Calcium-spark benchmark (Schaff et al. 2016, Tests 1-2): model, partition and reference."""
from __future__ import annotations

import numpy as np
import pytest

from viva_pde_particle.benchmarks.calcium_sparks import (
    TEST1,
    TEST2,
    calcium_sparks_model,
    channel_nodes,
    open_probability,
    separable_expectation,
    spark_error,
    spark_grid,
)
from viva_pde_particle.model import partition


def test_geometry_and_channels():
    g = spark_grid()
    assert g.num == (102, 22, 3) and g.spacing == pytest.approx((0.1, 0.1, 0.25))
    m = calcium_sparks_model(TEST1)
    closed = m.species_by_name("C").initial
    assert closed.sum() == 24 and closed[channel_nodes(g)].min() == 1  # 4 columns x 6 channels


def test_partition_keeps_influx_on_pde_side_only():
    rx = {r["name"]: r for r in partition(calcium_sparks_model(TEST1)).particles["reactions"]}
    assert set(rx) == {"open", "close"}  # influx O -> O + U is a PDE source only
    rx2 = {r["name"]: r for r in partition(calcium_sparks_model(TEST2)).particles["reactions"]}
    assert rx2["open"]["fields"] == ["U"] and rx2["open"]["rate_constant"] == pytest.approx(TEST2.k_on / TEST2.U0)


def test_open_probability_and_expectation():
    assert open_probability(0.0) == 0.0
    assert open_probability(100.0) == pytest.approx(1 / 6)
    times = np.array([0.0, 0.5])
    ref = separable_expectation(times, dt_ref=1e-3)
    assert ref.shape == (2, 3, 22, 102)
    np.testing.assert_allclose(ref[0], TEST1.U0)
    assert ref[1].max() > TEST1.U0 and ref[1].min() >= TEST1.U0 - 1e-12
    assert spark_error(ref, ref, TEST1.U0) == 0.0
