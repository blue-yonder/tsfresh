# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import unittest

import dask.dataframe as dd
import numpy as np
import pandas as pd

from tsfresh import extract_features, extract_relevant_features
from tsfresh.examples.driftbif_simulation import load_driftbif, sample_tau, velocity
from tsfresh.feature_extraction import MinimalFCParameters


class DriftBifSimlationTestCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_intrinsic_velocity_at_default_bifurcation_point(self):
        """
        The intrinsic velocity of a dissipative soliton at the Drift bifurcation point is zero.
        """
        ds = velocity(tau=1.0 / 0.3)
        self.assertEqual(ds.deterministic, 0.0)

    def test_relaxation_dynamics(self):
        """
        Test accuracy of integrating the deterministic dynamics [6, p. 116]
        """
        ds = velocity(tau=1.01 / 0.3, R=0)
        v0 = 1.01 * ds.deterministic

        Nt = 100  # Number of time steps
        v = ds.simulate(Nt, v0=np.array([v0, 0.0]))

        k3t = ds.kappa_3 * ds.tau
        k3st = ds.kappa_3 ** 2 * ds.tau
        a0 = v0 / ds.kappa_3

        def acceleration(t):
            return ds.kappa_3 * (
                a0
                * np.sqrt(k3t - 1)
                * np.exp(k3st * t)
                / np.sqrt(
                    np.exp(2.0 * k3st * t) * ds.Q * a0 ** 2
                    + np.exp(2.0 * ds.kappa_3 * t) * (k3t - 1 - ds.Q * a0 ** 2)
                )
            )

        t = ds.delta_t * np.arange(Nt)
        return np.testing.assert_array_almost_equal(
            v[:, 0], np.vectorize(acceleration)(t), decimal=8
        )

    def test_equlibrium_velocity(self):
        """
        Test accuracy of integrating the deterministic dynamics for equilibrium velocity [6, p. 116]
        """
        ds = velocity(tau=1.01 / 0.3, R=0)
        v0 = ds.deterministic

        Nt = 100  # Number of time steps
        v = ds.simulate(Nt, v0=np.array([v0, 0.0]))

        return np.testing.assert_array_almost_equal(
            v[:, 0] - v0, np.zeros(Nt), decimal=8
        )

    def test_dimensionality(self):
        ds = velocity(tau=1.0 / 0.3)
        Nt = 10
        v = ds.simulate(Nt)
        self.assertEqual(
            v.shape,
            (Nt, 2),
            "The default configuration should return velocities "
            "from a two-dimensional dissipative soliton.",
        )

        v = ds.simulate(Nt, v0=np.zeros(3))
        self.assertEqual(
            v.shape,
            (Nt, 3),
            "The returned vector should reflect the dimension of the initial condition.",
        )

    def test_relevant_feature_extraction(self):
        df, y = load_driftbif(100, 10, classification=False)

        df["id"] = df["id"].astype("str")
        y.index = y.index.astype("str")

        X = extract_relevant_features(
            df,
            y,
            column_id="id",
            column_sort="time",
            column_kind="dimension",
            column_value="value",
        )

        self.assertGreater(len(X.columns), 10)


class SampleTauTestCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_range(self):
        tau = sample_tau(100)
        self.assertGreaterEqual(min(tau), 2.87)
        self.assertLessEqual(max(tau), 3.8)

    def test_ratio(self):
        tau = sample_tau(100000, ratio=0.4)
        sample = np.array(tau)
        before = np.sum(sample <= 1 / 0.3)
        beyond = np.sum(sample > 1 / 0.3)
        self.assertTrue(abs(0.4 - float(before) / (before + beyond)) < 0.006)


class LoadDriftBifTestCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_classification_labels(self):
        X, y = load_driftbif(10, 100)
        self.assertEqual(set(y), {0, 1})

    def test_regression_labels(self):
        Nsamples = 10
        X, y = load_driftbif(Nsamples, 100, classification=False)
        self.assertEqual(
            y.size,
            np.unique(y).size,
            "For regression the target vector is expected to not contain any dublicated labels.",
        )

    def test_default_dimensionality(self):
        Nsamples = 10
        Nt = 100
        X, y = load_driftbif(Nsamples, Nt)
        self.assertEqual(X.shape, (2 * Nt * Nsamples, 4))

    def test_configured_dimensionality(self):
        Nsamples = 10
        Nt = 100
        X, y = load_driftbif(Nsamples, Nt, m=3)
        self.assertEqual(X.shape, (3 * Nt * Nsamples, 4))
