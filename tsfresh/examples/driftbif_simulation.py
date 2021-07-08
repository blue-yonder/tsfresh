# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

# Thanks to Andreas W. Kempa-Liehr for providing this snippet

import logging

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)


class velocity:
    """
    Simulates the velocity of a dissipative soliton (kind of self organized particle) [6]_.
    The equilibrium velocity without noise R=0 for
    $\tau>1.0/\\kappa_3$ is $\\kappa_3 \\sqrt{(tau - 1.0/\\kappa_3)/Q}.
    Before the drift-bifurcation $\tau \\le 1.0/\\kappa_3$ the velocity is zero.

    References
    ----------

    .. [6] Andreas Kempa-Liehr (2013, p. 159-170)
        Dynamics of Dissipative Soliton
        Dissipative Solitons in Reaction Diffusion Systems.
        Springer: Berlin


    >>> ds = velocity(tau=3.5) # Dissipative soliton with equilibrium velocity 1.5e-3
    >>> print(ds.label) # Discriminating before or beyond Drift-Bifurcation
    1

    # Equilibrium velocity
    >>> print(ds.deterministic)
    0.0015191090506254991

    # Simulated velocity as a time series with 20000 time steps being disturbed by Gaussian white noise
    >>> v = ds.simulate(20000)
    """

    def __init__(self, tau=3.8, kappa_3=0.3, Q=1950.0, R=3e-4, delta_t=0.05, seed=None):
        """
        :param tau: Bifurcation parameter determining the intrinsic velocity of the dissipative soliton,
                    which is zero for tau<=1.0/kappa_3 and np.sqrt(kappa_3**3/Q * (tau - 1.0/kappa_3)) otherwise
        :type tau: float
        :param kappa_3: Inverse bifurcation point.
        :type kappa_3:
        :param Q: Shape parameter of dissipative soliton
        :type Q: float
        :param R: Noise amplitude
        :type R: float
        :param delta_t: temporal discretization
        :type delta_t: float
        """
        # done: add start seed

        self.delta_t = delta_t
        self.kappa_3 = kappa_3
        self.Q = Q
        self.tau = tau
        self.a = self.delta_t * kappa_3 ** 2 * (tau - 1.0 / kappa_3)
        self.b = self.delta_t * Q / kappa_3
        self.label = int(tau > 1.0 / kappa_3)
        self.c = np.sqrt(self.delta_t) * R
        self.delta_t = self.delta_t

        if seed is not None:
            np.random.seed(seed)

        if tau <= 1.0 / kappa_3:
            self.deterministic = 0.0
        else:
            self.deterministic = kappa_3 ** 1.5 * np.sqrt((tau - 1.0 / kappa_3) / Q)

    def __call__(self, v):
        """
        returns deterministic dynamic = acceleration (without noise)

        :param v: initial velocity vector
        :rtype v: ndarray
        :return: velocity vector of next time step
        :return type: ndarray
        """

        return v * (1.0 + self.a - self.b * np.dot(v, v))

    def simulate(self, N, v0=np.zeros(2)):
        """

        :param N: number of time steps
        :type N: int
        :param v0: initial velocity vector
        :type v0: ndarray
        :return: time series of velocity vectors with shape (N, v0.shape[0])
        :rtype: ndarray
        """

        v = [v0]  # first value is initial condition
        n = N - 1  # Because we are returning the initial condition,
        # only (N-1) time steps are computed
        gamma = np.random.randn(n, v0.size)
        for i in range(n):
            next_v = self.__call__(v[i]) + self.c * gamma[i]
            v.append(next_v)
        v_vec = np.array(v)
        return v_vec


def sample_tau(n=10, kappa_3=0.3, ratio=0.5, rel_increase=0.15):
    """
    Return list of control parameters

    :param n: number of samples
    :type n: int
    :param kappa_3: inverse bifurcation point
    :type kappa_3: float
    :param ratio: ratio (default 0.5) of samples before and beyond drift-bifurcation
    :type ratio: float
    :param rel_increase: relative increase from bifurcation point
    :type rel_increase: float
    :return: tau. List of sampled bifurcation parameter
    :rtype tau: list
    """
    assert ratio > 0 and ratio <= 1
    assert kappa_3 > 0
    assert rel_increase > 0 and rel_increase <= 1
    tau_c = 1.0 / kappa_3

    tau_max = tau_c * (1.0 + rel_increase)
    tau = tau_c + (tau_max - tau_c) * (np.random.rand(n) - ratio)
    return tau.tolist()


def load_driftbif(n, length, m=2, classification=True, kappa_3=0.3, seed=False):
    """
    Simulates n time-series with length time steps each for the m-dimensional velocity of a dissipative soliton

    classification=True:
    target 0 means tau<=1/0.3, Dissipative Soliton with Brownian motion (purely noise driven)
    target 1 means tau> 1/0.3, Dissipative Soliton with Active Brownian motion (intrinsiv velocity with overlaid noise)

    classification=False:
    target is bifurcation parameter tau

    :param n: number of samples
    :type n: int
    :param length: length of the time series
    :type length: int
    :param m: number of spatial dimensions (default m=2) the dissipative soliton is propagating in
    :type m: int
    :param classification: distinguish between classification (default True) and regression target
    :type classification: bool
    :param kappa_3: inverse bifurcation parameter (default 0.3)
    :type kappa_3: float
    :param seed: random seed (default False)
    :type seed: float
    :return: X, y. Time series container and target vector
    :rtype X: pandas.DataFrame
    :rtype y: pandas.DataFrame
    """

    # todo: add ratio of classes

    if m > 2:
        logging.warning(
            "You set the dimension parameter for the dissipative soliton to m={}, however it is only"
            "properly defined for m=1 or m=2.".format(m)
        )

    id = np.repeat(range(n), length * m)
    dimensions = list(np.repeat(range(m), length)) * n

    labels = list()
    values = list()

    ls_tau = sample_tau(n, kappa_3=kappa_3)

    for i, tau in enumerate(ls_tau):
        ds = velocity(tau=tau, kappa_3=kappa_3, seed=seed)
        if classification:
            labels.append(ds.label)
        else:
            labels.append(ds.tau)
        values.append(ds.simulate(length, v0=np.zeros(m)).transpose().flatten())
    time = np.stack([ds.delta_t * np.arange(length)] * n * m).flatten()

    df = pd.DataFrame(
        {
            "id": id,
            "time": time,
            "value": np.stack(values).flatten(),
            "dimension": dimensions,
        }
    )
    y = pd.Series(labels)
    y.index = range(n)

    return df, y
