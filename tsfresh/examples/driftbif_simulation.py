# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

# Thanks to Andreas Kempa-Liehr for providing this snippet

import pandas as pd
import numpy as np

# todo: add possibility to extract data set for regression problem (estimation of tau parameter)
# todo: add unit test

class velocity(object):
    """
    Simulates the velocity of one dissipative soliton (kind of self organized particle)

    label 0 means tau<=1/0.3, Dissipative Soliton with Brownian motion (purely noise driven)
    label 1 means tau> 1/0.3, Dissipative Soliton with Active Brownian motion (intrinsiv velocity with overlaid noise)

    References
    ----------

    .. [6] Andreas Kempa-Liehr (2013, p. 159-170)
        Dynamics of Dissipative Soliton
        Dissipative Solitons in Reaction Diffusion Systems.
        Springer: Berlin


    >>> ds = velocity(tau=3.5) # Dissipative soliton with equilibrium velocity 1.5e-3
    >>> print(ds.label) # Discriminating before or beyond Drift-Bifurcation
    1
    >>> print(ds.deterministic) # Equilibrium velocity
    0.0015191090506254991
    >>> v = ds.simulate(20000) # Simulate velocity time series with 20000 time steps being disturbed by Gaussian white noise
    """

    def __init__(self, tau=2.87, kappa_3=0.3, Q=1950.0, R=3e-4, delta_t=0.005):
        """
        :param tau: time-scale constant
        :type tau: float
        :param kappa_3: Feedback of fast inhibitor
        :type kappa_3:
        :param Q: Shape parameter of dissipative soliton
        :type Q: float
        :param R: Noise amplitude
        :type R: float
        :param delta_t: temporal discretization
        :type delta_t: float
        """
        # todo: improve description of constants
        # todo: add start seed

        self.delta_t = delta_t
        self.a = self.delta_t * kappa_3 ** 2 * (tau - 1.0 / kappa_3)
        self.b = self.delta_t * Q / kappa_3
        self.label = int(tau > 1.0 / kappa_3)
        self.c = np.sqrt(self.delta_t) * R
        self.delta_t = self.delta_t

        if tau <= 1.0 / kappa_3:
            self.deterministic = 0.0
        else:
            self.deterministic = kappa_3 ** 1.5 * np.sqrt((tau - 1.0 / kappa_3) / Q)

    def __call__(self, v):
        """
        returns deterministic dynamic = acceleration (without noise)

        :param v: vector of velocity
        :rtype v:
        :return:
        :return type:
        """

        # todo: which type v, array?
        # todo: descripton of return?

        return v * (1.0 + self.a - self.b * np.dot(v, v))

    def simulate(self, N, v0=np.zeros(2)):
        """

        :param N: number of time steps
        :type N:
        :param v0: initial velocity
        :return:
        :rtype:
        """

        # todo: fill out docstring
        # todo: complete parameter description

        v = [v0]                        # first value is initial condition
        n = N - 1                       # Because we are returning the initial condition,
                                        # only (N-1) time steps are computed
        gamma = np.random.randn(n, 2)
        for i in xrange(n):
            next_v = self.__call__(v[i]) + self.c * gamma[i]
            v.append(next_v)
        v_vec = np.array(v)
        return v_vec


def load_driftbif(n, l):
    """
    Creates and loads the drift bifurcation dataset.

    :param n: number of different samples
    :type n: int
    :param l: length of the time series
    :type l: int
    :return: X, y. Time series container and target vector
    :rtype X: pandas.DataFrame
    :rtype y: pandas.DataFrame
    """

    # todo: add ratio of classes
    # todo: add start seed
    # todo: draw tau random from range [2, 4] so we really get a random dataset
    # todo: add variable for number of dimensions

    m = 2 # number of different time series for each sample
    id = np.repeat(range(n), l * m)
    dimensions = list(np.repeat(range(m), l)) * n

    labels = list()
    values = list()

    ls_tau = np.linspace(2.87, 3.8, n).tolist()

    for i, tau in enumerate(ls_tau):
        ds = velocity(tau=tau)
        labels.append(ds.label)
        values.append(ds.simulate(l).transpose().flatten())
    time = np.stack([ds.delta_t * np.arange(l)] * n * m).flatten()

    df = pd.DataFrame({'id': id, "time": time, "value": np.stack(values).flatten(), "dimension": dimensions})
    y = pd.Series(labels)
    y.index = range(n)
    
    return df, y
