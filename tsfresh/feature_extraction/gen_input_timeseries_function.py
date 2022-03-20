import itertools
import pandas as pd

########
# Where I will put a function which generates input timeseries from timeseries.
def engineer_input_timeseries(ts, compute_deriv, compute_phasediff, n_layers = 1):
    ''' 
        time series differencing and phase difference operations to add new engineered time series to the input time series
        NOTE: For generalisation, the convention specified in (Scott) the honours project paper has been changed.
        NOTE: Call this function n times on each output for nth order differencing etc... 

        params:
             ts (pd.DataFrame): time series input with n ts_kinds (n columns) ---- e.g. phase0, phase1, phase2.
             n_layers (int): number of times to do time series differencing and phase differences. Do not reccomend this number be large as has poor scaling.
        returns:
             ts (pd.DataFrame): time series with 3 phases and 18 signals. See function implementation for details
    '''
    ts = pd.DataFrame(ts)
    
    # first order differencing
    def timediff(ts,ts_kinds):
        for ts_kind in ts_kinds:
            ts["dt_" + ts_kind] = ts[ts_kind].diff()
            ts.loc[0,["dt_" + ts_kind]] = 0 # adjust for the NaN value for temporal derivatives at first index...
        return ts

    # phase differences
    def spatialdiff(ts,ts_kinds):
        combs = itertools.combinations(ts_kinds, r = 2)
        for first_ts_kind, second_ts_kind in combs:
            ts["D_" + first_ts_kind + second_ts_kind] = ts[first_ts_kind] - ts[second_ts_kind] 
        return ts
    
    # compute phase differences and derivatives
    ts_kinds = ts.columns
    if compute_deriv:
        ts = timediff(ts, ts_kinds)
    if compute_phasediff:
        ts = spatialdiff(ts,ts_kinds)

    return ts
