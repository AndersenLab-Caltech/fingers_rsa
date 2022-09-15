"""Helper functions for working with PyNWB data structures"""

import numpy as np
import pandas as pd
import pynwb
import tqdm


def count_trial_spikes(
    nwb: pynwb.file.NWBFile,
    start: float = 0.0,
    end: float = 1.0,
) -> pd.DataFrame:
    """Count spikes within the trial-aligned interval.

    :param nwb: NeuroDataWithoutBorders object including `trials` and `units`
    :param start: start time in seconds, relative to trial start
    :param end: end time in seconds, relative to trial start

    :returns: DataFrame of spike counts, indexed by trial (rows) and unit
        (columns)
    """
    times_df = align_spike_times_to_trials(nwb, start=start, end=end)
    count_df = times_df.applymap(len)
    return count_df


def align_spike_times_to_trials(
    nwb: pynwb.file.NWBFile,
    start: float = 0.0,
    end: float = 1.0,
) -> pd.DataFrame:
    """Find trial-aligned spike times for each unit.

    Only returns spikes within the given trial-relative interval.

    :param nwb: NeuroDataWithoutBorders object including `trials` and `units`
    :param start: start time in seconds, relative to trial start
    :param end: end time in seconds, relative to trial start

    :returns: DataFrame of trial-relative spike times, indexed by trial (rows)
        and unit (columns)
    """
    units = nwb.units
    trials = nwb.trials

    trial_times = trials.to_dataframe().start_time
    trial_times.index.name = "trial_id"
    include_intervals = pd.DataFrame(
        {
            "start": trial_times + start,
            "end": trial_times + end,
            "reference": trial_times,
        }
    )

    aligned_spike_times_list = [
        align_spike_times_to_absolute_interval(
            units,
            **interval,
        )
        for _, interval in tqdm.tqdm(
            include_intervals.iterrows(),
            desc=f"aligning trials {nwb.session_id}",
            total=len(include_intervals),
        )
    ]
    df = pd.concat(
        aligned_spike_times_list,
        keys=trial_times.index,
        axis=1,
    ).T
    return df


def align_spike_times_to_absolute_interval(
    units: pynwb.misc.Units,
    start: float,
    end: float,
    reference: float,
) -> pd.Series:
    """Find the spike times for each unit that fall within the given interval.

    :param units: NWB units table
    :param start: start time in seconds, relative to session-reference
    :param end: end time in seconds, relative to session-reference
    :param reference: reference time in seconds, relative to session-reference

    :returns: Series of spike times (with `reference` subtracted), indexed by
        unit
    """
    num_units = len(units)
    relative_spike_times_list = []
    for unit_idx in range(num_units):
        unit_spike_times: np.ndarray = units.get_unit_spike_times(
            unit_idx, (start, end)
        )
        relative_spike_times_list.append(unit_spike_times - reference)
    population_spike_times = pd.Series(
        relative_spike_times_list, index=units.to_dataframe().index
    )
    population_spike_times.index.name = "unit_id"
    return population_spike_times
