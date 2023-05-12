import numpy as np


def weighted_median(data: np.ndarray, weights: np.ndarray) -> int:
    """
    Args:
      data (list or numpy.array): data
      weights (list or numpy.array): weights
    """
    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
    midpoint = 0.5 * sum(s_weights)
    if any(weights > midpoint):
        return (data[weights == np.max(weights)])[0]
    cs_weights = np.cumsum(s_weights)
    idx = np.where(cs_weights <= midpoint)[0][-1]
    return (
        np.mean(s_data[idx : idx + 2])
        if cs_weights[idx] == midpoint
        else s_data[idx + 1]
    )
