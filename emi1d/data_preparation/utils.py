
def get_single_metadata(meta_str, only_sfreq=False):
    """
    Parse metadata string and return a dictionary containing component, config,
    frequency, and source-receiver distance (s). If only_sfreq is True,
    return only s and frequency.

    Parameters:
    meta_str: str
        Metadata string in the format "component_config_freq_s".
        only_sfreq (bool, optional): If True, return only s and frequency.

    Returns:
        dict: Dictionary containing parsed metadata. If only_sfreq is True,
        contains only 'freq' and 's' keys.
    """
    meta_list = meta_str.split("_")

    if len(meta_list) != 4:
        raise ValueError("Invalid metadata string format. Expected format: 'component_config_freq_s'")

    meta_list = meta_str.split(sep="_")
    meta_dict = {
        "component": meta_list[0],
        "config": meta_list[1],
        "freq": float(meta_list[2]),
        "s": float(meta_list[3]),
    }
    if only_sfreq:
        s_freq = {key: value for key, value in meta_dict.items() if key in ['s', 'freq']}
        return s_freq
    else:
        return meta_dict

def linear_error(freq, slope=0.02):
    """
    Compute the standard deviation of a measurement (jitter)
    based on a linear model :math:`\sigma_{\mathrm{noise}}=slope * freq`
    Values of the slope are determined empirically from
    the observed data.

    Parameters
    ----------
    freq: float
        Frequency in Hertz
    Slope: float
        Slope

    Return
    ------
    sigma_noise: float
        Standard deviation of the measurement
    """
    sigma_noise = slope * freq
    return sigma_noise


def data_error(measurement, sigma_noise):
    """
    Compute the data error used for inversion

    Parameters
    ----------
    measurement: float
        Observed :math:`H_{\mathrm{s}}/H_{\mathrm{0}}`
        at a given frequency and source-receiver separation
    sigma_noise: float
        Standard deviation of the measurement

    Return
    ------
    data_error: float
        Data error of the measurement expressed as a ratio
    """
    data_error = sigma_noise / measurement
    return data_error

def make_regular_grid(nlay, dz):
    return = dz * np.ones((nlay, ))

def make_irregular_grid(dz):
    pass

def print_grid(grid, model=None):
    pass
