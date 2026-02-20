"""
measurements.py contains functions that perform measurements to validate the performance of the transmitter and
reciever chains. It also contains functions to plot spectra and waveforms
"""
from typing import Dict, Tuple
import warnings
from numpy import zeros, float64, isclose, cos, sin, pi, exp, arange, \
                  mean, log10, pad, complex64, ceil, log2, fft, complex128, linspace, argsort
from numpy import sum as np_sum
from numpy import abs as np_abs
from numpy.typing import NDArray
from scipy.signal import welch
import matplotlib.pyplot as plt

from .transmitter import TETRA_SYMBOL_RATE

ACPR_OFFSETS_TETRA_SPEC_DICT = {"25 kHz": (25E3, -55, -36), "-25 kHz": (-25E3, -55, -36),
                                "50 kHz": (50E3, -65, -36), "-50 kHz": (-50E3, -65, -36),
                                "75 kHz": (75E3, -65, -36), "-75 kHz": (-75E3, -65, -36)}

WIDEBAND_NOISE_TETRA_SPEC_DICT = {"100 kHz to 250 kHz": ((100E3, 250E3), -74, -55),
                                  "-100 kHz to -250 kHz": ((-250E3, -100E3), -74, -55),
                                  "250 kHz to 500 kHz": ((250E3, 500E3), -80, -55),
                                  "-250 kHz to 500 kHz": ((-500E3, -250E3), -80, -55)}

WIDEBAND_NOISE_SWEEP_OFFSET = 12.5e3


def _rrc_fir(sps: int, symbol_span: int = 30, beta: float = 0.35) -> NDArray[float64]:
    """
    Generate fir coefficents of an RRC filter via impulse response truncation, where symbol span is how many symbols
    the FIR filter will span (normally infinite) and sps is the number of samples per symbol

    :param sps: Number of samples per symbol
    :type sps: int
    :param symbol_span: The number of symbols that the fir shall span, larger numbers increase accuracy at the cost of
     increased complexity of calculation and filter size
    :type symbol_span: int
    :param beta: The rolloff factor of the root raised cosine filter to design, defaults to 0.35 which is the TETRA
     standard value. Defaults to 30 minimum, sometimes refered to as alpha.
    :type beta: float
    :return: Returns the FIR coefficents of the rrc filter as float64 values stored in a numpy array, ready to be
     convolved with the target data, note that the filter is centered at f=0Hz.
    :rtype: NDArray[float64]
    """
    if symbol_span < 30:
        raise RuntimeWarning(f"Measurements: symbol span of RRC filter is :{symbol_span}, expected >=30 per 300-394")

    n_len = (symbol_span * sps) + 1
    n0 = (n_len-1)//2
    h = zeros(shape=n_len, dtype=float64)
    for n in range(n_len):
        tn = (n-n0)/sps
        if isclose(tn, 0):
            h[n] = 1 - beta + ((4*beta)/pi)
        else:
            if isclose(tn, (1/(4*beta))) or isclose(tn, (-1/(4*beta))):
                h[n] = (beta/(2**(0.5)))*((1+(2/pi))*sin(pi/(4*beta)) + (1-(2/pi))*cos(pi/(4*beta)))
            else:
                h[n] = ((sin(pi*tn*(1-beta)))+(4*beta*tn*cos(pi*tn*(1+beta))))/(pi*tn*(1-((4*beta*tn)**2)))
    h /= np_sum(h)
    return h


def _fft_convolve_rows(data_2d: NDArray[complex64], hcoef: NDArray[float64]) -> NDArray[complex128]:
    """
    Performs convolution via fft on many rows of numpy data with each N number of columns/entries, against hcoef with
    L number of points resulting a full return result of length N+L-1

    :param data_2d: 2-Dimen. numpy array, where each row much be convolved with hcoef with N columns
    :type data_2d: NDArray[complex64]
    :param hcoef: Filter impulse response to coefficents to convolve with L number of taps/coefficents
    :type hcoef: NDArray[float64]
    :return: Returns full convolution result as complex128 result, with length = N+L-1
    :rtype: NDArray[complex128]
    """

    _, n_len = data_2d.shape
    l_len = hcoef.size
    n_full = n_len + l_len - 1  # Length of row convolution result for "full"

    nfft = 1 << int(ceil(log2(n_full)))  # align with 2^N, for tx output this is 2_097_152 point fft

    data_fft = fft.fft(data_2d, nfft, axis=1)  # Row wise fft
    filt_fft = fft.fft(hcoef, nfft)

    result = fft.ifft(data_fft * filt_fft[None, :], axis=1)  # row wise multiplication, ifft row wise

    return result[:, :n_full]  # Discard zero-padding, only return the full convolvution result


def tx_acpr_measurement(tx_data: NDArray[complex64], sn0: int, snmax: int,
                        sample_rate: int,
                        ignore_warnings: bool = False) -> Dict[str, Tuple[float, float, str, bool]]:
    """
    Performs ACPR measurements on transmit data for 25, 50, and 75 kHz channel offsets through a RRC filter with 0.35
    roll off, and determines the relative power ratio, then after considering all M burst determines the mean power
    ratio and absolute power and evaluates against the TETRA MS Tx specifications for class 4 above 700 MHz. If the mask
    fails then it raises a warning, and regardless of failure returns the resulting test data in a dict.

    :param tx_data:  2-D array of M bursts of data, arranged as burst per row, stored as complex64 or a single 1 burst,
     stored in a 1-D array of complex64 data
    :type tx_data: NDArray[complex64]
    :param sn0: The SN0 index of the burst data
    :type sn0: int
    :param snmax: The SNmax index of the burst data
    :type snmax: int
    :param sample_rate: The sample rate of the passed data
    :type sample_rate: int
    :param ignore_warnings: If true does not raise warnings when tests fail, default is False
    :type ignore_warnings: bool
    :return: Averages over M bursts during the SN0-SNmax period finding the desired (0Hz) band power through an ideal
     RRC, then translates the data to center at frequency offsets of 25, 50, 75 kHz and performs the same calculation
     and determines the relative and absolute powers w.r.t. P0, raises a Warning if the mask is exceeded, and
     returns a dict with format: {"Frequency Offset", (absolute power, relative power to P0, requirement, Passed?)}
    :rtype: Dict[str, Tuple[float, float, str, bool]]
    """
    # Assumed averaging has already took place at higher level, and that the original signal is complex baseband
    sps = int(sample_rate / TETRA_SYMBOL_RATE)
    rrc_coefficents = _rrc_fir(sps)
    rrc_gd = (len(rrc_coefficents)-1) // 2

    if tx_data.ndim == 1:
        tx_data = tx_data.reshape(1, -1)
    # 1. Compute P0 of desired signal which is at baseband as reference point
    tx_pad = pad(tx_data, ((0, 0), (rrc_gd, rrc_gd)), 'constant')  # pad zeros at the end
    # 1a. Apply RRC filtering then isolated SN0-SNMax
    tx_full = _fft_convolve_rows(tx_pad, rrc_coefficents)
    # Isolate desired portion
    tx_full = tx_full[:, sn0 + 2*rrc_gd: (snmax+sps) + 2*rrc_gd]
    # 1b. Calculate the power of the signal row wise and store the result
    p0 = mean(np_abs(tx_full)**2, axis=1) / 50  # Row wise powers ready for acpr calculations
    p0_dbm = ((10**log10(p0.mean() + 1e-30)) + 30).astype(float64)

    results_dict = {"0 Hz": (0.0, float(round(p0_dbm, 2)), "None", True)}
    t = arange(tx_data.shape[1], dtype=float64)

    for band, value in ACPR_OFFSETS_TETRA_SPEC_DICT.items():
        # 2. Shift data to align at frequency offset as 0Hz
        lo_vector = exp(-1j * (2*pi*value[0]) * t/sample_rate)
        band_pad = pad(tx_data.copy() * lo_vector[None, :], ((0, 0), (rrc_gd, rrc_gd)), 'constant')
        band_full = _fft_convolve_rows(band_pad, rrc_coefficents)
        band_full = band_full[:, sn0 + 2*rrc_gd: (snmax+sps) + 2*rrc_gd]

        # 3. Calculate statistics
        pband = mean(np_abs(band_full)**2, axis=1) / 50
        pband_rel = (pband.copy() + 1e-30) / (p0 + 1e-30)
        pband_dbc = 10*log10(pband_rel.mean())
        pband_abs = pband_dbc + p0_dbm
        passed = True
        req = f"{value[1]:.2f} dBc"
        if pband_dbc > value[1]:
            req = f"{value[2]:.2f} dBm"
            # If we are within the absolute spec then we can still be in compliance
            if pband_abs > value[2]:
                passed = False
                if not ignore_warnings:
                    warnings.warn(f"\nTx ACPR Measurement Failure: for {band} measured power ratio was {pband_dbc:.2f}"
                                  f" with a real value of {pband_abs:.2f}, which is greater than the relative and"
                                  f" absolute specifications of {value[1]:.2f} and "
                                  f" {value[2]:.2f} respectively. P0 and Pband stats are:"
                                  f" P0std: {(10*log10(p0.std() + 1e-30)) +30:.3f} dBm,"
                                  f" P0mean: {p0_dbm} dBm,"
                                  f" Pbandstd: {(10*log10(pband.std() + 1e-30)) +30:.3f} dBm,"
                                  f" Pbandmean: {(10*log10(pband.mean() + 1e-30)) +30:.3f} dBm,")
        results_dict[band] = (float(round(pband_dbc, 2)), float(round(pband_abs, 2)), req, passed)

    return results_dict


def _pad_and_rrc_convolve(tx_data: NDArray[complex64], hcoef: NDArray[float64]) -> NDArray[complex128]:
    """
    Pads the start and end of tx_data input, an expected 2-dimen. numpy array of data by the group delay of hcoef,
    then convolves it via fft, and returns the isolated unpadded result, discarding the pads

    :param tx_data: 2-D array of M bursts of data, arranged as burst per row, stored as complex64
    :type tx_data: NDArray[complex64]
    :param hcoef: Filter impulse response to coefficents to convolve with L number of taps/coefficents
    :type hcoef: NDArray[float64]
    :return: Returns convolution result as complex128 result, with length = len(tx_data)
    :rtype: NDArray[complex128]
    """
    rrc_gd = (len(hcoef)-1) // 2
    tx_pad = pad(tx_data.copy(), ((0, 0), (rrc_gd, rrc_gd)), 'constant')  # pad zeros at the end
    tx_full = _fft_convolve_rows(tx_pad, hcoef)
    return tx_full[:, 2*rrc_gd: -2*rrc_gd]


def tx_wideband_noise_measurement(tx_data: NDArray[complex64], sn0: int, snmax: int,
                                  sample_rate: int,
                                  ignore_warnings: bool = False) -> Dict[str, Tuple[float, float, str, bool]]:
    """
    Performs wideband noise measurements on transmit data for for the 100-250khz and 250-500khz band offsets
    through a RRC filter with 0.35 rolloff. Measures at overlapping offsets in the band to determine the worse case
    performance, and evaluattes it against the TETRA MS Tx specifications for class 4 above 700 MHz. If the mask
    fails then it raises a warning, and regardless of failure returns the resulting test data in a dict.

    :param tx_data:  2-D array of M bursts of data, arranged as burst per row, stored as complex64 or a single 1 burst,
     stored in a 1-D array of complex64 data
    :type tx_data: NDArray[complex64]
    :param sn0: The SN0 index of the burst data
    :type sn0: int
    :param snmax: The SNmax index of the burst data
    :type snmax: int
    :param sample_rate: The sample rate of the passed data
    :type sample_rate: int
    :param ignore_warnings: If true does not raise warnings when tests fail, default is False
    :type ignore_warnings: bool
    :return: Averages over M bursts during the SN0-SNmax period finding the desired (0Hz) band power through an ideal
     RRC, then translates the data to center at overlapping 12.5kHz offsets through out the 100-250kHz and 250-500kHz
     band and performs the same calculation and finds the worst case performance and compare it against the TETRA
     wideband noise mask specficiations, if it fails it raises a warning, and regardless of failure returns a dict
     with format: {"Frequency Offset", (absolute power, relative power to P0, requirement, Passed?)}
    :rtype: Dict[str, Tuple[float, float, str, bool]]
    """
    # 1. Filter tx_data and gate useful part
    sps = int(sample_rate / TETRA_SYMBOL_RATE)
    rrc_coefficents = _rrc_fir(sps)

    if tx_data.ndim == 1:
        tx_data = tx_data.reshape(1, -1)
    tx_full = _pad_and_rrc_convolve(tx_data, rrc_coefficents)
    # Isolate desired portion
    tx_full = tx_full[:, sn0: snmax+sps]

    p0 = mean(np_abs(tx_full)**2, axis=1) / 50  # Row wise powers ready for acpr calculations
    p0_dbm = ((10**log10(p0.mean() + 1e-30)) + 30).astype(float64)

    results_dict = {"0 Hz": (0.0, float(round(p0_dbm, 2)), "None", True)}
    t = arange(tx_data.shape[1], dtype=float64)

    for band, value in WIDEBAND_NOISE_TETRA_SPEC_DICT.items():
        # Instead of measuring at a single point we must measure at multiple points overlaping across the band, and
        # report the worst case value
        num_steps = int((round(value[0][1] - value[0][0]) / WIDEBAND_NOISE_SWEEP_OFFSET)) + 1
        if num_steps < 0:
            num_steps = abs(num_steps)+1
            lo_steps = linspace(value[0][0],
                                value[0][0] - WIDEBAND_NOISE_SWEEP_OFFSET*(num_steps-1), num_steps, dtype=float64)
        else:
            lo_steps = linspace(value[0][0],
                                value[0][0] + WIDEBAND_NOISE_SWEEP_OFFSET*(num_steps-1), num_steps, dtype=float64)

        pband_tracker = {'pband_abs_max': -200, 'pband_dbc_max': -100, 'offset_max': 0}
        for k in range(num_steps):
            # 2. Shift data to align at frequency offset as 0Hz
            lo_vector = exp(-1j * (2*pi*lo_steps[k]) * t/sample_rate)
            band_full = _pad_and_rrc_convolve((tx_data.copy() * lo_vector[None, :]), rrc_coefficents)
            band_full = band_full[:, sn0: snmax+sps]

            # 3. Calculate statistics
            pband_rel = ((mean(np_abs(band_full)**2, axis=1) / 50).copy() + 1e-30) / (p0 + 1e-30)
            pband_dbc = 10*log10(pband_rel.mean())
            pband_abs = pband_dbc + p0_dbm
            if pband_abs > pband_tracker['pband_abs_max']:
                pband_tracker['pband_abs_max'] = pband_abs
                pband_tracker['pband_dbc_max'] = pband_dbc
                pband_tracker['offset_max'] = lo_steps[k]
            elif pband_dbc > pband_tracker['pband_dbc_max']:
                pband_tracker['pband_abs_max'] = pband_abs
                pband_tracker['pband_dbc_max'] = pband_dbc
                pband_tracker['offset_max'] = lo_steps[k]

        # Review results of band
        passed = True
        req = f"{value[1]:.2f} dBc"
        if pband_tracker['pband_dbc_max'] > value[1]:
            req = f"{value[2]:.2f} dBm"
            # If we are within the absolute spec then we can still be in compliance
            if pband_tracker['pband_abs_max'] > value[2]:
                passed = False
                if not ignore_warnings:
                    warnings.warn(f"\nTx Wideband Noise Measurement Failure: for {band} band measured at"
                                  f"{pband_tracker['offset_max']}"
                                  f" the power ratio was {pband_tracker['pband_dbc_max']:.2f}"
                                  f" with a real value of {pband_tracker['pband_abs_max']:.2f}, which is greater than"
                                  f" the relative and absolute specifications of {value[1]:.2f} and "
                                  f"{value[2]:.2f} respectively.")
        results_dict[band] = (float(round(pband_tracker['pband_dbc_max'], 2)),
                              float(round(pband_tracker['pband_abs_max'], 2)), req, passed)

    return results_dict


def psd_welch(tx_data: NDArray[complex64],
              sn0: int, snmax: int, sample_rate: float,
              onesided: bool = False,
              window: str = "hann",
              nperseg: int = 32768):
    """
    Creates a matplotlit plt of the PSD of the passed tx_data via Welch's Method.

    :param tx_data: 2-D array of M bursts of data, arranged as burst per row, stored as complex64 or a single 1 burst,
     stored in a 1-D array of complex64 data
    :type tx_data: NDArray[complex64]
    :param sn0: The SN0 index of the burst data
    :type sn0: int
    :param snmax: The SNmax index of the burst data
    :type snmax: int
    :param sample_rate: The sample rate of the passed data
    :type sample_rate: float
    :param onesided: Wether a onesided or two sided (complex) PSD plot is generated
    :type onesided: bool
    :param window: The type of window to use, must be compatible with scipy getwindow() method
    :type window: str
    :param nperseg: The number of samples per segment for Welch's method, should be a 2^N value, and result in a segment
     time length that is at least 4 times that of the total data lenght to ensure stable PSD estimates, controls the
     frequency resolution of the resulting PSD
    :type nperseg: int
    """

    if tx_data.ndim == 1:
        tx_data = tx_data.reshape(1, -1)
    n_len = tx_data.shape[1]
    tseg = (1/sample_rate) * nperseg  # The duration of each segment of the periodogram
    if 4*tseg > n_len*(1/sample_rate):
        warnings.warn(f"\n PSD via Welch's method was called with: nperseg of {nperseg} with sample rate: "
                      f"{sample_rate} Hz resulting in a segment period of {(tseg*1000):.3f} ms however, data length is "
                      f"{n_len*(1/sample_rate)*1000:.2f} ms, (<4:1), therefore PSD estimates may be unstable")
    psd_accum = None
    freq_list = zeros(1, dtype=float64)
    for r in range(tx_data.shape[0]):
        f, tx_amp = welch(tx_data[r, sn0:snmax], fs=sample_rate,
                          window=window, nperseg=nperseg,
                          noverlap=int(nperseg//2), return_onesided=onesided)
        tx_power = tx_amp / 50.0
        if psd_accum is None:
            psd_accum = tx_power
            freq_list = f
        else:
            psd_accum += tx_power

    tx_power_mean = psd_accum / tx_data.shape[0]
    tx_dbm_density = 10*log10(tx_power_mean + 1e-30) + 30

    sort_index = argsort(freq_list)  # Regular return is not order from increasing when doubled sided

    plt.figure()  # type: ignore
    plt.plot(freq_list[sort_index], tx_dbm_density[sort_index])  # type: ignore
    plt.grid()  # type: ignore
    plt.xlabel("Frequency (Hz)")  # type: ignore
    plt.ylabel("PSD (dBm/Hz) into 50ohm")  # type: ignore
    plt.title(f"Welch PSD of tx_data, RBW: {(1/tseg):.2f}")  # type: ignore
    plt.show()  # type: ignore
