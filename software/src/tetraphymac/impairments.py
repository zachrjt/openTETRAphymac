"""
impairments.py contains common functions for both the transmitter.py and receiver.py end simulations to simulate
impairments such as phase noise, AWGN, and non-linearities
"""

import bisect
from typing import Tuple

from numpy import abs as np_abs, ceil, complex128, cos, empty, exp, float64, log10, ones_like, pi, sqrt, sum as np_sum
from numpy import zeros
from numpy.fft import fftshift as np_fftshift, irfft as np_irfft, rfft as np_rfft, rfftfreq as np_rfftfreq
from numpy.random import PCG64, Generator
from numpy.typing import NDArray

from scipy.signal import remez as sp_remez, upfirdn as sp_upfirdn
from scipy.signal.windows import blackmanharris

from .constants import TETRA_SYMBOL_RATE, TIMESLOT_BIT_LENGTH

PN_SIMULATION_CROSS_OVER_POINTS = (2.0E3, 8.0E3)  # The filter cross over points for the low and high band FIR's

###################################################################################################


def loglog_interpolate(f_mask: list[float64] | NDArray[float64], dbc_mask: list[float64] | NDArray[float64],
                       f_points: NDArray[float64], delta_f: float64 | float) -> NDArray[float64]:
    """
    Performs log-log interpolation, log y values and using log(x) values, and extrapolation using a set of x and y
    points to calculate the y values (in dBx) for the passed f_points x array evaluation points.

    For any f_point below delta_f, it uses the value of delta_f and "holds". If delta_f is below f_mask, it extrapolates
    the y value of delta_f by extending the log-log slope from the points y1, y2, x1, x2 = dbc_mask[0], dbc_mask[1],
    f_mask[0], f_mask[1]. If f_point is equal to f_mask[0], holds dbc_mask[0] for any f_point <= delta_f.

    CAUTION if delta_f > dbc_mask[0], it
    interpolates the expected y value using y1, y2, x1, x2 = dbc_mask[0], dbc_mask[1], f_mask[0], f_mask[1]
    and holds it for f_point < delta_f.

    For any f_points past f_mask[-1], the value is held constant from dbc_mask[-1].

    Returns array of y values in dBx corresponding to the points from f_points.

    :param f_mask: Frequency points that coorespond to the dbc_mask values
    :type f_mask: list[float64] | NDArray[float64]
    :param dbc_mask: dBx (dBc) values related to the frequency points in f_mask used to interpolate/extrapolate values
    for the evaluation points in f_points
    :type dbc_mask: list[float64] | NDArray[float64]
    :param f_points: The frequency points at which to interpolate/extrapolate y values from using the mask dbc and f
    values.
    :type f_points: list[float64] | NDArray[float64]
    :param delta_f: The minimum frequency point to calculate and hold from, typically equal to the minimum resolvable
    frequency of the FFT/filter being useds such as (f-sampl/n-points), f_points < delta_f, hold this point's y-value
    :type delta_f: float64 | float
    :return: Returns interpolated/extrapolated/held values of y calculated from f_mask/dbc_mask for points in f_points
    :rtype: NDArray[float64]
    """
    # 1. Generate empty list of y values
    output_log_vals = zeros(f_points.size)
    # 2. Evaluate the initial slope for extrapolation of points below f_mask[0]
    init_slope = (dbc_mask[1] - dbc_mask[0]) / (log10(f_mask[1]/f_mask[0]))
    init_intercept = dbc_mask[1] - init_slope*log10(f_mask[1])
    # 3. Iterate over the interpolation grid and evaluate each point
    for n, offset in enumerate(f_points):
        if offset <= f_mask[0]:
            # We are at an offset lower than mask, hold flat from zero to delta-f to the low resolvable freq.
            if offset <= delta_f:
                output_log_vals[n] = init_intercept + init_slope*log10(delta_f)
            else:
                output_log_vals[n] = init_intercept + init_slope*log10(offset)
        elif f_mask[0] < offset < f_mask[-1]:
            # Intra-points interpolation mode
            rh_index = bisect.bisect_right(f_mask, offset)
            if rh_index == len(f_mask):
                output_log_vals[n] = dbc_mask[-1]
            else:
                slope_m = (dbc_mask[rh_index] - dbc_mask[rh_index-1]) / (log10(f_mask[rh_index]/f_mask[rh_index-1]))
                yintercept_a = dbc_mask[rh_index] - slope_m*log10(f_mask[rh_index])
                output_log_vals[n] = yintercept_a + slope_m*log10(offset)
        else:
            output_log_vals[n] = dbc_mask[-1]
    return output_log_vals


def gen_phase_noise_mask_fir(mask: dict[str, Tuple[float | float64, float | float64]], f_s: float64 | float,
                             low_factor: int = 150, high_factor: int = 4,
                             low_n_h: int = 8192, high_n_h: int = 2048):
    """
    Generates two FIR filter coefficent arrays, informally: low and high, that each approxmate the spectral shape of the
    passed in the low [0, 8000] and high [2000, 1E6] regions of the passed phase noise mask.

    The FIR filters are designed to work at their respective sampling rates: f_high = f_s // high_factor and
    f_low = f_s // low_factor, with low_n_h and high_n_h number of taps.

    A cross over region of 2000 - 8000 Hz is hardcoded to allow for greater dynamic range for each range of the mask.
    A blackmanharris window is used to truncate the FIR impulse response and minimize stopband oscilations to prevent
    stopband noise that passes through from exceeding the other filter's own mask curves.

    Note: low_factor, high_factor are selected carefully such that they are factors of f_s and the number of points in
    a single burst at simulation rate: (255x64x10) = 116,3200 as well as being higher enough such that the resulting
    rate falls outside the respective low/high fir band target area.

    Note: low_n_h and high_n_h are selected carefully as to ensure the minimum frequency resolution (f_s/X_factor)/x_n_h
    is small compared to the fir region lower end and be a 2**N value.

    :param mask: Target SSB phase noise mask dict that contains tuples of (frequency offset in Hz, ssb PSD in dBc/Hz)
    :type mask: dict[str, Tuple[float, float]] Example: {"10Hz": (10.0, -101.0), ...} where 10.0 is the frequency in Hz
    and -101.0 is the single sided PSD in units of dBc/Hz
    :param f_s: Phase Noise simulation frequency in Hz or samples/sec
    for the evaluation points in f_points
    :type f_s: float64 | float
    :param low_factor: The decimation factor of the low band fir section compared to f_s, factor of f_s and 116,320
    :type low_factor: int
    :param high_factor: The decimation factor of the high band fir section compared to f_s, factor of f_s and 116,320
    :type high_factor: int
    :param low_n_h: The number of taps in the low band FIR, must be 2**N value.
    :type low_n_h: int
    :param high_n_h: The number of taps in the high band FIR, must be 2**N value.
    :type high_n_h: int
    :return: Returns interpolated/extrapolated/held values of y calculated from f_mask/dbc_mask for points in f_points
    :rtype: Tuple[NDArray[float64], NDArray[float64]] in form (low_band_fir, high_band_fir)
    """
    # a. Sort mask and verify that it does not exceed bounds of [10hz, fs/2)
    sorted_mask = list(sorted(mask.values(), key=lambda a: a[0]))
    mask_dbc = [float64(y[1]) for y in sorted_mask]
    mask_f = [float64(x[0]) for x in sorted_mask]

    low_fs = f_s/low_factor
    high_fs = f_s/high_factor

    if mask_f[0] < 10.0:
        raise ValueError(f"Lowest SSB Phase Noise mask offset: {sorted_mask[0][0]} is below 10.0 Hz")
    if mask_f[-1] > (f_s/2):
        raise ValueError(f"Highest SSB Phase Noise mask offset:{sorted_mask[-1][0]} is greater than Nyquist:"
                         f" {f_s/2}")

    # b. Verify n_h, factors, and sample rates meet requirements
    if f_s % low_factor != 0 or ((f_s / TETRA_SYMBOL_RATE) * (TIMESLOT_BIT_LENGTH / 2)) % low_factor != 0:
        raise ValueError(f"Low Factor of: {low_factor} is not factor both Simulation Sample Rate: {f_s} and/or number"
                         f" of timeslot points at simulation rate:"
                         f"{((f_s / TETRA_SYMBOL_RATE) * (TIMESLOT_BIT_LENGTH / 2))}")

    if low_fs/low_n_h > mask_f[0]:
        raise RuntimeError(f"Low Band minimum resolvable frequency of {low_fs/low_n_h:.3f} Hz is greater than"
                           f" minimum mask frequency point: {mask_dbc[0]:.3f} Hz")
    if PN_SIMULATION_CROSS_OVER_POINTS[1]*1.25 > low_fs/2:
        raise RuntimeError(f"Low Band nyquist sample rate of {low_fs/2:.3f} Hz is too close (within 25%) of highest low"
                           f" band mask frequency point: {PN_SIMULATION_CROSS_OVER_POINTS[1]:.3f} Hz")
    if low_n_h % 2 != 0:
        raise ValueError(f"Low Band FIR Filter n-taps: {low_n_h}, is not of type 2**N")

    if f_s % high_factor != 0 or ((f_s / TETRA_SYMBOL_RATE) * (TIMESLOT_BIT_LENGTH / 2)) % high_factor != 0:
        raise ValueError(f"High Factor of: {low_factor} is not factor both Simulation Sample Rate: {f_s} and/or number"
                         f" of timeslot points at simulation rate:"
                         f" {((f_s / TETRA_SYMBOL_RATE) * (TIMESLOT_BIT_LENGTH / 2))}")
    if high_fs/high_n_h > PN_SIMULATION_CROSS_OVER_POINTS[0]:
        raise RuntimeError(f"High Band minimum resolvable frequency of {low_fs/low_n_h:.3f} Hz is greater than"
                           f" minimum high a band frequency point: {PN_SIMULATION_CROSS_OVER_POINTS[0]} Hz")
    if high_fs/2 < 1.25*mask_dbc[-1]:
        raise RuntimeError(f"High band nyquist sample rate of {high_fs/2:.3f} is too close (within 25%) of highest high"
                           f" band frequency point: {mask_dbc[-1]}")
    if high_n_h % 2 != 0:
        raise ValueError(f"Low Band FIR Filter n-taps: {high_n_h}, is not of type 2**N")

    # 1. Generate interpolation grid for upper and lower bands
    low_f_points = np_rfftfreq(low_n_h, 1/low_fs).astype(float64)
    high_f_points = np_rfftfreq(high_n_h, 1/high_fs).astype(float64)

    log_interpolated_low = loglog_interpolate(mask_f, mask_dbc, low_f_points, mask_f[0])
    log_interpolated_high = loglog_interpolate(mask_f, mask_dbc, high_f_points, high_fs/high_n_h)

    # 2. Generate and apply weighting function to handle crossover
    f1_cross = float64(PN_SIMULATION_CROSS_OVER_POINTS[0])
    f2_cross = float64(PN_SIMULATION_CROSS_OVER_POINTS[1])

    def _generate_weight_vec(f1: float64, f2: float64, f_grid: NDArray[float64]) -> NDArray[float64]:
        """
        Helper function generates cosine^2 weighting function on log(f) scale used to blend together low band and high
        band FIR filter regions using two cross over points, calculates weights on a passed grid of frequency points:
        """
        w_low = ones_like(f_grid)
        w_low[f_grid >= f2] = 0
        taper_indices = (f_grid > f1) & (f_grid < f2)

        u = (log10(f_grid[taper_indices]) - log10(f1))/(log10(f2) - log10(f1))
        w_low[taper_indices] = cos((pi/2)*u)**2
        return w_low

    low_power_weight = _generate_weight_vec(f1_cross, f2_cross, low_f_points)
    high_power_weight = 1 - _generate_weight_vec(f1_cross, f2_cross, high_f_points)

    linear_weighted_low = low_power_weight * (10**(log_interpolated_low/10))
    linear_weighted_high = high_power_weight * (10**(log_interpolated_high/10))

    def _generate_fir_coef(psd: NDArray[float64], n_h: int, fs: float64 | float) -> NDArray[float64]:
        """
        Helper function to generates both low and high band fir coefficents using the resulting psd shape
        with n_h number of points. Applies a blackmanharris window function for truncation to greatly improve stopband
        attenuation, compensates for window and conversion losses in terms of power since input signal is AWGN

        Returns n_h number of filter coefficents.
        """
        h_target = sqrt(psd * fs)
        h = np_irfft(h_target, n=n_h)
        h = np_fftshift(h)
        # Apply window
        w = blackmanharris(n_h)
        h *= w

        # Compensate for window loss
        h_actual = np_rfft(h, n=n_h)

        scale = sqrt(
            np_sum(np_abs(h_target)**2) /
            np_sum(np_abs(h_actual)**2)
        )
        h *= scale
        return h

    h_low = _generate_fir_coef(2 * linear_weighted_low, low_n_h, low_fs)
    h_high = _generate_fir_coef(2 * linear_weighted_high, high_n_h, high_fs)
    return h_low, h_high


class ColouredNoiseGenerator:
    """
    ColouredNoiseGenerator is a generic class used to handle generating coloured noise from white/AWGN noise using a
    FIR filter.

    Its' only method is next(n_samp), which returns n_samples of coloured noise by sampling white noise and filtering
    it through a FIR filter set during initialzation. It can be used for low, high, or mid band filtering for multi-band
    interpolated phase noise simulation.
    """
    def __init__(self, h_coef: NDArray[float64], nfft: int, rng_gen: Generator):
        """
        Initilizes the ColouredNoiseGenerator instance, does not perform any "warmup" of FIR memory, expects this to be
        handled externally by using it's .next(...) method.

        :param self: ColouredNoiseGenerator instance
        :param h_coef: FIR tap coefficents used for generating coloured noise
        :type h_coef: NDArray[float64]
        :param nfft: FIR tap coefficents used for generating coloured noise
        :type nfft: int, must be a value of 2**N
        :param rng_gen: FIR tap coefficents used for generating coloured noise
        :type rng_gen: Generator, a numpy.random.Generator instance of any type, PCG64 recommended
        """
        self.h_coef = h_coef
        self.h_len = self.h_coef.size
        self.rng_gen = rng_gen
        if nfft <= self.h_len:
            raise ValueError(f"FFT Length, {nfft}, for Coloured Noise Generation is <= Length of Colour Noise FIR:"
                             f" {self.h_len}, recommend 4x NFFT length compared to FIR tap number for FFT efficency")
        if nfft % 2 != 0:
            raise ValueError(f"Coloured Noise Generator n-FFT: {nfft}, is not of 2**N type.")
        self.nfft = nfft
        self.h = np_rfft(self.h_coef, n=self.nfft)
        self.hist = zeros(self.h_len - 1, dtype=float64)

        self.max_block_size = self.nfft - (self.h_len - 1)
        if self.max_block_size <= 0:
            raise RuntimeError(f"nFFT size is too small, filter size: {self.h_len}, fft size is {self.nfft}")
        self.x = empty(self.nfft, dtype=float64)
        self.x_f = empty((self.nfft // 2) + 1, dtype=complex128)
        self.y = empty(self.nfft, dtype=float64)

    def next(self, n_samp: int) -> NDArray[float64]:
        """
        Generate n_samp number of coloured noise samples by filtering white noise through the instances FIR filter set
        during initialization.

        :param n_samp: Number of coloured noise samples to generate at the sample rate used to generate the FIR filter
        :type n_samp: int
        :return: Returns the coloured noise with one-sided PSD matching that of the FIR filter self.h_coef
        :rtype: NDArray[float64]
        """
        out = empty(n_samp, dtype=float64)
        i = 0
        while i < n_samp:
            # Determine size of block to handle
            m_block_size = min(self.max_block_size, n_samp - i)
            # 1. generate M number of white noise samples
            w_new = self.rng_gen.standard_normal(m_block_size).astype(float64)
            # 2. Prepend previous awgn samples which are in FIR memory to M current samples
            self.x[:self.h_len-1] = self.hist[:]
            self.x[self.h_len-1:self.h_len-1+m_block_size] = w_new
            self.x[self.h_len-1+m_block_size:] = 0  # if (N-1) < nfft - (L-1), remaining indexes set to zero
            # Update hist array for next call
            self.hist[:] = self.x[m_block_size:m_block_size + self.h_len - 1]
            # 3. Convert to frequency domain
            self.x_f = np_rfft(self.x, n=self.nfft)
            # 4. Multiply samples+memory with filter and return to time domain
            self.y = np_irfft(self.x_f * self.h, n=self.nfft)
            # 5. Discard prepended history to isolate useful part, recall useful part is only  (L + M -1)
            out[i: i + m_block_size] = self.y[self.hist.size:self.hist.size + m_block_size]
            i += m_block_size
        return out


class PNBandGenerator:
    """
    PNBandGenerator, phase-noise-band-generator, is a generic class used to handle generating coloured noise at
    simulation rate self.f_sim, by generating at a lower self.f_sim/self.sample_rate_factor rate filtering with the
    h_pn_coef FIR coefficents passed at initialization, then upsampling at self.sample_rate_factor and filtering with
    interpolation FIR self.h_interpolate_coef.

    One method, .warmup(n_samp), should be used after initalization to fill the coloured noise fir and interpolation fir
    memories with valid data and push out filter transients, the recommend minimum number of points to warmup is:
    len(h_pn_coef)*2.


    Its' primary method is generate(n_samp), which generates n_samp * self.sample_rate_factor number of coloured noise
    samples at self.f_sim[ulation] rate.
    """
    def __init__(self, f_sim: float64 | float, sample_rate_factor: int, h_pn_coef: NDArray[float64],
                 h_interpolate_coef: NDArray[float64], rng_gen: Generator | None):
        """
        Initilizes the PNBandGenerator instance, does not perform any "warmup" of FIR memory, expects this to be
        handled externally by using it's warmup method.

        :param self: PNBandGenerator instance
        :param f_sim: Simulation rate in units of Hz or samples/s
        :type f_sim: float64 | float
        :param sample_rate_factor: The upsample factor up to to f_sim, must be factor of f_sim and 116,320
        :type sample_rate_factor: int
        :param h_pn_coef: FIR tap coefficents used for generating coloured noise at lower rate
        :type h_pn_coef: NDArray[float64]
        :param h_interpolate_coef: FIR tap coefficents used interpolate coloured noise samples by sample_rate_factor to
        f_sim rate
        :type h_interpolate_coef: NDArray[float64]
        :param rng_gen: FIR tap coefficents used for generating coloured noise
        :type rng_gen: Generator, a numpy.random.Generator instance of any type, PCG64 recommended
        """
        self.fs_sig = f_sim
        self.factor = sample_rate_factor
        self.fs = f_sim / sample_rate_factor
        self.rng_gen = Generator(PCG64()) if rng_gen is None else rng_gen

        self.pn_gen = ColouredNoiseGenerator(h_pn_coef, (h_pn_coef.size * 4), self.rng_gen)
        self.upsample_fir_h = (self.factor *
                               h_interpolate_coef *
                               (1/np_sum(h_interpolate_coef)))
        self.h_len = int(ceil((self.upsample_fir_h.size-1) / self.factor))
        self.upsample_fir_mem = zeros(self.h_len, dtype=float64)

    def warmup(self, n_samp: int) -> NDArray[float64]:
        """
        Generate n_samp number of coloured noise samples to warm up the underlying ColouredNoiseGenerator PSD FIR and
        the self.upsample_fir_mem FIR memory. Not performing a warmup or not warmuping up a sufficent amount, can result
        in coloured noise samples out of .generate(...) method not conforming to the expected PSD shape due to filter
        startup transisents

        Note: The recommend minimum number of points to warmup is: len(h_pn_coef)*2.

        :param n_samp: Number of coloured noise samples to generate at the base rate of self.f_sim/self.factor
        :type n_samp: int
        :return: Returns self.factor * n_samp samples of coloured noise at self.f_sim rate, typically discarded.
        :rtype: NDArray[float64]
        """
        temp = empty(shape=(n_samp + self.h_len), dtype=float64)
        # Warmup noise generator fir memory
        temp[self.h_len:] = self.pn_gen.next(n_samp)
        self.upsample_fir_mem = temp[-self.h_len:]
        y = sp_upfirdn(self.upsample_fir_h, temp, up=self.factor)
        return y[self.h_len*self.factor: self.h_len*self.factor + n_samp*self.factor]

    def generate(self, n_samp: int) -> NDArray[float64]:
        """
        Generate n_samp number of coloured noise samples by filtering white noise through a FIR PSD mask,
        self.ColouredNoiseGenerator.h_coef, and upsampling by a factor self.factor and then filtering through a
        interpolation filter, self.upsample_fir_h, to get to sampling rate: self.f_sim

        :param n_samp: Number of coloured noise samples to generate at the base rate of self.f_sim/self.factor
        :type n_samp: int
        :return: Returns self.factor * n_samp samples of coloured noise at self.f_sim rate.
        :rtype: NDArray[float64]
        """
        samples = empty(shape=(n_samp + self.h_len), dtype=float64)
        samples[:self.h_len] = self.upsample_fir_mem
        samples[self.h_len:] = self.pn_gen.next(n_samp)
        # Using scipy upfirdn as polyphase filter to greatly (150x) reduce number of calculations compared to standard
        # filtering or FFT filtering for these filter sizes
        pn_samples = sp_upfirdn(self.upsample_fir_h, samples, up=self.factor)
        self.upsample_fir_mem = samples[-self.h_len:]
        return pn_samples[self.h_len*self.factor: self.h_len*self.factor + n_samp*self.factor]


class PhaseNoiseSimulator:
    """
    PhaseNoiseSimulator, is a generic class used to handle simulation of phase noise that meets/follows a passed SSB
    PSD mask in units of dBc/Hz. Used for simulating the cumulative phase noise caused by tx or rx hardware sections.

    It offers highly efficent, in both memory and time, simulation of a phase noise meeting a target mask by splitting
    a target mask into two bands based on frequency offset, low and high, and simulating each band a lower rate and then
    upsampling and adding together to reach the full simulation rate.

    Note: A result of the technique used is that low and high frequency offsets (below 2000 and above 8000) are
    indepedent.

    Caution: Since SSB masks contain large dynamic range (10's of db across decades of frequencies),
    PhaseNoiseSimulator works best when passing ssb masks whose points are quasi-decade in frequency, with slopes of
    1/f^alpha as measured in log(f), such as -40db/decade, -38db/decade, etc. Plateaus may not be able to be modelled at
    frequencies below 10kHz, and frequencies below 10Hz are not allowed, and frequencies above 1E6 are not recommended.

    Its' primary method is apply_phase_noise(...), which using the parameters set at initialization, will generate and
    apply the resulting phase noise to an passed input signal without using the small angle approxmate, i.e, as
    x(t) * exp(1j * phi(t)), where phi(t) is the coloured noise random process following the target SSB PSD mask.
    """
    def __init__(self, f_sim: int, ssb_mask: dict[str, Tuple[float, float]],
                 rng_gen_low: Generator | None = None, rng_gen_high: Generator | None = None):
        """
        Initilizes the PhaseNoiseSimulator instance, performs warmup to fill the interpolation and coloured noise
        generation fir memories and remove filter startup transisent.

        :param self: PhaseNoiseSimulator instance
        :param f_sim: Simulation rate in units of Hz or samples/s
        :type f_sim: float64 | float
        :param ssb_mask: Target SSB phase noise mask dict that contains tuples of (freq. offset in Hz, ssb in dBc/Hz)
        :type ssb_mask: dict[str, Tuple[float, float]] Example: {"10Hz": (10.0, -101.0), ...} where 10.0 is the freq.
        in units of Hz and -101.0 is the single sided PSD in units of dBc/Hz
        :param rng_gen_low: FIR tap coefficents used for generating coloured noise
        :type rng_gen_low: Generator, a numpy.random.Generator instance of any type, PCG64 recommended
        :param rng_gen_high: FIR tap coefficents used for generating coloured noise
        :type rng_gen_high: Generator, a numpy.random.Generator instance of any type, PCG64 recommended
        """
        self.fs_sig = float64(f_sim)
        self.low_factor = 150
        self.high_factor = 4
        self.low_n_h = 2**14
        self.high_n_h = 2**12

        # Create FIR coefficents for band generators and unsampling filters
        self.low_h_coef, self.high_h_coef = gen_phase_noise_mask_fir(ssb_mask, self.fs_sig,
                                                                     self.low_factor, self.high_factor,
                                                                     self.low_n_h, self.high_n_h)

        self.upsample_fir_low_h = sp_remez(4096, [0, 12E3, 40E3, self.fs_sig/2], [1, 0], fs=self.fs_sig)
        self.upsample_fir_high_h = sp_remez(512, [0, 1E6, 1.2E6, self.fs_sig/2], [1, 0], fs=self.fs_sig)

        # Initilize and warmup low and high band phase noise generators
        self.low_band_gen = PNBandGenerator(self.fs_sig, self.low_factor, self.low_h_coef,
                                            self.upsample_fir_low_h, rng_gen_low)
        self.low_band_gen.warmup(5*self.low_h_coef.size)

        # print(f"h_low: {np_sum(self.low_h_coef**2)} vs sample variance {np_var(self.low_band_gen.generate(2**10))}")

        self.high_band_gen = PNBandGenerator(self.fs_sig, self.high_factor, self.high_h_coef,
                                             self.upsample_fir_high_h, rng_gen_high)
        self.high_band_gen.warmup(5*self.high_h_coef.size)

    def apply_phase_noise(self, signal_block: NDArray[complex128]) -> NDArray[complex128]:
        """
        Generate n_samp number of coloured noise samples by filtering white noise through a FIR PSD mask,
        self.ColouredNoiseGenerator.h_coef, and upsampling by a factor self.factor and then filtering through a
        interpolation filter, self.upsample_fir_h, to get to sampling rate: self.f_sim

        :param signal_block: Input signal, whose length must have a factor of 255 and 64
        :type signal_block: int
        :return: Returns signal_block * exp(1j * phi(f)), where phi(f) is a coloured noise approximating the target mask
        :rtype: NDArray[complex128]
        """
        # generate phase noise samples from each block, then add them
        n_signal = signal_block.size
        if n_signal % (255*64) != 0:  # Verify we are working in blocks at a time
            raise ValueError(f"Passed number signal samples: {n_signal} does not align with the size of timeslots"
                             f"'bursts: {255*64}, Recommended to check and correct filter delay handling")

        pn_noise = self.low_band_gen.generate(n_signal//self.low_factor)
        pn_noise += self.high_band_gen.generate(n_signal//self.high_factor)

        # apply phase noise to signal, using vector multiplication
        return (signal_block.reshape(-1) * exp(1j * pn_noise)).reshape(signal_block.shape)
