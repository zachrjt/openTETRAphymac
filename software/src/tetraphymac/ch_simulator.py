"""
ch_simulator.py contains functions and classes used to simulate channels between a MS/BS transmitter and a MS/BS
reciever.
"""
from typing import Tuple, Literal
from abc import ABC, abstractmethod

from numpy import round as np_round, zeros, float64, complex128, pi, arange, cos, sin, sum as np_sum, empty, int64, \
                  sqrt as np_sqrt, isclose, exp, full, concatenate, ceil, allclose as np_allclose, floor, \
                  linspace, any as np_any
from numpy.typing import NDArray
from numpy.random import SeedSequence, Generator, PCG64

from scipy.signal import remez as sp_remez, upfirdn as sp_upfirdn, lfilter
from scipy.constants import c as C_SPEED_OF_LIGHT

from .constants import TETRA_PROPAGATION_MODELS, TetraPropagationModels, PropagationTapParameters, \
                       TetraTapGainProcess, PropagationModelParameters, \
                       TETRA_DEFAULT_MODEL_VELOCITIES_KPH, TETRA_FADING_SIMULATION_RATE, StreamPosition
from .constants import TETRA_TX_SIMULATION_SAMPLE_RATE


###################################################################################################
class ComplexGainProcess(ABC):
    """
    Abstract class designed to sit in for any possible complex gain process used with PropgationTaps for a Propagation
    Model.

    Only abstract method is next(), which generates the required number of samples, so that a PropagationTap can apply
    them, if the process can be generated deterministically, then repeatable control wether or not subsequent calls
    will generate the same complex gain values as the current call.
    """

    @abstractmethod
    def next(self, n_samples: int, repeatable: bool = False) -> NDArray[complex128]:
        """
        Base abstract method that generates and returns n_samples of complex gain for the implemented process
        """
        raise NotImplementedError

    @abstractmethod
    def null_advance(self, n_samples: int, repeatable: bool = False) -> None:
        """
        Base abstract method that advances the internal determinstic process states without performing computations
        """
        raise NotImplementedError


class StaticProcess(ComplexGainProcess):
    """
    Static process is non-stocastic process that represents a channel where the LOS is present and thus only undergoes
    a doppler shift if doppler shift is passed to it.
    """
    _sample_idx: int64
    fd: float
    wd: float
    doppler_yes: bool

    def __init__(self, f_sim: int, f_doppler: float):
        """
        Initializes the StaticProcess for fading for generating complex gain values at rate: f_sim. If f_doppler is
        non-zero, then the result gain output has unity magnitude, but cooresponds to a complex exponential, i.e
        frequency shift by f_doppler.

        :param f_sim: The simulation rate of the channel simulation, expected to be TRANSMIT_SIMULATION_SAMPLE_RATE
        :type f_sim: int
        :param f_doppler: The doppler frequency shift in hertz
        :type f_doppler: float
        """
        # Might handle variable simulation rates for the fading sim in the future but unlikely
        if f_sim != TETRA_TX_SIMULATION_SAMPLE_RATE:
            raise ValueError(f"Passed simulation sample rate: {f_sim} is not the expected"
                             f" rate of {TETRA_TX_SIMULATION_SAMPLE_RATE}")
        if isclose(f_doppler, 0):
            self.doppler_yes = False
            self.fd = 0.0
            self.wd = 0.0
        else:
            self.fd = f_doppler
            self.wd = f_doppler * 2 * pi
            self.cos_pi_wd_tsim = self.wd * 2 * pi * (1/f_sim)
        self.f_sim = f_sim
        self._sample_idx = int64(0)

    def next(self, n_samples: int, repeatable: bool = False) -> NDArray[complex128]:
        """
        Generates LOS static complex gain values for length of n_samples, if f_doppler was passed, then the complex gain
        generated is simply the delta dirac(f_doppler), i.e. a frequency shift

        Note: because of interpolation, finite leakage of the doppler spectrum occurs at frequnecy multiples of
        TETRA_FADING_SIMULATION_RATE (80khz), rejection of leakage from interpolation is min. 190dB within 900khz,
        and 150dB at most at n*960khz spectral images.

        :param n_samples: Number of complex gain samples to generate at rate f_sim
        :type n_samples: int
        :param repeatable: If True, does not increment sample counter of gain generation, therefore for the next same
        sized input signal repeats the same complex gain profile, repeats for next call as long as set to True
        :type repeatable: bool = False
        :return: Returns n_samples number of complex gain for the Rayleigh process at f_sim
        :rtype: NDArray[complex128]
        """
        if self.doppler_yes:
            # Generate doppler shift array
            t = arange(self._sample_idx, n_samples, dtype=float64)
            out = exp(1j * (self.cos_pi_wd_tsim * t)).astype(complex128)
        else:
            out = full(shape=n_samples, fill_value=(1.0 + 0.0j), dtype=complex128)

        if not repeatable:
            self._sample_idx += int64(n_samples)

        return out

    def null_advance(self, n_samples: int, repeatable: bool = False) -> None:
        """
        null_advance advances the internal deterministic states for gain generation without computing gain values
        this reduces computation for non-tx periods whilst maintaining fading statistics w.r.t to time.

        If repeatable = True, does not increment internal state more of a dummy state for completeness of PropagationTap
        implementation completeness.

        :param n_samples: Amount of samples to increment internal deterministic states for gain generation
        :type n_samples: int
        :param repeatable: If True, does not increment sample counter of gain generation
        :type repeatable: bool = False
        """
        self._sample_idx += 0 if repeatable else int64(n_samples)


class RayleighProcess(ComplexGainProcess):
    """
    `RayleighProcess` is a class used to handle generation of Rayleigh fading process for PropagationTaps.
    It handles indepedently generating samples of a complex gain process using a deterministic sum-of-sinusoids method
    that has good agreeance with the expected statistics using a method proposed by Zheng and Xiao.

    It interpolates the complex gain samples up to full simulation rate: `f_sim` from a base rate of
    `TETRA_FADING_SIMULATION_RATE` with high (>180dB within an offset of 12 * `TETRA_FADING_SIMULATION_RATE`) rejection
    of spectral images created during upsampling.

    The resulting statistics agree well the base rate sampled data and the expected statistics for a Rayleigh fading
    enviroment interms of i.i.d. I and Q components, gain magnitude/enevelope, gain power enevelope, autocorrelation,
    and time delay w.r.t. to an indepedently generated sequence generated with the same base seed sequence.


    Its' primary method is .next(`n_samples`), generates and interpolates a n_samples of complex gain samples at `f_sim`
    rate and returns the complex gain/fade samples. It can also be configured to allow
    for subsequent repitition of the sample complex gain profile, easing scenario testing by remove the need to
    initialize multiple propagation taps/models with the same seed.
    """
    m: int
    _sample_idx: int64
    _sample_frac_idx: int
    upsample_factor: int
    fd: float
    wd: float
    f_fade_sim: int
    seed_seq: SeedSequence

    _phi_gen: Generator
    _psi_gen: Generator
    _theta_gen: Generator
    phi_n: NDArray[float64]
    psi_n: NDArray[float64]
    theta_n: NDArray[float64]

    alpha_n: NDArray[float64]
    cos_phase_coef: NDArray[float64]
    sin_phase_coef: NDArray[float64]

    interpolator_mem_valid: bool
    _h1_mem: NDArray[complex128]
    _interp_buffer: NDArray[complex128]
    buffer_len: int

    def __init__(self, f_sim: int, f_doppler: float, seed_seq: SeedSequence | None = None, m_order: int = 264):
        """
        Initializes the RayleighProcess for generated complex gain samples at rate: f_sim, for doppler shift of
        f_doppler with an m_order number of sine/cosine generators. Uses seed_seq if passed to spawn grandchildern
        seeds to drive generators with indepdent but reproducable results.

        :param f_sim: The simulation rate of the channel simulation, expected to be TRANSMIT_SIMULATION_SAMPLE_RATE
        :type f_sim: int
        :param f_doppler: The doppler frequency shift in hertz
        :type f_doppler: float
        :param seed_seq: The numpy SeedSequence used to spawn grandchildern seeds to drive generators for the 3 random
        variables that enable the process, if None is passed, internally spawns a new SeedSequence from np.random
        :type seed_seq: SeedSequence | None
        :param m_order: Number of sinusoid generators for each of the sine/cosine generators, should be greater than 8,
        increasing the number improves statistics at cost of computational complexity, default is 264
        :type m_order: int = 264
        """
        # Handle order checks and time tracker
        self.m = m_order
        if m_order < 8:
            raise RuntimeWarning(f"RayleighFadingSimulator order of: {m_order} is not recommended due to disagreeance"
                                 " with ideal statistics, recommended M>=8")
        # Recall output is deterministic, instead of tracking phase for each of the M oscilators, we just evaluate phase
        # using the time value since numpy will wrap it and speed benefit is not really apparent here unlike hw with LUT
        self._sample_idx = int64(0)

        # Handle frequency checks
        self.wd = abs(f_doppler) * 2 * pi
        self.f_fade_sim = TETRA_FADING_SIMULATION_RATE
        # Might handle variable simulation rates for the fading sim in the future but unlikely
        if f_sim != TETRA_TX_SIMULATION_SAMPLE_RATE:
            raise ValueError(f"Passed simulation sample rate: {f_sim} is not the expected"
                             f" rate of {TETRA_TX_SIMULATION_SAMPLE_RATE}")
        self.total_upsample = 144
        self.stage_upsample = 12

        # Generate random phi, psi, theta values
        if seed_seq is None:
            self.seed_seq = SeedSequence()
        else:
            self.seed_seq = seed_seq
        # Generate grandchildren seed's
        _phi_seed, _psi_seed, _theta_seed = self.seed_seq.spawn(3)
        self._phi_gen = Generator(PCG64(_phi_seed))
        self._psi_gen = Generator(PCG64(_psi_seed))
        self._theta_gen = Generator(PCG64(_theta_seed))

        self.phi_n = self._phi_gen.uniform(low=-pi, high=pi, size=self.m)[:, None]
        self.psi_n = self._psi_gen.uniform(low=-pi, high=pi, size=self.m)[:, None]
        self.theta_n = self._theta_gen.uniform(low=-pi, high=pi, size=self.m)

        # Precalculate all constants in the oscilators
        self.alpha_n = (2*pi*(arange(1, self.m+1)) - pi + self.theta_n)/(4*self.m)
        self.cos_phase_coef = (self.wd/self.f_fade_sim) * cos(self.alpha_n)[:, None]
        self.sin_phase_coef = (self.wd/self.f_fade_sim) * sin(self.alpha_n)[:, None]

        self.interpolator_mem_valid = False

        self._initialize_filters(f_sim)

    def _initialize_filters(self, f_sim: int):
        """
        Generates interpolate filters and normalizes their DC gain to 1. Allocates filter and output buffer memory.
        Calculates total filter group delay, and then generates cooresponding samples based on the group delay such that
        when ._apply_complex_gain(signal) is called for the first time, output samples are non-transient and coorespond
        with t=0.

        :param f_sim: The simulation rate of the channel simulation, expected to be TRANSMIT_SIMULATION_SAMPLE_RATE
        :type f_sim: int
        """
        # Generate interpolation filters, and initialize interpolation memories
        f_mid_rate = f_sim // self.stage_upsample
        h1 = sp_remez(4096, [0, 17_500, 20_000, f_mid_rate/2], [1, 0], weight=[1, 100], fs=f_mid_rate)
        h1 *= (self.stage_upsample / np_sum(h1))
        self._h1_interp = h1.astype(complex128)

        h2 = sp_remez(4096, [0, 48_000, 80_000, f_sim/2], [1, 0], weight=[1, 100], fs=f_sim)
        h2 *= (self.stage_upsample / np_sum(h2))
        self._h2_interp = h2.astype(complex128)

        h3 = sp_remez(46, [0, 30_000, 900_000, f_sim/2], [1, 0], weight=[1, 100], fs=f_sim)
        h3 *= (1 / np_sum(h3))
        self._h3_cleanup = h3.astype(complex128)

        # Calculate filter delay, allocate filter memory
        self._cascade_group_delay = int(((self._h1_interp.size - 1) / 2) * self.stage_upsample
                                        + ((self._h2_interp.size - 1) / 2)
                                        + ((self._h3_cleanup.size - 1) / 2))

        gd_base = self._cascade_group_delay // self.total_upsample
        mem_size = (self._h1_interp.size // 2) + 2
        self._h1_mem = zeros(shape=(mem_size), dtype=complex128)

        # Compensate for group delay such that when .apply_complex_gain() is called 1st time, samples start at t=0
        # This is not strictly needed but helps if output is compared to external code with same seed and implementation
        self._sample_idx = int64(gd_base)
        # Allocate interpolation buffer, note max used size is self.upsample_factor - 1
        self._buffer = zeros(self.total_upsample - 1, dtype=complex128)
        self.buffer_len = 0
        # Generate warmup samples, next highest power of 2 compared to h1_mem size
        warmup = 1 << (self._h1_mem.size - 1).bit_length()
        warmup_samples = self._generate(warmup, (gd_base-warmup))
        self._h1_mem[:] = warmup_samples[-self._h1_mem.size:]

        self.interpolator_mem_valid = True
        self._sample_frac_idx = 0

    def _generate(self, n_samples: int, start_index: int | None = None) -> NDArray[complex128]:
        """
        Generates samples of the complex gain process that follows Rayleigh fading with no LOS components and isotropic
        scattering, yielding the classic Clark's/Jake's Doppler PSD.

        Uses the sum-of-sinusoids method, specifically the Zheng-Xiao method to produce samples at rate:
        TETRA_FADING_SIMULATION_RATE.

        Note: the resulting output is deterministic in generation, therefore start_index can be used to control when
        samples are generate. Leave start_index as default None to utilize an internal _sample_idx track that increments
        n_samples after n_samples are called, or specify if a specific start time is desired to produce the values for
        time = [start_index, start_index + n_samples) * (1/TETRA_FADING_SIMULATION_RATE)

        :param n_samples: How many samples to generate at rate TETRA_FADING_SIMULATION_RATE
        :type n_samples: int
        :param start_index: Default None, if specified as int, does not increment internal _sample_idx tracker, and
        instead generates samples for time = [start_index, start_index + n_samples) * (1/TETRA_FADING_SIMULATION_RATE)
        :type start_index: int | None = None
        :return: Returns n_samples of the complex gain process, starting at the specified start index or the object's
        _sample_idx counter.
        :rtype: NDArray[complex128]
        """
        # Z (Result) = a + jb
        # Generate phase time-increment values
        idx = (self._sample_idx if start_index is None else start_index) + arange(n_samples, dtype=float64)
        if start_index is None:
            # If we are warming up or specifing the start for some reason, dont add to the global sample index
            self._sample_idx += int64(n_samples)

        a = np_sum(cos((idx * self.cos_phase_coef) + self.phi_n), axis=0)
        b = np_sum(sin((idx * self.sin_phase_coef) + self.psi_n), axis=0)
        z = (np_sqrt(1/self.m))*(a + 1j*b)
        return z

    def _interpolate(self, fade_samples: NDArray[complex128], repeatable: bool = False):
        """
        Interpolates fade_samples generated at TETRA_FADING_SIMULATION_RATE, up to f_sim rate set during initialization,
        typically 11.52MHz.

        Does this through a 3 step process:
        1. 12x upsampling via 4096 tap polyphase upsampling
        2. 12x Upsampling via 4096 tap polyphase upsampling
        3. Cleanup filter via 46 tap FIR filter

        The result rejection of spectral images is at least 180dB for images at N * TETRA_FADING_SIMULATION_RATE
        and 150dB for images at N * 12 * TETRA_FADING_SIMULATION_RATE

        :param fade_samples: Input array of complex128 process samples sampled at TETRA_FADING_SIMULATION_RATE rate,
        which obey statisics of Rayleigh fading for the given f_doppler passed during initialization
        :type signal: NDArray[complex128]
        :param repeatable: If True, does not rewrite filter memory, thus allowing for the same gain process sample to be
        output repeatably.
        :type repeatable: bool = False
        :return: Returns input signal.size number samples multiplied by complex gain as: signal * z(t)
        :rtype: NDArray[complex128]
        """
        n_samples = fade_samples.size
        x = empty(shape=(self._h1_mem.size + n_samples), dtype=complex128)
        x[:self._h1_mem.size] = self._h1_mem
        x[self._h1_mem.size:] = fade_samples

        # Update filt memory
        if not repeatable:
            self._h1_mem = x[-self._h1_mem.size:]
        # Stage 1 interpolation
        y1 = sp_upfirdn(h=self._h1_interp, x=x, up=self.stage_upsample)
        # Stage 2 interpolation
        y2 = sp_upfirdn(h=self._h2_interp, x=y1, up=self.stage_upsample)
        # Stage 3 cleanup
        y3 = lfilter(self._h3_cleanup, [1.0], y2)

        # Remove front pad which has filter transients in it
        start = self._h1_mem.size * self.total_upsample
        stop = start + (n_samples * self.total_upsample)
        return y3[start:stop]

    def apply_complex_gain(self, signal: NDArray[complex128], repeatable: bool = False) -> NDArray[complex128]:
        """
        Wrapper for testing next() when using class on it's own.
        """
        # This is an older method for testing RayleighProcess on it's own
        out_array = self.next(signal.size, repeatable)

        return signal * out_array

    def next(self, n_samples: int, repeatable: bool = False) -> NDArray[complex128]:
        """
        Generates Rayleigh fading process complex gain via sum of sinusoids method at lower rate, then interpolates up
        to f_sim sample rate and return n_samples number worth of gain values.

        Note: because of interpolation, finite leakage of the doppler spectrum occurs at frequnecy multiples of
        TETRA_FADING_SIMULATION_RATE (80khz), rejection of leakage from interpolation is min. 190dB within 900khz,
        and 150dB at most at n*960khz spectral images.

        :param n_samples: Number of complex gain samples to generate at rate f_sim
        :type n_samples: int
        :param repeatable: If True, does not increment sample counter of gain generation, therefore for the next same
        sized input signal repeats the same complex gain profile, repeats for next call as long as set to True
        :type repeatable: bool = False
        :return: Returns n_samples number of complex gain for the Rayleigh process at f_sim
        :rtype: NDArray[complex128]
        """
        # If we our FIR interpolation memory is invalid because we call null_advance previously enough
        # Need to regenerate/warmup interpolation memory
        if not self.interpolator_mem_valid:
            # Generate warmup samples, next highest power of 2 compared to h1_mem size
            # 1. Generate warmup to fill memory of
            warmup = 1 << (self._h1_mem.size - 1).bit_length()
            warmup_samples = self._generate(warmup, (int(self._sample_idx) - warmup))
            self._h1_mem[:] = warmup_samples[-self._h1_mem.size:]

            # 2. Fill up buffer to match absolute timing
            pending_interp_samples = (-self._sample_frac_idx) % self.total_upsample
            x_temp = self._generate(1, None)
            y_temp = self._interpolate(x_temp)
            self._buffer[:pending_interp_samples] = y_temp[-pending_interp_samples:]
            self.buffer_len = pending_interp_samples

        # Determine how many base rate samples are required, taking into account leftover data in buffer
        if self.buffer_len < n_samples:
            needed_samples = max(0, n_samples - self.buffer_len)
            buffer_used = self.buffer_len
        else:
            needed_samples = 0
            buffer_used = n_samples

        # integer ceiling form, instead of int(ceil(float))
        base_samples = (needed_samples + self.total_upsample - 1) // self.total_upsample
        if base_samples == 0:
            leftover_samples = self.buffer_len - buffer_used
        else:
            leftover_samples = (base_samples * self.total_upsample) - needed_samples

        out_array = empty(shape=(n_samples), dtype=complex128)
        out_array[:buffer_used] = self._buffer[:buffer_used]

        if base_samples != 0:
            # Generate base rate samples of the process
            samples = self._generate(base_samples, (None if not repeatable else int(self._sample_idx)))
            y = self._interpolate(samples, repeatable)
            out_array[buffer_used:] = y[:needed_samples]
            if not repeatable:
                self._buffer[:leftover_samples] = y[needed_samples:]
                self.buffer_len = leftover_samples
        else:
            if not repeatable:
                self._buffer[:leftover_samples] = self._buffer[buffer_used: self.buffer_len]
                self.buffer_len = leftover_samples

        if not self.interpolator_mem_valid:
            if not repeatable:
                self._sample_frac_idx = 0
                self.interpolator_mem_valid = True
            else:
                self.buffer_len = 0
                self._sample_idx -= 1

        return out_array

    def null_advance(self, n_samples: int, repeatable: bool = False) -> None:
        """
        null_advance advances the internal deterministic states for gain generation without computing gain values
        this reduces computation for non-tx periods whilst maintaining fading statistics w.r.t to time.

        If repeatable = True, does not increment internal state more of a dummy state for completeness of PropagationTap
        implementation completeness.

        :param n_samples: Amount of samples to increment internal deterministic states for gain generation
        :type n_samples: int
        :param repeatable: If True, does not increment sample counter of gain generation
        :type repeatable: bool = False
        """
        # In this mode, we increment the sample_idx but do not generate gain_samples
        # because of this there may be a discontinuity in the interpolation filter memory when we
        # finally do call next(), so we change a state variable to track that
        if not repeatable:
            if n_samples <= self.buffer_len:
                # Throw out self._inter_buffer samples then
                self.buffer_len -= n_samples
            else:
                # FIR interpolation memory is not longer valid, need to regenerate on the next next() call
                self.interpolator_mem_valid = False

                remaining = n_samples - self.buffer_len
                self.buffer_len = 0

                fractional = self._sample_frac_idx + remaining

                self._sample_idx += int64(fractional // self.total_upsample)
                self._sample_frac_idx = fractional % self.total_upsample

##########################################################


def _generate_lagrange_delay_fir(d: float, n_order: int = 3) -> NDArray[complex128]:
    """
    Generates Lagrange delay FIR with fractional delay d, with length/number of coefficents = n_order+1.

    Note: Lagrange filters have small passbands, thus the usable bandwidth is limited to 0.10-0.15 of the sampling rate.

    :param d: Delay of desired filter, in fractional samples of the sample rate used for filtering
    :type d: float
    :param n_order: Order of lagrange filter, results in length of filter n_order + 1, default is order 3, length 4
     increasing order has very diminishing returns, even order filters posses generally worse behaviour
    :type n_order: int
    :return: Returns numerator coefficents of a Lagrange FIR with delay equal to d, in complex128 format
    :rtype: NDArray[complex128]
    """
    h_n = zeros(shape=n_order+1, dtype=complex128)

    for n in range(n_order+1):
        n_k = 1.0
        k_vals = [x for x in range(n_order+1) if x != n]
        for k in k_vals:
            n_k *= (d-k)/(n-k)
        h_n[n] = complex128(n_k)

    return h_n


def _calculate_delay_parameters(f_sim: float, delay: float,
                                n_lagrange_order: int = 3) -> Tuple[int, NDArray[complex128] | None]:
    """
    Determines the delay in terms of samples at rate: f_sim, then if the resulting delay is a fractional sample number,
    generates a Lagrange Delay fir, returns the non-filter integer delay component and the filter coefficents if there
    is a fractional part.

    :param f_sim: frequency of simulation in Hz
    :type f_sim: float
    :param delay: Delay in seconds, can be fractional
    :type delay: float
    :param n_lagrange_order: Order of lagrange filter, results in length of filter n_order + 1, default is order 3
     increasing order has very diminishing returns, even order filters posses generally worse behaviour
    :type n_lagrange_order: int
    :return: Returns the a tuple of integer delay in number of samples at f_sim, seperate from filter delay, element [0]
    and in element [1] returns either None if there is no fractional delay/filter required for the given delay and f_sim
    :rtype: Tuple[int, NDArray[complex128] | None]
    """
    if delay > 0.0:
        delay_samples = delay * f_sim
        filt_int_delay = n_lagrange_order/2

        int_delay = int(np_round(delay_samples - filt_int_delay))
        filt_frac_delay = delay_samples - int_delay - filt_int_delay

        # Fractional part of Lagrange delay FIR is limited to [-0.5, +0.5)
        if filt_frac_delay >= 0.5:
            int_delay += 1
            filt_frac_delay -= 1.0
        elif filt_frac_delay < -0.5:
            int_delay -= 1
            filt_frac_delay += 1.0

        filt_total_delay = filt_frac_delay + filt_int_delay

        h_delay_coef = _generate_lagrange_delay_fir(filt_total_delay, 3)
    else:
        int_delay = 0
        h_delay_coef = None

    return int_delay, h_delay_coef


def _calculate_doppler_shift(f_ch: float, v_kph: float, doppler_relative_to_430mhz: bool = True) -> float:
    """
    Calculates the doppler shift in Hz based on channel frequency: f_ch, and speed: v_kph. If doppler_relative_to_430mhz
    is True, default, calculates the equivalent doppler shift at 430MHz at the passed v_kph

    :param f_ch: Frequency of the RF channel in Hz.
    :type f_ch: float
    :param v_kph: Relative speed in km/h
    :type v_kph: float
    :param doppler_relative_to_430mhz: Boolean, default True, if true calculates the doppler shift if it were at 430MHz
    instead of the passed frequency f_ch, this is used in TETRA for testing when f_ch < 380 MHz or f_ch > 520 MHz
    :type doppler_relative_to_430mhz: bool = True
    :return: Returns doppler shift in Hz; normalized to 430MHz if doppler_relative_to_430mhz is True
    :rtype: float
    """
    f_ref = 430.0E6 if doppler_relative_to_430mhz else f_ch
    f_doppler = (v_kph / 3.6) / (C_SPEED_OF_LIGHT / f_ref)
    return f_doppler

###################################################################################################


class PropagationTap:
    """
    `PropagationTap` is a class used to implement the delay-gain tap paths of a `PropagationModel`. Its' defining
    parameters are: mean delay, gain process type, and doppler shift.

    It implements a powerful block-streaming `process` that can handle theoretically infinite continuous signal's
    by processing arbitrary sized (size >= `required_history`) "blocks" of data which can be singular bursts as defined
    by TETRA's physical channels, portions of bursts, or multiple bursts. It only needs to know information about the
    block in terms of is it continous with the preceeding or subsequent burst, and is it ramped-up/down at start/end.

    In handling blocks of data (parts of burst or whole bursts) it applies an arbitrary signal delay using a
    integer/sample delay line memory/buffer and a Lagrange FIR fractional delay FIR. It then calls on gain process(es)
    to generate deterministically fading process gain values for the output samples.

    PropagationTap implements optional repeatability in its' methods allowing for exact repetition of output signals
    despite a previous call with repeatable=True. Repeatable can also be used to apply the same complex-gain process to
    different signals and have the exact same fading profile without needed to externally handle saving and restoring
    states.

    As a high level summary here are the properties that its' implementation fulfills, refer to `process()`
    documentation for more details.

    -**Block invariant**:
        -process(A:B) == process(A) : process(B), thus input stream can be concatenated or divided as needed

    -**Burst invariant**:
        -Equivalent burst segmentations produce identical output.

    -**Full mode is non-destructive**:
        -Calling full() does not alter subsequent same() processing.

    -**Repeatable mode is side-effect free**:
        -Repeatable calls do not advance any persistent simulation state.

    -**Null advance is equivalent to processing silence**:
        -null_advance(n) produces the same state evolution as processing n zero-valued samples.

    -**Deterministic**:
        -Identical initial state, input, and parameters produce identical output.

    Its' primary method is `process()` and `null_advance` which correspond to processing some output signal transmitted
    through the channel, or not transmitted during a non-tx period (advances gain process time states, and flushes
    burst data from preceeding burst out of delay line / filter memories)
    """
    delay: float
    int_delay: int
    h_delay_fir: NDArray[complex128] | None
    scale: float
    fir_startup_priming: bool
    required_history: int
    fir_start_transient_delay: int
    _flush_zeros: NDArray[complex128]

    tap_gain_process: TetraTapGainProcess
    rayleigh_process: RayleighProcess | None
    static_process: StaticProcess | None

    def __init__(self, f_sim: int, f_doppler: float,
                 tap_data: PropagationTapParameters, seed_seq: SeedSequence | None = None):
        """
        Initializes the `PropagationTap` for handling input samples at rate: `f_sim`, for doppler shift of
        `f_doppler` with mean delay, fading gain process, and mean gain as described in `tap_data`
        PropagationTapParameters object. Uses seed_seq if passed to spawn grandchildern seeds to drive generators with
        indepdent but reproducable results.

        As part of initilization spawns gain process(es) attributes and performs warmup on them.

        :param f_sim: The simulation rate of the channel simulation, expected to be TRANSMIT_SIMULATION_SAMPLE_RATE
        :type f_sim: int
        :param f_doppler: The doppler frequency shift in hertz
        :type f_doppler: float
        :param tap_data: A PropagationTapParameters object that details the gain process type, mean delay, and amplitude
         scaling / normalized average gain.
        :type tap_data: PropagationTapParameters
        :param seed_seq: The numpy SeedSequence used to spawn grandchildern seeds to drive generators for the 3 random
        variables that enable the process, if None is passed, internally spawns a new SeedSequence from np.random
        :type seed_seq: SeedSequence | None
        """
        # 1. Handle initialization for the delay process
        # If we have non zero-delay, determine a Lagrange FIR that can provide fractional delay, and remaining delay
        self.delay = abs(tap_data.delay)
        self.int_delay, self.h_delay_fir = _calculate_delay_parameters(f_sim, self.delay, 3)
        if self.h_delay_fir is not None:
            self.h_zi = zeros(shape=self.h_delay_fir.size - 1, dtype=complex128)
            self.required_history = self.int_delay + (self.h_delay_fir.size - 1)
            self.fir_start_transient_delay = int(ceil((self.h_delay_fir.size - 1) / 2))
            self.fir_end_transient_delay = int(floor((self.h_delay_fir.size - 1) / 2))
        else:
            self.h_zi = zeros(shape=1, dtype=complex128)
            self.required_history = self.int_delay

        self._flush_zeros = zeros(self.required_history, dtype=complex128)
        self.fir_startup_priming = True
        self.int_delay_buffer = zeros(shape=self.int_delay, dtype=complex128)

        # 2. Handle ampltiude scaling
        self.scale = tap_data.amplitude_scale
        if self.scale < 0 or self.scale > 1:
            raise ValueError(f"Passed tap parameters has amplitude scale value of: {tap_data.amplitude_scale}"
                             f", valid range is: [0, 1], confirm in linear scale and not in dB")

        # If we have delay filter, apply scaling there to reduce number of multiplications per element
        if self.h_delay_fir is not None:
            self.h_delay_fir *= self.scale

        # 3. Handle tap_gain_process
        if tap_data.process not in [p.value for p in TetraTapGainProcess]:
            raise ValueError(f"Passed tap parameters has tap-gain-process value of: {tap_data.process}"
                             f", valid processes are of type in: {[p.value for p in TetraTapGainProcess]}")
        self.tap_gain_process = tap_data.process

        # Determine the tap gain process composistion
        match self.tap_gain_process:
            case TetraTapGainProcess.STATIC_PROCESS:
                self.rayleigh_process = None
                self.static_process = StaticProcess(f_sim=f_sim, f_doppler=f_doppler)
            case TetraTapGainProcess.CLASS_PROCESS:
                self.rayleigh_process = RayleighProcess(f_sim=f_sim, f_doppler=f_doppler, seed_seq=seed_seq)
                self.static_process = None
            case TetraTapGainProcess.RICE_PROCESS:
                self.rayleigh_process = RayleighProcess(f_sim=f_sim, f_doppler=f_doppler, seed_seq=seed_seq)
                # TETRA 300-392 V2.4.2 defines RICE as combination of both Rayleigh/CLASS and Static, but
                # with Static process having a doppler shift of 0.7 * f_doppler
                self.static_process = StaticProcess(f_sim=f_sim, f_doppler=0.7 * f_doppler)

    def _gain_handler(self, n_samples: int, repeatable: bool = False) -> NDArray[complex128]:
        """
        _gain_handler is a wrapper function that `process()` uses to fetch samples of fading process complex gain,
        handling the three types of processes: Static, Rice, and Rayleigh.

        If repeatable = True, then the gain samples are generated but the internal state does not advance thus
        the next _gain_handler() call will regenerate the same gain values.

        :param n_samples: The number of fading process gain samples required to generate.
        :type n_samples: int
        :param repeatable: True if repeatable, i.e. PropagationTap internal state does not change, or
         False if not repeatable and thus internal state (delay memory, gain process) advances by `n_samples`
        :type repeatable: bool, default False
        :return: returns n_samples of complex gain generated from the internal process(es)
        :rtype: NDArray[complex128]
        """
        if self.static_process is not None and self.rayleigh_process is not None:
            # RICE fading
            # TETRA 300-392 V2.4.2 defines RICE as combination of both Rayleigh/CLASS and Static, but
            # with Static process having a doppler shift of 0.7 * f_doppler
            gain = self.rayleigh_process.next(n_samples=n_samples, repeatable=repeatable) * (1/np_sqrt(2))
            gain += self.static_process.next(n_samples=n_samples, repeatable=repeatable) * (1/np_sqrt(2))
        elif self.rayleigh_process is not None and self.static_process is None:
            # Rayleigh/CLASS fading
            gain = self.rayleigh_process.next(n_samples=n_samples, repeatable=repeatable)
        elif self.static_process is not None:
            # Static Fading
            gain = self.static_process.next(n_samples=n_samples, repeatable=repeatable)
        else:
            gain = full(shape=n_samples, fill_value=(1.0 + 0.0j), dtype=complex128)

        return gain

    def _null_gain_handler(self, n_samples: int, repeatable: bool = False) -> None:
        """
        null_gain_handler is a wrapper that `null_advance()` uses to handle telling the gain proccess(es) to
        null_advance, i.e. increment the internal deterministic states by `n_samples` but to avoid performing gain
        computations to improve efficiency.

        :param n_samples: The number of zero/null non-Tx samples to process, advances the internal state of gain
         processes within PropagationTap without actually spending computation calculation gain.
        :type n_samples: int
        :param repeatable: True if repeatable, i.e. PropagationTap internal state does not change, or
         False if not repeatable and thus internal state (delay memory, gain process) advances by `n_samples`
        :type repeatable: bool, default False
        """
        if self.static_process is not None and self.rayleigh_process is not None:
            # Rice fading
            self.rayleigh_process.null_advance(n_samples=n_samples, repeatable=repeatable)
            self.static_process.null_advance(n_samples=n_samples, repeatable=repeatable)
        elif self.rayleigh_process is not None and self.static_process is None:
            # Rayleigh/CLASS fading
            self.rayleigh_process.null_advance(n_samples=n_samples, repeatable=repeatable)
        elif self.static_process is not None:
            # Static Fading
            self.static_process.null_advance(n_samples=n_samples, repeatable=repeatable)

    def reset_state(self) -> None:
        """
        reset_state clears the delay filter memory as well as the delay buffer and sets the interal flag,
        `fir_startup_priming` to True, so that the next call knows to smooth the initial FIR transisent.
        """

        self.int_delay_buffer[:] = 0.0 + 0.0j

        if self.h_delay_fir is not None:
            self.h_zi[:] = complex128(0.0 + 0.0j)

        self.fir_startup_priming = True

    def null_advance(self, n_samples: int | None = None, repeatable: bool = False) -> NDArray[complex128]:
        """
        null_advances advanced internal delay line memory states and gain-process states by `n_samples`, it will flush
        out any remaining delayed signal from preceeding process(A) calls.

        Compared to passing zeros to process(zeros(n_samples)), null_advance is more efficent and does not calculate
        extra unneeded gain values or delay responses saving computation whilst still advancing internal state.

        Used in the following cases:
        1. Flushing PropagationTap delay memory buffer from preceeding bursts
        2. Non-tx period means there is a time delay/gap between subsequent bursts, if null_advance is not used in
         between the tx periods, then the gain_process deterministic state does not advance. Otherwise fading process
         deterministic state does not advance during the null time, thus the short period statistics sampled may not
         represent the expected process with respect to time because time passed but the process did not evolve

        :param n_samples: The number of zero/null non-Tx samples to process, advances the internal state/delay memories
         by that much, and will flush out any remaining data with delay line memory.
        :type n_samples: int | None, defaults to None which assigns a minimal value of `required_history` that can flush
         and return preceeding process(A) outputs without extra post-pended zeros on the return.
        :param repeatable: True if desired output is repeatable, i.e. PropagationTap internal state does not change, or
         False if not repeatable and thus internal state (delay memory, gain process) advances by `n_samples`
        :type repeatable: bool, default False
        :return: returns n_samples number of output points contains zeros and preceeded by any signal remaining with
         the delay filter and delay line memories.
        :rtype: NDArray[complex128]
        """
        # Null means we are not transmitting anything but want timing to continue
        # thus insert zeros to flush the memory of process and increment processes
        # if we have no samples in memory, we can tell the processes to increment
        # their process trackers internally but without computation to reduce complexity
        if n_samples is None:
            n_samples = self.required_history

        if n_samples < self.required_history:
            raise ValueError(f"PropagationTap cannot process data with length less than: {self.required_history},"
                             f" recieved n_samples with size: {n_samples}")
        if not self.fir_startup_priming:
            # We can push out the remaining samples
            n_zeros = n_samples - self.required_history
            out = self.process(self._flush_zeros, self.required_history, StreamPosition.END_BURST,
                               'same', repeatable=repeatable)
            if n_zeros > 0:
                tail_zeros = zeros(shape=(n_samples-self.required_history), dtype=complex128)
                out = concatenate((out, tail_zeros))
                self._null_gain_handler((n_samples-self.required_history), repeatable=repeatable)
        else:
            out = zeros(shape=(n_samples), dtype=complex128)
            self._null_gain_handler((n_samples), repeatable=repeatable)

        return out

    def process(self, samples: NDArray[complex128],
                process_length: int,
                stream_position: StreamPosition | Literal['ISOLATED', 'START', 'MIDDLE', 'END'],
                mode: Literal["same", "full"] = "same",
                repeatable: bool = False) -> NDArray[complex128]:
        """
        `process` handles taking an input signal array then applying gain and delay to it and returning the output for a
        given PropgationTap. Because PropgationTap is built on a block-streaming model, other arguments are required to
        provide context to `process` in how to handle the input data and the resulting output

        PropgationTap.process is designed to handle both single isolated blocks of data as well as continuous streams
        while having the following functional properties:

        1.**Block Invariance (Streaming Invariance)**
            -**process(A:B, same) == process(A. same) : process(B, same)**
                - The means that a single burst of data can be indexed into multiple smaller blocks without any effect
                - This is key to allow for a theorecticaly infinite continous signal to be processed block(s) at a time
                 without knowing the future data in that stream.

        2.**Burst Invariant**
            -**Equivalent burst segmentations produce identical outputs, given `stream_position`'s are correct**
                - Say we have a burst, long or short, the only atributes we need to care about when describing it are:
                    - Is it continuous with previous and/or subsequent data that we have or will process?
                    - Does it ramp up and/or ramp down at the start and end of the data array given?
                - So, a single burst can be processed as a single sample array with `stream_position` 'ISOLATED'
                - Or, it can also be split up into N number of smaller arrays as long as no array is smaller than
                 `required_history` as follows: / 'START' | 'MIDDLE' | ... | 'MIDDLE' | 'END' \\
                - As long as when split up, each portion is described accurately as start, middle, end
                 the combined output of process of all these will equal a single 'ISOLATED' call

        3.**Repeatability**
            -**line n: process(a, repeatable=True) == line n+1: process(a, repeatable=True)**
                - This means that subsequent calls are not affected by any call that uses repeatable=True
                - This is key to allowing a fixed fading/gain profile for different input burst data blocks
                - It also allows for speculative transmission, we can transmit and handle a burst, then decided not to
                 without needing to worry about handling save states

        4.**'full' Mode is non-Destructive and Purely Observational**
            -**For an input burst seperated into A:B, doing process(A, 'full') will not affect the subsequent output of
             process(B) as compared to instead using process(A, 'same') originally**
                - The impact of this is that 'full' can be used when the full output of an input signal is desired,
                 such as in the case of examining a single MIDDLE burst block, without affecting a subsequent process
                 of the next block which is continuous
                - This is because the internal state of PropgationTap and its' gain processes, are not incremented
                 permanently for the lookahead portion that is length `required_history` long
                - This also means that 'full' is observational, if a user desires a flushed output, they should use
                 `same` then null_advance()

        **Note 1 (Use of 'full' mode)**:
        - When using 'full' mode with 'START' or 'MIDDLE' type blocks of data, additional
         future context is required since the subsequent block is continuous there is no way to predict a 'full' output
         without knowing atleast `required_history` number of future samples. In this case samples.size must be > than
         `process_length` it can be oversized but must atleast be `required_history` longer than `process_length`.
        - When using 'full' mode with 'ISOLATED' or 'END' type blocks of data, additional data is not required, zeros
         are assumed to be subsequent and used to push the output out.

        **Note 2 (`stream_position` behaviour)**:
        - `stream_position` describes the type of samples being provided, and provides context to `process()` on the
         presence of ramping up/down at start/end and continuous with previous or subsequent bursts for `full` mode.
        - Is the `samples` stream position is not correct FIR delay transisents can appear in the result as the presence
         and location of ramping helps `process()` smooth and remove artifacts, or `full` output results can be invalid
         because for example, zeros are expected after a `MIDDLE` block and thus it tapers down

        **Note 3 (input signal requirements)**:
        The input signal is expected to have the following:
        1. Signal bandwidth of importance is < 0.15*`f_sim`, outside this delay filter behaviour breaks down.
        2. The signal cannot be zero values (0.0 + 0.0j) any where except at the start and/or depedent on
         `stream_position`. If zero is required in the middle of a signal either segment it or ensure the zero portion
          is finite such as -50dbm. Otherwise delay filter transients can appear within the signal
        3. Signal ramping up/down should occur at the start/end of burst/block as described by `stream_position`,
         the ramping does not have to go to zero exactly but should go trend towards a finite but small value.
          Otherwise the built-in delay filter transient suppression may not be sufficent enough to remove over/under-
          shoot from signal edges.


        :param samples: Complex128 array of singla input burst/block samples to handle.
        :type samples: NDArray[complex128]
        :param process_length: The length of input from 'samples[:current_length]' to process, Note that:
         process_length >= samples.size. Samples.size may be larger to provide additional samples required for a full
         'full' `mode` output, or if a larger signal length is processed block by block
        :type process_length: int
        :param stream_position: Describes inputs data/block as [cont. w/ prev? | cont. w/ next? | ramp up? | ramp down?]
        , 'ISOLATED':[No,No,Yes,Yes], 'START':[No,Yes,Yes,No], 'MIDDLE':[Yes,Yes,No,No], 'END':[Yes,No,No,Yes]
        :type stream_position: StreamPosition
        :param mode: Output mode, if 'same' only outputs `current_length` number processed samples aligning with block
         boundaries, if 'full' includes all `current_length` input samples within output, WARNING, performing a 'full'
         output does not alter the internal state differently than a 'same' call. 'full' is designed as a lookahead only
         . If a user desires a flushed state including all input data:
         use 'same' then `tap.null_advance(tap.required_history)` or `full` then `tap.reset_state()`
        :type mode: Literal["same", "full"], default is 'same'
        :param repeatable: True if desired output is repeatable, i.e. PropagationTap internal state does not change, or
         False if not repeatable and thus internal state (delay memory, gain process) advances by `current_length`
        :type repeatable: bool, default False
        :return: Returns either `current_length` if `mode` = 'same', or `current_length` + `required_history` if `mode`
        = full, number of samples that have been delayed and had gain applied per the PropagationTap parameters.
        :rtype: NDArray[complex128]
        """

        if mode not in ['same', 'full']:
            raise ValueError(f"Expect mode value to be in: {['same', 'full']}, got {mode}")
        if process_length > samples.size:
            raise ValueError(f"Insufficent additional samples were passed, recieved:"
                             f" {samples.size}, excpected {process_length} based"
                             " on `current_length` argument.")

        # The reason for this limitation is otherwise the delay buffer and filter memory can be populated with zeros
        # and produce excess unmitgiated start/tail transients
        if samples.size < self.required_history:
            raise ValueError(f"PropagationTap cannot process data with length less than: {self.required_history},"
                             f" recieved samples with length: {samples.size}")

        # Handle if a null_advance is pushing out the rest of a burst
        suppress_end_transient = False
        suppress_middle_transient = False
        if self.h_delay_fir is not None:
            # Note that because current_length > self.required_history, there is never a time when self.int_delay_buffer
            # has data in it and self.h_zi doesnt not, vice versa, thus we only need to check one
            if not np_any(samples[:process_length]):
                suppress_middle_transient = True

        # 1. Calculate size, allocate, and fill input array to filter
        # Calculate the flush pads at start and end
        startup_padding = 0
        delay_buffer_tail_padding = 0
        lookahead_length = 0
        post_tail_padding = 0
        if self.fir_startup_priming:
            startup_padding += (0 if self.h_delay_fir is None else self.h_delay_fir.size - 1)

        if mode == "full":
            required_extra_samples = self.required_history - self.int_delay
            lookahead_length += self.required_history
            if stream_position in (StreamPosition.START_BURST, StreamPosition.MIDDLE_BURST):
                if (samples.size - process_length) < required_extra_samples:
                    raise ValueError(f"Insufficent additional samples were passed for"
                                     f" stream burst type: {stream_position} in mode: {mode},"
                                     f" expected: {required_extra_samples}, got:"
                                     f" {(samples.size - process_length)}")
            else:
                suppress_end_transient = True

        # If we need to suppress the tail transient, append (L-1 extra samples)
        if suppress_end_transient:
            post_tail_padding += (0 if self.h_delay_fir is None else self.h_delay_fir.size - 1)
        elif suppress_middle_transient:
            delay_buffer_tail_padding += (0 if self.h_delay_fir is None else self.h_delay_fir.size - 1)

        input_size = startup_padding + delay_buffer_tail_padding + process_length + lookahead_length + post_tail_padding
        x = zeros(shape=input_size, dtype=complex128)
        start = 0

        # 2. Handle the integer delay buffer
        x[start: start + self.int_delay] = self.int_delay_buffer[:]
        start += self.int_delay

        # 4. Handle padding
        if startup_padding > 0:
            x[start: start + startup_padding] = samples[0]
        start += startup_padding

        if delay_buffer_tail_padding > 0:
            # In the middle padding we are compensating for tail transistion to zero
            # to help with this we perform (L-1) point linear slope from self._int_delay_buffer[-1] to 0
            # Later on the pad is removed but this reduces/mitigates the FIR transient in the returned data
            pad = linspace(self.int_delay_buffer[-1], (0.0 + 0.0j),
                           (delay_buffer_tail_padding + 2), endpoint=True, dtype=complex128)[1:-1]
            x[start: start + delay_buffer_tail_padding] = pad[:]
        start += delay_buffer_tail_padding

        # Copy extra data into int_delay_buffer
        if not repeatable:
            if stream_position in (StreamPosition.START_BURST, StreamPosition.MIDDLE_BURST):
                # We dont want our lookahead portion in memory
                # instead we want to keep the memory as if we were in "same" mode
                self.int_delay_buffer[:] = samples[process_length - self.int_delay: process_length]
            else:
                self.int_delay_buffer[:] = samples[-self.int_delay:]

        # 5. Add current_samples
        # The integer delay buffer already contributes `int_delay` samples to x (current_length - int_delay) samples
        # of the current block are copied which is the case for both "full" and "same" modes
        x[start: start + (process_length - self.int_delay)] = samples[:(process_length-self.int_delay)]
        start += process_length - self.int_delay
        same_end_index = start

        # 6. Handle full mode extra post-pend
        if mode == "full":
            if stream_position in (StreamPosition.START_BURST, StreamPosition.MIDDLE_BURST):
                x[start:
                  start + self.required_history] = samples[(process_length-self.int_delay):
                                                           (process_length-self.int_delay) + self.required_history]
                start += self.required_history
            else:
                # In the case of ISOLATED or END_BURST, we only need to attach the remaining samples within
                # samples, the remaining (L-1) samples if we have an FIR, are already initialized to zero
                x[start: start + self.int_delay] = samples[(process_length-self.int_delay): process_length]
                start += self.int_delay

        # 7. Handle any tail padding
        if post_tail_padding > 0:
            # In the tail padding we are compensating for tail transistion to zero
            # to help with this we perform (L-1) point linear slope from samples[-1] to 0
            # Later on the pad is removed but this reduces/mitigates the FIR transient in the returned data
            if np_allclose(samples[-1], 0.0):
                # If for some reason "full" is called on a zero pad flush from advance_null, there is no point
                # in calculate a linspace, set all pad values to zero.
                pad = zeros(shape=post_tail_padding, dtype=complex128)
            else:
                pad = linspace(samples[-1], (0.0 + 0.0j), (post_tail_padding + 2),
                               endpoint=True, dtype=complex128)[1:-1]
            x[start: start + post_tail_padding] = pad[:]

        # 8. Filter the input
        if self.h_delay_fir is not None:
            # We have an FIR filter to process
            previous_zi = self.h_zi.copy()

            if mode == "same":
                y, self.h_zi = lfilter(self.h_delay_fir, [1.0], x, zi=self.h_zi)
                # If we want repeatbility, reset the filter state to the original
                if repeatable:
                    self.h_zi = previous_zi.copy()
            else:
                if not repeatable:
                    # Full mode
                    # We want flush repeatability, determine and set the filter state to an intermediate state
                    # that cooresponds to if we were doing "same" type output instead of "full"
                    # Calculate and copy intermediate_zi state
                    _, intermediate_zi = lfilter(self.h_delay_fir, [1.0], x[:same_end_index], zi=previous_zi)
                    # rerun to get full output, but now having saved an intermediate state to allow for 'same' next time
                    # Only used self.h_zi here instead of previous_zi because the linter/type-checker gets upset
                    # self.h_zi never gets updated so it is equivalent besides clarity.
                    y, _ = lfilter(self.h_delay_fir, [1.0], x, zi=self.h_zi)
                    # Update the self.h_zi for next call to the intermediate state
                    self.h_zi = intermediate_zi
                else:
                    # Nothing is repeated, being full does not affect anything in this regard
                    y, _ = lfilter(self.h_delay_fir, [1.0], x, zi=self.h_zi)
        else:
            y = x
        # 9. Extract the output of interest
        start_delay_offset = 0
        end_delay_offset = 0

        if self.h_delay_fir is not None:
            if startup_padding > 0:
                start_delay_offset = self.fir_start_transient_delay
            elif delay_buffer_tail_padding > 0:
                start_delay_offset = self.fir_end_transient_delay

            if post_tail_padding > 0:
                end_delay_offset = self.fir_end_transient_delay

        start = self.int_delay + start_delay_offset
        end = self.int_delay + start_delay_offset + startup_padding + delay_buffer_tail_padding
        y = concatenate((y[:start], y[end:end + process_length + lookahead_length + post_tail_padding]))
        start = self.int_delay + process_length + end_delay_offset
        y = concatenate((y[:start], y[start + post_tail_padding:]))
        # 10. Generate complex gain
        if mode == "full":
            # Since in full mode, we need to ensure that the extra gain samples generated to get full output
            # are repeatable, as if the current block was in mode "same" to ensure the next block has continous
            # gain process
            gain_a = self._gain_handler(process_length, repeatable=repeatable)
            gain_b = self._gain_handler(self.required_history, repeatable=True)
            gain = concatenate((gain_a, gain_b), dtype=complex128)
        else:
            gain = self._gain_handler(process_length, repeatable)
        if startup_padding > 0 and delay_buffer_tail_padding > 0:
            raise RuntimeError(f"PropagationTap, both start and delay buffer tail padding is non zero:"
                               f" {startup_padding}, and {delay_buffer_tail_padding} respectively.")
        # 11. Check if downstream delay arrays is zero
        if self.h_delay_fir is not None:
            # Note that because current_length > self.required_history, there is never a time when self.int_delay_buffer
            # has data in it and self.h_zi doesnt not, vice versa, thus we only need to check one
            if not np_any(self.int_delay_buffer):
                self.fir_startup_priming = True
            else:
                self.fir_startup_priming = False
        # 12. Apply gain and return
        return y * gain

###################################################################################################


class PropagationModel:
    # TODO: Finish implementing PropagationModel
    f_doppler: float
    f_ch: float
    f_sim: int

    seed_seq: SeedSequence
    _tap_seeds: list[SeedSequence]

    propagation_params: PropagationModelParameters
    taps: list[PropagationTap]

    def __init__(self, f_sim: int, f_ch: float, model_type: TetraPropagationModels | PropagationModelParameters,
                 seed_seq: SeedSequence | None = None, v_kph: float | None = None,
                 doppler_relative_to_430mhz: bool = True):

        # Can pass either custom PropagationModelParameters, or simply reference one defined for TETRA in constants
        if isinstance(model_type, TetraPropagationModels) and model_type not in TETRA_PROPAGATION_MODELS:
            raise ValueError(f"Passed propagation model: {model_type}, not defined in constants.py values:"
                             f": {TETRA_PROPAGATION_MODELS.keys()}")

        if not isinstance(model_type, PropagationModelParameters):
            raise ValueError(f"Passed propagation model: {model_type} is not a predefined one of types:"
                             f"{TETRA_PROPAGATION_MODELS.keys()} or a custom one correctly defined as type:"
                             f" {type(PropagationModelParameters)}")

        # confirm custom model parameters are specified
        if not model_type.taps:
            raise ValueError(f"Passed custom propagation model: {model_type}, does not have any taps defined")

        # Determine doppler frequency for tap gain models
        if v_kph is None:
            if isinstance(model_type, TetraPropagationModels) and \
               model_type not in TETRA_DEFAULT_MODEL_VELOCITIES_KPH:
                # Model is predefined one but not one used in TETRA testing thus does not have default velocity (RA, BU)
                raise ValueError(f"Passed velocity (kph) to PropagationModel is None, and model type: {model_type}"
                                 f" is not a default one with TETRA defined velocity like: "
                                 f"{TETRA_DEFAULT_MODEL_VELOCITIES_KPH.keys()}")

            if isinstance(model_type, TetraPropagationModels):
                # Assign a default velocity defined in TETRA standard, (STATIC, TU, HT, EQ): 0.0, 50.0, 200.0, 200.0
                v_kph = TETRA_DEFAULT_MODEL_VELOCITIES_KPH[model_type]
            else:
                # Model is a custom one, but no velocity was passed to use with it
                raise ValueError(f"Passed velocity (kph) to PropagationModel is None, and model type: {model_type}"
                                 f" is not a default one with TETRA defined velocity like: "
                                 f"{TETRA_DEFAULT_MODEL_VELOCITIES_KPH.keys()}")

        # Calculate doppler, note that TETRA testing keeps doppler shift constant relative to 430 MHz outside the
        # default band: [350MHz, 520MHz], but this can be turned off via the `doppler_relative_to_430mhz`
        self.f_doppler = _calculate_doppler_shift(f_ch, v_kph, doppler_relative_to_430mhz)
        self.f_sim = f_sim
        self.f_ch = f_ch
        # Initialze the taps

        if seed_seq is None:
            self.seed_seq = SeedSequence()
        else:
            self.seed_seq = seed_seq

        # Extract/assign the tap parameters
        if isinstance(model_type, TetraPropagationModels):
            self.propagation_params = TETRA_PROPAGATION_MODELS[model_type]
        else:
            self.propagation_params = model_type

        # Initilize the taps
        self.taps = []
        self._tap_seeds = self.seed_seq.spawn(len(self.propagation_params.taps))
        for n, tap in enumerate(self.propagation_params.taps):
            self.taps.append(PropagationTap(f_sim, self.f_doppler, tap, self._tap_seeds[n]))

###################################################################################################
