"""
ch_simulator.py contains functions and classes used to simulate channels between a MS/BS transmitter and a MS/BS
reciever.
"""
from typing import Tuple, Literal
from abc import ABC, abstractmethod
from numpy import round, zeros, float64, complex128, pi, arange, cos, sin, sum as np_sum, empty, int64, \
                  sqrt as np_sqrt, isclose, exp, full, concatenate
from numpy.typing import NDArray
from numpy.random import SeedSequence, Generator, PCG64

from scipy.signal import remez as sp_remez, upfirdn as sp_upfirdn, lfilter
from scipy.constants import c as C_SPEED_OF_LIGHT

from .constants import TETRA_PROPAGATION_MODELS, TetraPropagationModels, PropagationTapParameters, \
                       TetraTapGainProcess, PropagationModelParameters, OPENTETRAPHYMAC_DEFAULT_RX_FREQUENCY, \
                       TETRA_DEFAULT_MODEL_VELOCITIES_KPH, TETRA_FADING_SIMULATION_RATE, StreamPosition
from .transmitter import TRANSMIT_SIMULATION_SAMPLE_RATE


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
        if f_sim != TRANSMIT_SIMULATION_SAMPLE_RATE:
            raise ValueError(f"Passed simulation sample rate: {f_sim} is not the expected"
                             f" rate of {TRANSMIT_SIMULATION_SAMPLE_RATE}")
        if isclose(f_doppler, 0):
            self.doppler_yes = False
            self.fd = 0.0
            self.wd = 0.0
        else:
            self.fd = f_doppler
            self.wd = f_doppler * 2 * pi
            self.cos_pi_wd_tsim = (self.wd * 2 * pi * (1/f_sim))
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


class RayleighProcess(ComplexGainProcess):
    """
    RayleighFadingSimulation is a class used to handle generation of Rayleigh fading process for PropagationTaps.
    It handles indepedently generating samples of a complex gain process using a deterministic sum-of-sinusoids method
    that has good agreeance with the expected statistics using a method proposed by Zheng and Xiao.

    It interpolates the complex gain samples up to full simulation rate: f_sim from a base rate of
    TETRA_FADING_SIMULATION_RATE with high (>180dB within an offset of 12 * TETRA_FADING_SIMULATION_RATE) rejection
    of spectral images created during upsampling.

    The resulting statistics agree well the base rate sampled data and the expected statistics for a Rayleigh fading
    enviroment interms of i.i.d. I and Q components, gain magnitude/enevelope, gain power enevelope, autocorrelation,
    and time delay w.r.t. to an indepedently generated sequence generated with the same base seed sequence.


    Its' primary method is .next(n_samples), generates and interpolates a n_samples of complex gain samples at f_sim
    rate and returns the complex gain/fade samples. It can also be configured to allow
    for subsequent repitition of the sample complex gain profile, easing scenario testing by remove the need to
    initialize multiple propagation taps/models with the same seed.
    """
    m: int
    _sample_idx: int64
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

    _h_interp: NDArray[complex128]
    _h_interp_mem: NDArray[complex128]
    _mem_len: int
    _interp_buffer: NDArray[complex128]
    _interp_buffer_len: int

    def __init__(self, f_sim: int, f_doppler: float, seed_seq: SeedSequence | None, m_order: int = 264):
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
        self.wd = (abs(f_doppler) * 2 * pi)
        self.f_fade_sim = TETRA_FADING_SIMULATION_RATE
        # Might handle variable simulation rates for the fading sim in the future but unlikely
        if f_sim != TRANSMIT_SIMULATION_SAMPLE_RATE:
            raise ValueError(f"Passed simulation sample rate: {f_sim} is not the expected"
                             f" rate of {TRANSMIT_SIMULATION_SAMPLE_RATE}")
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

        self._initialize_filters(f_sim)

    def _initialize_filters(self, f_sim: int):
        """
        Generates interpolate filters and normalizes their DC gain to 1. Allocates filter and output buffer memory.
        Calculates total filter group delay, and then generates cooresponding samples based on the group delay such that
        when ._apply_complex_gain(signal) is called for the first time, output samples are non-transisent and coorespond
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
        self._h1_mem = zeros(shape=(3*gd_base), dtype=complex128)

        # Compensate for group delay such that when .apply_complex_gain() is called 1st time, samples start at t=0
        # This is not strictly needed but helps if output is compared to external code with same seed and implementation
        self._sample_idx = int64(gd_base)
        # Allocate interpolation buffer, note max used size is self.upsample_factor - 1
        self._buffer = zeros(self.total_upsample - 1, dtype=complex128)
        self._buffer_len = 0
        # Generate warmup samples
        warmup = 32768
        warmup_samples = self._generate(warmup, (gd_base-warmup))
        _ = self._interpolate(warmup_samples)

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
        z = (np_sqrt(2/self.m))*(a + 1j*b)
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

        # Remove front pad which has filter transisents in it
        start = self._h1_mem.size * self.total_upsample
        stop = start + (n_samples * self.total_upsample)
        return y3[start:stop]

    def apply_complex_gain(self, signal: NDArray[complex128], repeatable: bool = False) -> NDArray[complex128]:
        """
        Generates Rayleigh fading process complex gain via sum of sinusoids method at lower rate, then interpolates up
        to f_sim sample rate of the passed input signal and apply the complex gain to input parameter signal.

        Note: because of interpolation, finite leakage of the doppler spectrum occurs at frequnecy multiples of
        TETRA_FADING_SIMULATION_RATE (80khz), rejection of leakage from interpolation is min. 190dB within 900khz,
        and 150dB at most at n*960khz spectral images.


        :param signal: Input array of complex128 data sampled at the f_sim rate passed to object at initilization
        :type signal: NDArray[complex128]
        :param repeatable: If True, does not increment sample counter of gain generation, therefore for the next same
        sized input signal repeats the same complex gain profile, repeats for next call as long as set to True
        :type repeatable: bool = False
        :return: Returns input signal.size number samples multiplied by complex gain as: signal * z(t)
        :rtype: NDArray[complex128]
        """
        # Determine how many base rate samples are required, taking into account leftover data in buffer
        needed_samples = max(0, signal.size - self._buffer_len)
        # integer ceiling form, instead of int(ceil(float))
        n_samples = (needed_samples + self.total_upsample - 1) // self.total_upsample
        if n_samples == 0:
            leftover_samples = self._buffer_len - signal.size
        else:
            leftover_samples = n_samples * self.total_upsample - needed_samples

        out_array = empty(shape=(signal.size), dtype=complex128)
        out_array[:self._buffer_len] = self._buffer[:self._buffer_len]

        if n_samples != 0:
            # Generate base rate samples of the process
            samples = self._generate(n_samples, int(self._sample_idx) if repeatable else None)
            y = self._interpolate(samples, repeatable)
            out_array[self._buffer_len:] = y[:needed_samples]
            if not repeatable:
                self._buffer[:leftover_samples] = y[needed_samples:]
                self._buffer_len = leftover_samples
        else:
            if not repeatable:
                self._buffer[:leftover_samples] = self._buffer[signal.size: self._buffer_len]
                self._buffer_len = leftover_samples

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
        # Determine how many base rate samples are required, taking into account leftover data in buffer
        needed_samples = max(0, n_samples - self._buffer_len)
        # integer ceiling form, instead of int(ceil(float))
        n_samples = (needed_samples + self.total_upsample - 1) // self.total_upsample
        if n_samples == 0:
            leftover_samples = self._buffer_len - n_samples
        else:
            leftover_samples = n_samples * self.total_upsample - needed_samples

        out_array = empty(shape=(n_samples), dtype=complex128)
        out_array[:self._buffer_len] = self._buffer[:self._buffer_len]

        if n_samples != 0:
            # Generate base rate samples of the process
            samples = self._generate(n_samples, int(self._sample_idx) if repeatable else None)
            y = self._interpolate(samples, repeatable)
            out_array[self._buffer_len:] = y[:needed_samples]
            if not repeatable:
                self._buffer[:leftover_samples] = y[needed_samples:]
                self._buffer_len = leftover_samples
        else:
            if not repeatable:
                self._buffer[:leftover_samples] = self._buffer[n_samples: self._buffer_len]
                self._buffer_len = leftover_samples

        return out_array

###################################################################################################


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

        int_delay = int(round(delay_samples - filt_int_delay))
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
    delay: float
    int_delay: int
    h_delay_fir: NDArray[complex128] | None
    scale: float
    prime_fir_state: bool
    required_history: int

    tap_gain_process: TetraTapGainProcess
    rayleigh_process: RayleighProcess | None
    static_process: StaticProcess | None

    def __init__(self, f_sim: int, f_doppler: float, tap_data: PropagationTapParameters, seed_seq: SeedSequence):

        # 1. Handle initialization for the delay process
        # If we have non zero-delay, determine a Lagrange FIR that can provide fractional delay, and remaining delay
        self.delay = abs(tap_data.delay)
        self.int_delay, self.h_delay_fir = _calculate_delay_parameters(f_sim, self.delay, 3)
        if self.h_delay_fir is not None:
            self.h_zi = zeros(shape=self.h_delay_fir.size - 1, dtype=complex128)
            self.required_history = self.int_delay + (self.h_delay_fir.size - 1)
        else:
            self.h_zi = zeros(shape=1, dtype=complex128)
            self.required_history = self.int_delay

        self.prime_fir_state = True
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
        if tap_data.process not in TetraTapGainProcess:
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
                self.static_process = StaticProcess(f_sim=f_sim, f_doppler=(0.7 * f_doppler))

    def gain_handler(self, n_samples: int, repeatable: bool = False) -> NDArray[complex128]:
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

    def reset_state(self):
        self.int_delay_buffer[:] = 0.0 + 0.0j

        if self.h_delay_fir is not None:
            self.h_zi[:] = 0.0 + 0.0j

        self.prime_fir_state = True

    def null(self, n_samples: int):
        # Null means we are not transmitting anything but want timing to continue
        # thus insert zeros to flush the memory of process and increment processes
        # if we have no samples in memory, we can tell the processes to increment
        # their process trackers internally but without computation to reduce complexity
        pass

    def process(self, samples: NDArray[complex128],
                current_length: int,
                stream_position: StreamPosition,
                mode: Literal["same", "full"] = "same",
                repeatable: bool = False) -> NDArray[complex128]:
        if current_length > samples.size:
            raise ValueError(f"Insufficent additional samples were passed, recieved:"
                             f" {samples.size}, excpected {current_length} based"
                             " on `current_length` argument.")
        # 1. Calculate size, allocate, and fill input array to filter
        # Calculate the flush pads at start and end
        startup_padding = 0
        lookahead_length = 0
        if self.prime_fir_state:
            startup_padding += (0 if self.h_delay_fir is None else self.h_delay_fir.size - 1)

        if mode == "full":
            required_extra_samples = self.required_history - self.int_delay
            lookahead_length += self.required_history
            if stream_position in (StreamPosition.START_BURST, StreamPosition.MIDDLE_BURST):
                if (samples.size - current_length) < required_extra_samples:
                    raise ValueError(f"Insufficent additional samples were passed for"
                                     f" stream burst type: {stream_position} in mode: {mode},"
                                     f" expected: {required_extra_samples}, got:"
                                     f" {(samples.size - current_length)}")

        input_size = startup_padding + current_length + lookahead_length
        x = zeros(shape=input_size, dtype=complex128)
        start = 0 + startup_padding

        # 2. Handle the integer delay buffer
        x[start: start + self.int_delay] = self.int_delay_buffer[:]
        start += self.int_delay
        if not repeatable:
            if stream_position in (StreamPosition.START_BURST, StreamPosition.MIDDLE_BURST):
                # We dont want our lookahead portion in memory
                # instead we want to keep the memory as if we were in "same" mode
                self.int_delay_buffer[:] = samples[-self.required_history: -(self.required_history - self.int_delay)]
            else:
                self.int_delay_buffer[:] = samples[-self.int_delay:]

        # 3. Add current_samples
        # The integer delay buffer already contributes `int_delay` samples to x (current_length - int_delay) samples
        # of the current block are copied which is the case for both "full" and "same" modes
        x[start: start + (current_length - self.int_delay)] = samples[:(current_length-self.int_delay)]
        start += current_length - self.int_delay

        # 4. Handle full mode extra post-pend
        if mode == "full":
            if stream_position in (StreamPosition.START_BURST, StreamPosition.MIDDLE_BURST):
                x[start:
                  start + self.required_history] = samples[(current_length-self.int_delay):
                                                           (current_length-self.int_delay) + self.required_history]
            else:
                # In the case of ISOLATED or END_BURST, we only need to attach the remaining samples within
                # samples, the remaining (L-1) samples if we have an FIR, are already initialized to zero
                x[start: start + self.int_delay] = samples[(current_length-self.int_delay): current_length]

        # 5. Filter the input
        if self.h_delay_fir is not None:
            # We have an FIR filter to process
            previous_zi = self.h_zi.copy()

            if mode == "same":
                y, self.h_zi = lfilter(self.h_delay_fir, [1.0], x, zi=self.h_zi)
                # If we want repeatbility, reset the filter state to the original
                if repeatable:
                    self.h_zi = previous_zi
            else:
                if not repeatable:
                    # Full mode
                    # We want flush repeatability, determine and set the filter state to an intermediate state
                    # that cooresponds to if we were doing "same" type output instead of "full"

                    # Calculate and copy intermediate_zi state
                    _, intermediate_zi = lfilter(self.h_delay_fir, [1.0], x[:-lookahead_length], zi=previous_zi)

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
        # 6. Extract the output of interest
        y = y[startup_padding: startup_padding + current_length + lookahead_length]

        # 7. Generate complex gain
        if mode == "full" and not repeatable:
            # Since in full mode, we need to ensure that the extra gain samples generated to get full output
            # are repeatable, as if the current block was in mode "same" to ensure the next block has continous
            # gain process
            gain_a = self.gain_handler(current_length, repeatable=False)
            gain_b = self.gain_handler(self.required_history, repeatable=True)
            gain = concatenate((gain_a, gain_b), dtype=complex128)
        else:
            gain = self.gain_handler(current_length, repeatable)

        # 8. Apply gain and return
        return y * gain
###################################################################################################


class PropagationModel:
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
        if isinstance(model_type, TetraPropagationModels) and model_type not in TETRA_PROPAGATION_MODELS.keys():
            raise ValueError(f"Passed propagation model: {model_type}, not defined in constants.py values:"
                             f": {TETRA_PROPAGATION_MODELS.keys()}")

        else:
            if not isinstance(model_type, PropagationModelParameters):
                raise ValueError(f"Passed propagation model: {model_type} is not a predefined one of types:"
                                 f"{TETRA_PROPAGATION_MODELS.keys()} or a custom one correctly defined as type:"
                                 f" {type(PropagationModelParameters)}")
            else:
                # confirm custom model parameters are specified
                if not model_type.taps:
                    raise ValueError(f"Passed custom propagation model: {model_type}, does not have any taps defined")

        # Determine doppler frequency for tap gain models
        if v_kph is None:
            if isinstance(model_type, TetraPropagationModels) and \
               model_type not in TETRA_DEFAULT_MODEL_VELOCITIES_KPH.keys():
                # Model is predefined one but not one used in TETRA testing thus does not have default velocity (RA, BU)
                raise ValueError(f"Passed velocity (kph) to PropagationModel is None, and model type: {model_type}"
                                 f" is not a default one with TETRA defined velocity like: "
                                 f"{TETRA_DEFAULT_MODEL_VELOCITIES_KPH.keys()}")
            elif isinstance(model_type, TetraPropagationModels):
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
