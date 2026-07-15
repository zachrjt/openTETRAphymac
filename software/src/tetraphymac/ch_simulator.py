"""
ch_simulator.py contains functions and classes used to simulate channels between a MS/BS transmitter and a MS/BS
reciever.
"""
from typing import Tuple

from numpy import round, zeros, float64, complex128, pi, arange, cos, sin, sum as np_sum, ceil, empty, int64, \
                  sqrt as np_sqrt, max as np_max
from numpy.typing import NDArray
from numpy.random import SeedSequence, Generator, PCG64

from scipy.signal import remez as sp_remez, upfirdn as sp_upfirdn
from scipy.constants import c as C_SPEED_OF_LIGHT

from .constants import TETRA_PROPAGATION_MODELS, TetraPropagationModels, PropagationTapParameters, \
                       TetraTapGainProcess, PropagationModelParameters, OPENTETRAPHYMAC_DEFAULT_RX_FREQUENCY, \
                       TETRA_DEFAULT_MODEL_VELOCITIES_KPH, TETRA_FADING_SIMULATION_RATE
from .transmitter import TRANSMIT_SIMULATION_SAMPLE_RATE

###################################################################################################


def _generate_lagrange_delay_fir(d: float, n_order: int = 3) -> NDArray[complex128]:
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
    f_ref = 430.0E6 if doppler_relative_to_430mhz else f_ch
    f_doppler = (v_kph / 3.6) / (C_SPEED_OF_LIGHT / f_ref)
    return f_doppler

###################################################################################################


class RayleighFadingSimulator:
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

    def __init__(self, f_sim: int, f_fade_sim: int, f_doppler: float, seed_seq: SeedSequence | None, m_order: int = 3):
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
        self.f_fade_sim = f_fade_sim
        if f_fade_sim/2 < f_doppler:
            raise ValueError(f"RayleighFadingSimulator sample rate: {f_fade_sim} is too low for doppler shift of:"
                             f" {f_doppler}")
        elif f_sim % f_fade_sim != 0:
            raise ValueError(f"Passed RayleighFadingSimulation sample rate: {f_fade_sim}, not a factor of"
                             f" passed simulation rate: {f_sim}")
        elif f_sim < f_fade_sim:
            raise ValueError(f"Passed RayleighFadingSimulation sample rate: {f_fade_sim}, is larger than"
                             f" passed simulation rate: {f_sim}")
        self.upsample_factor = f_sim // f_fade_sim

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

        # Generate interpolation filter, and initialize interpolation memory
        h = sp_remez(8192, [0, 2800.0, f_fade_sim, f_sim/2], [1, 0], weight=[1, 100], fs=f_sim)
        dc_gain = np_sum(h)
        h *= (self.upsample_factor / dc_gain)
        self._h_interp = h.astype(complex128)

        self._mem_len = int(ceil((self._h_interp.size-1) / self.upsample_factor))
        self._h_interp_mem = empty(shape=self._mem_len, dtype=complex128)
        self._interp_buffer = empty(self.upsample_factor, dtype=complex128)
        self._interp_buffer_len = 0
        # Generate warmup history for interpolation memory
        self._h_interp_mem = self.evaluate(self._mem_len, start_index=-self._mem_len)

    def evaluate(self, n_samples: int, start_index: int | None = None) -> NDArray[complex128]:
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

    def apply_complex_gain(self, signal: NDArray[complex128]) -> NDArray[complex128]:
        # Determine how many base rate samples are required, taking into account leftover data in buffer
        needed_samples = np_max(0, signal.size - self._interp_buffer_len)
        # integer ceiling form, instead of int(ceil(float))
        n_samples = (needed_samples + self.upsample_factor - 1) // self.upsample_factor
        k_samples = needed_samples
        if n_samples == 0:
            leftover_samples = self._interp_buffer_len - signal.size
        else:
            leftover_samples = n_samples * self.upsample_factor - k_samples
        # Generate base rate samples of the process
        fifo_array = empty(shape=(signal.size), dtype=complex128)
        fifo_array[:self._interp_buffer_len] = self._interp_buffer[:self._interp_buffer_len]
        if n_samples != 0:
            x = empty(shape=(self._mem_len + n_samples), dtype=complex128)
            x[:self._mem_len] = self._h_interp_mem
            x[self._mem_len:] = self.evaluate(n_samples)
            self._h_interp_mem = x[-self._mem_len:]
            # Interpolate up
            y = sp_upfirdn(h=self._h_interp, x=x, up=self.upsample_factor)

            # Discard the initial transisent/memory
            y = y[self._mem_len * self.upsample_factor:
                  self._mem_len*self.upsample_factor + n_samples*self.upsample_factor]
            # Copy into buffer any leftover data
            fifo_array[self._interp_buffer_len:] = y[:k_samples]
            self._interp_buffer[:leftover_samples] = y[k_samples:]

        # Handle buffer, must shuffle buffer samples around if we dont use the entire buffer
        if n_samples == 0:
            self._interp_buffer[:leftover_samples] = self._interp_buffer[signal.size: self._interp_buffer_len]
            self._interp_buffer_len = leftover_samples
        else:
            self._interp_buffer_len = leftover_samples

        return signal * fifo_array

###################################################################################################


class PropagationTap:
    delay: float
    int_delay: int
    h_delay_fir: NDArray[complex128] | None
    scale: float

    tap_gain_process: TetraTapGainProcess
    rayleigh_process: RayleighFadingSimulator | None
    static_process_present: bool

    def __init__(self, f_sim: int, f_doppler: float, tap_data: PropagationTapParameters, seed_seq: SeedSequence):

        # 1. Handle initialization for the delay process
        # If we have non zero-delay, determine a Lagrange FIR that can provide fractional delay, and remaining delay
        self.delay = abs(tap_data.delay)
        self.int_delay, self.h_delay_fir = _calculate_delay_parameters(f_sim, self.delay, 3)

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
                self.static_process_present = True
            case TetraTapGainProcess.CLASS_PROCESS:
                self.rayleigh_process = RayleighFadingSimulator(f_sim=f_sim, f_fade_sim=TETRA_FADING_SIMULATION_RATE,
                                                                f_doppler=f_doppler, seed_seq=seed_seq)
                self.static_process_present = False
            case TetraTapGainProcess.RICE_PROCESS:
                self.rayleigh_process = RayleighFadingSimulator(f_sim=f_sim, f_fade_sim=TETRA_FADING_SIMULATION_RATE,
                                                                f_doppler=f_doppler, seed_seq=seed_seq)
                self.static_process_present = True

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
