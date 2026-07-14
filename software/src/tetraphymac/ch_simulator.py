"""
ch_simulator.py contains functions and classes used to simulate channels between a MS/BS transmitter and a MS/BS
reciever.
"""
from typing import Tuple

from numpy import round, zeros, float64, complex128, pi, arange, cos, sin, sum as np_sum, ceil, empty, int64
from numpy.typing import NDArray
from numpy.random import SeedSequence, Generator, PCG64

from scipy.signal import remez as sp_remez, upfirdn as sp_upfirdn

from .constants import TETRA_PROPAGATION_MODELS, TetraPropagationModels, PropagationTapParameters, \
                       TetraTapGainProcess, PropagationModelParameters, OPENTETRAPHYMAC_DEFAULT_RX_FREQUENCY
from .transmitter import TRANSMIT_SIMULATION_SAMPLE_RATE


def _generate_lagrange_delay_fir(d: float, n_order: int = 3) -> NDArray[complex128]:
    h_n = zeros(shape=n_order+1, dtype=complex128)

    for n in range(n_order+1):
        n_k = 1.0
        k_vals = [x for x in range(n_order+1) if x != n]
        for k in k_vals:
            n_k *= (d-k)/(n-k)
        h_n[n] = complex128(n_k)

    return h_n


def _calculate_delay_parameters(fs: float, delay: float,
                                n_lagrange_order: int = 3) -> Tuple[int, NDArray[complex128] | None]:
    if delay > 0.0:
        delay_samples = delay * fs
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


class RayleighFadingSimulator:

    def __init__(self, f_sim: int, f_fade_sim: int, f_doppler: float, seed_seq: SeedSequence | None, m_order: int = 3):
        # Handle order checks and time tracker
        self.m = m_order
        if m_order < 8:
            raise RuntimeWarning(f"RayleighFadingSimulator order of: {m_order} is not recommended due to disagreeance"
                                 " with ideal statistics, recommended M>=8")
        # Recall output is deterministic, instead of tracking phase for each of the M oscilators, we just evaluate phase
        # using the time value since numpy will wrap it and speed benefit is not really apparent here unlike hw with LUT
        self.t_sample = int64(0)

        # Handle frequency checks
        self.fd = abs(f_doppler)
        self.wd = (f_doppler * 2 * pi)
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
        # Generate grandchildern seed's
        _phi_seed, _psi_seed, _theta_seed = self.seed_seq.spawn(3)
        self._phi_gen = Generator(PCG64(_phi_seed))
        self.psi_gen = Generator(PCG64(_psi_seed))
        self._theta_gen = Generator(PCG64(_theta_seed))

        self.phi_n = self._phi_gen.uniform(low=-pi, high=pi, size=self.m)[:, None]
        self.psi_n = self.psi_gen.uniform(low=-pi, high=pi, size=self.m)[:, None]
        self.theta_n = self._theta_gen.uniform(low=-pi, high=pi, size=self.m)

        self.alpha_n = (2*pi*(arange(1, self.m+1)) - pi + self.theta_n)/(4*self.m)
        self._wd_cos_alpha_n = self.wd * cos(self.alpha_n)[:, None]
        self._wd_sin_alpha_n = self.wd * sin(self.alpha_n)[:, None]

        # Generate interpolation samples and filter warmup
        self._h_interp = complex128(sp_remez(8192, [0, 2800.0, f_fade_sim, f_sim/2], [1, 0], weight=[1, 100], fs=f_sim))
        dc_gain = np_sum(self._h_interp)
        self._h_interp *= (self.upsample_factor / dc_gain)

        self._mem_len = int(ceil((self._h_interp.size-1) / self.upsample_factor))
        self._h_interp_mem = empty(shape=self._mem_len-1, dtype=complex128)
        t = arange(-self._mem_len, 0) * (1 / self.f_fade_sim)
        self._h_interp_mem = self.evaluate(t)

    def evaluate(self, t: NDArray[float64]):
        # Z (Result) = a + jb
        a = np.sum(np.cos((t * self.wd_cos_alpha_n) + self.phi_n), axis=0)
        b = np.sum(np.sin((t * self.wd_sin_alpha_n) + self.psi_n), axis=0)
        z = (np.sqrt(2/self.M))*(a + 1j*b)
        return z
    def apply_complex_gain(self, signal: NDArray[complex128]):
        # 1. Generate enough samples


class PropagationTap:

    def __init__(self, fs: float, tap_data: PropagationTapParameters, seed_seq: SeedSequence):

        # 1. Handle initialization for the delay process
        # If we have non zero-delay, determine a Lagrange FIR that can provide fractional delay, and remaining delay 
        self.delay = abs(tap_data.delay)
        self.int_delay, self.h_delay_fir = _calculate_delay_parameters(fs, self.delay, 3)


        # 2. Handle ampltiude scaling
        self.scale = tap_data.amplitude_scale
        if self.scale < 0 or self.scale > 1:
            raise ValueError(f"Passed tap parameters has amplitude scale value of: {tap_data.amplitude_scale}"
                             f", valid range is: [0, 1], confirm in linear scale and not in dB")

        # If we have delay filter, apply scaling there to reduce number of multiplications per element
        if self.h_delay_fir is not None:
            self.h_delay_fir *= self.scale

        # 3. Handle tap_gain_process
        self.tap_gain_process = tap_data.process
        if self.tap_gain_process not in TetraTapGainProcess:
            raise ValueError(f"Passed tap parameters has tap-gain-process value of: {tap_data.process}"
                             f", valid processes are of type in: {[p.value for p in TetraTapGainProcess]}")

        # Initilize fading gain model


class PropagationModel:

    def __init__(self, model_type: TetraPropagationModels | PropagationModelParameters, v_kph: float | None = None):

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
        
        # Generate the taps
    
    # 