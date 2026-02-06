from typing import Literal
from pathlib import Path
from sys import byteorder
from numpy import complex64, pi, float64, ravel, zeros, float32, int16, vstack, \
    fromfile, int64, concatenate, right_shift, clip, arange, cos, full, where, \
    left_shift, sqrt
from numpy import abs as np_abs
from numpy import max as np_max
from numpy import round as np_round
from numpy.typing import NDArray
from scipy.signal import convolve as sp_convolve

from .constants import BASEBAND_SAMPLING_FACTOR

NUMBER_OF_FRACTIONAL_BITS = 17        # Relates to Q17 fixed point representation used
PLUTOSDR_DAC_BIT_NUMBER = 12          # Number of bits in pluto sdr
OPENTETRAPHYMAC_HW_DAC_NUMBER = 10    # Number of bits in openTETRAphymac hw implementation (AD9115)

VALID_ROUNDING_METHODS = ('rti', 'rtz', 'truncate', 'unbiased')
VALID_START_GUARD_PERIOD_OFFSETS = [0, 10, 12, 34, 120]
VALID_END_GUARD_PERIOD_OFFSETS = [0, 8, 10, 14, 16]

###################################################################################################

def generate_ramping_lut_quantized(n: int, sps:int=BASEBAND_SAMPLING_FACTOR) -> NDArray[int64]:
    # We want the first and last symbol to have a constant envelope so we are not ramping during it
    # therefore we calculate using n-2 to account for this
    k = arange((n-2)*sps, dtype=int64)
    profile = 0.5 * (1.0 - cos(pi * k / (((n-2)*sps)-1)))
    lut = np_round(profile * (1 << NUMBER_OF_FRACTIONAL_BITS)).astype(int64)
    lut[0] = 0
    lut[-1] = 1 << NUMBER_OF_FRACTIONAL_BITS

    # prepend and postpend the full symbol period 0 at the start and 1 at the end
    lut = concatenate((zeros(sps, dtype=int64), lut))
    lut = concatenate((lut, full(sps, (1 << NUMBER_OF_FRACTIONAL_BITS), dtype=int64)))

    return lut

RAMPING_LUT_4 = generate_ramping_lut_quantized(4)
RAMPING_LUT_5 = generate_ramping_lut_quantized(5)
RAMPING_LUT_6 = generate_ramping_lut_quantized(6)
RAMPING_LUT_7 = generate_ramping_lut_quantized(7)
RAMPING_LUT_8 = generate_ramping_lut_quantized(8)
RAMPING_LUT_17 = generate_ramping_lut_quantized(17)

###################################################################################################

def save_burst_iqfile(input_data: NDArray[int64] | NDArray[float32] | NDArray[int16], filepath:str,
                  dac_bits:int=PLUTOSDR_DAC_BIT_NUMBER, msb_aligned:bool=True,
                  endian:Literal["big"] | Literal["little"] ="big"):

    # PlutoSDR can read data saved as 16-bits but in 12bit format, MSB aligned
    # i.e. 12 bits but shift left 4 times
    if input_data.ndim != 2:
        raise RuntimeError(f"IQ data to save to file has passed dimensions of {input_data.ndim}, expected 2")
    # now we need to interleave the data such that it goes i_ch,q_ch|i_ch,q_ch
    temp = ravel(vstack((input_data[0], input_data[1])), order="F")
    # peform conversion and rounding to dacBit number and aligned as needed
    if input_data.dtype == int64:
        # add rounding offset
        shift = NUMBER_OF_FRACTIONAL_BITS - (dac_bits-1)
        round_word = int64(1 << ((shift)-1))
        temp = right_shift((temp + round_word), (NUMBER_OF_FRACTIONAL_BITS - (dac_bits-1))).astype(int16)

    elif input_data.dtype == int16:
        # format would be the 10bit dac values for the openTETRAphymac hw
        if (temp.max() > ((1 << OPENTETRAPHYMAC_HW_DAC_NUMBER-1)-1)
            or temp.min() < -(1<<(OPENTETRAPHYMAC_HW_DAC_NUMBER-1))):
            raise ValueError(f"Expected 10bit value with maximum + value of "
                             f"{((1 << OPENTETRAPHYMAC_HW_DAC_NUMBER-1)-1)}, found {np_max(temp)}")
        temp = left_shift(temp, (dac_bits-OPENTETRAPHYMAC_HW_DAC_NUMBER)).astype(int16)

    elif input_data.dtype == float32:
        # normalize data by the complex magnitude of the channels
        scale = (1 << (dac_bits - 1)) - 1
        peak = float(np_max(sqrt(input_data[0].astype(float64)**2 + input_data[1].astype(float64)**2)))
        if peak == 0:
            temp = zeros(shape=temp.shape, dtype=float64)
            raise RuntimeWarning("Peak complex magnitude of passed waveform to save to file is zero")

        temp = temp.astype(float64) / peak
        temp = np_round(temp * scale).astype(int16)
    else:
        raise RuntimeError(f"Datatype of passed IQ data to save to file is: {input_data.dtype},"
                           f" expected np.int64, np.int16 or np.float32")
    # 12 bits but in 16bit format MSB aligned
    if msb_aligned:
        shift = 16 - dac_bits
        temp = left_shift(temp, shift).astype(int16)

    # Check endianess
    if byteorder == endian:
        pass
    else:
        temp = temp.byteswap()

    # open file and save data
    file_path = Path(filepath)
    file_path.touch(exist_ok=True)
    try:
        temp.tofile(file_path)
    except IOError as e:
        print(e)


###################################################################################################

def read_burst_iqfile(filepath:str, output_type:Literal["int64"] | Literal["float32"]="int64",
               dac_bits:int=PLUTOSDR_DAC_BIT_NUMBER, msb_aligned:bool=False,
               endian:Literal["big"] | Literal["little"]="big"
               ) -> tuple[NDArray[int64], NDArray[int64]] | tuple[NDArray[float32], NDArray[float32]]:
    # PlutoSDR can read data saved as 16-bits but in 12bit format, MSB aligned
    # i.e. 12 bits but shift left 4 times
    if output_type not in ("int64", "float32"):
        raise ValueError(f"Passed output data type is: {output_type} expected int64 or float32")

    file_path = Path(filepath)
    temp = fromfile(file_path, dtype=int16)

    # Match endianess to system
    if byteorder == endian:
        pass
    else:
        temp = temp.byteswap()

    i_data = temp[0::2]
    q_data = temp[1::2]

    # 12 bits is typically LSB aligned in 16bit
    if msb_aligned:
        shift = 16 - dac_bits
        i_signed = (i_data >> shift).astype(int16)
        q_signed = (q_data >> shift).astype(int16)
    else:
        sign = int16(1 << (dac_bits-1))
        mask = int16((1 << dac_bits) - 1)
        i_unsigned = (i_data & mask).astype(int16)
        q_unsigned = (q_data & mask).astype(int16)
        i_signed = ((i_unsigned ^ sign) - sign).astype(int16)
        q_signed = ((q_unsigned ^ sign) - sign).astype(int16)

    # Convert data to desired output format
    if output_type == "int64":
        # We converting to Q17 format
        shift = NUMBER_OF_FRACTIONAL_BITS - (dac_bits-1)
        i_copy = left_shift(int64(i_signed), shift).astype(int64)
        q_copy = left_shift(int64(q_signed), shift).astype(int64)

    else:
        scale = float(1 << (dac_bits-1))
        i_copy = i_signed.astype(float32)
        i_copy /= scale
        q_copy = q_signed.astype(float32)
        q_copy /= scale
    return i_copy, q_copy

###################################################################################################

def oversample_data_quantized(input_data: NDArray[complex64], over_sample_rate:int) -> NDArray[int64]:
    output_data = zeros(shape=(2, input_data.size*over_sample_rate), dtype=int64)
    # because the modulation mapped only on the unit circle, the max value is 1, and min is zero,
    # the culmative baseband processing gain ~ one, so we do not need to scale to prevent overflow in this case
    temp_i_val = np_round((input_data.real * (1 << NUMBER_OF_FRACTIONAL_BITS)))
    temp_q_val = np_round((input_data.imag * (1 << NUMBER_OF_FRACTIONAL_BITS)))
    output_data[0][0::over_sample_rate] = temp_i_val.astype(int64)
    output_data[1][0::over_sample_rate] = temp_q_val.astype(int64)
    return output_data

###################################################################################################

def assert_tail_is_zero(i_ch, q_ch, samples):
    tail_p = i_ch[-samples:].astype(int64)**2 + q_ch[-samples:].astype(int64)**2
    if np_max(tail_p) != 0:
        raise RuntimeError("Burst tail is not fully gated to zero.")

###################################################################################################

def oversample_data_float(input_data: NDArray[complex64], over_sample_rate:int) -> NDArray[float32]:
    output_data = zeros(shape=(2, input_data.size*over_sample_rate), dtype=float32)
    temp_i_val = input_data.real.astype(float32)
    temp_q_val = input_data.imag.astype(float32)
    output_data[0][0::over_sample_rate] = temp_i_val.astype(float32)
    output_data[1][0::over_sample_rate] = temp_q_val.astype(float32)
    return output_data

###################################################################################################

def q17_rounding(accumulated_results:NDArray[int64], rounding:str = "rti",
                 right_shift_number:int=NUMBER_OF_FRACTIONAL_BITS):
    """
    Performs rounding on int64 object after accumulation during convolution
    """
    if rounding not in VALID_ROUNDING_METHODS:
        raise ValueError(f"Rounding method passed of {rounding} invalid, expected type in: {VALID_ROUNDING_METHODS}")

    if right_shift == 0:
        raise ValueError(f"Right shift value of: {right_shift}, invalid")
    y = accumulated_results.astype(int64, copy=True)
    match rounding:
        case "rti":
            # ECP5 FPGA style np_round to Infinity by adding 2^16 and right shifting by 17 bits
            y += int64(1 << (right_shift_number-1))
        case "rtz":
            # ECP5 FPGA style np_round to Zero by adding (2^16 - 1)
            y += int64((1 << (right_shift_number-1))-1)
        case "truncate":
            pass
        case "unbiased":
            # Software only implementation of Bankers rounding, unbiased
            sign = where(y < 0, int64(-1), int64(1))
            a = np_abs(y)
            trunc = right_shift(a, right_shift_number) # floor
            rem = a & int64((1 << right_shift_number) - 1) # remainder bits
            half = int64(1 << (right_shift_number - 1))

            gt = rem > half
            eq = rem == half

            # if exactly halfway, np_round up only if we are even value
            inc = gt | (eq & ((trunc & 1) == 1))

            trunc = trunc + inc.astype(int64)
            return trunc * sign
    return right_shift(y, right_shift_number)

###################################################################################################

def dsp_fir_quantized_stream(input_symbols:NDArray[int64], h_coefficients:NDArray[int64],
                               input_fir_state: NDArray[int64] | None = None,
                               gain:int=1) -> tuple[NDArray[int64], NDArray[int64]]:
    '''
    Handles the convolution process with continuous state control, 
    but models the 54bit accumulation result and subsequent rounding to Q17
    '''
    if gain not in [1, 2, 4, 8]:
        raise ValueError(f"Passed gain value: {gain}, is not 1 or multiple of 2.")

    n_taps = h_coefficients.size
    if input_fir_state is not None:
        if input_fir_state.size != (n_taps - 1):
            raise ValueError(f"Length of FIR state passed is {input_fir_state.size},"
                             f" expected {(n_taps - 1)} based on h_coefficients passed.")
    else:
        input_fir_state = zeros(shape=(n_taps-1), dtype=int64)

    # Prepend the state values, burst isolation flushes are handled externally to the function
    input_data_extended = concatenate((input_fir_state, input_symbols))

    # Stream the convolution results, note that prepended values come from state/memory of FIR from previous burst
    full_accumulated_result = sp_convolve(input_data_extended, h_coefficients,
                                          method="direct", mode="full").astype(int64)

    # Data-aligned segment:
    burst_segment = full_accumulated_result[(n_taps-1):(n_taps-1 + input_symbols.size)].copy()
    # The rounding implemented is not pure unbiased rounding due to behaviour with negative values,
    # however it provides greater precision than pure truncation and follow ECP5 DSP slices implementation

    if gain == 1:
        shift_value = NUMBER_OF_FRACTIONAL_BITS
    else:
        shift_value = int(NUMBER_OF_FRACTIONAL_BITS - (gain // 2))

    output_accumulated_result = q17_rounding(burst_segment, right_shift_number=shift_value)

    # Saturate output incase of clipping, it is not expected for this to occur due to the FIR gains and such
    output_accumulated_result = clip(output_accumulated_result, -(1<<NUMBER_OF_FRACTIONAL_BITS),
                             (1<<NUMBER_OF_FRACTIONAL_BITS)-1).astype(int64)

    # The new state of the FIR is the last n_taps-1 values of what was passed
    new_fir_state = input_data_extended[-(n_taps-1):].copy()

    return output_accumulated_result, new_fir_state

###################################################################################################

def dsp_fir_float_stream(input_symbols:NDArray[float32], h_coefficients:NDArray[float32],
                               input_fir_state: NDArray[float32] | None = None,
                               gain:int=1) -> tuple[NDArray[float32], NDArray[float32]]:
    '''
    Handles the convolution process with continuous state control but uses float32 values
    '''
    if gain not in [1, 2, 4, 8]:
        raise ValueError(f"Passed gain value: {gain}, is not 1 or multiple of 2.")

    n_taps = h_coefficients.size
    if input_fir_state is not None:
        if input_fir_state.size != (n_taps - 1):
            raise ValueError(f"Length of FIR state passed is {input_fir_state.size},"
                             f" expected {(n_taps - 1)} based on h_coefficients passed.")
    else:
        input_fir_state = zeros(shape=(n_taps-1), dtype=float32)

    # Prepend the state values, burst isolation flushes are handled externally to the function
    input_data_extended = concatenate((input_fir_state, input_symbols))

    # Stream the convolution results, note that prepended values come from state/memory of FIR from previous burst
    full_accumulated_result = sp_convolve(input_data_extended, h_coefficients,
                                       method="direct", mode="full").astype(float32)

    # Data-aligned segment:
    burst_segment = full_accumulated_result[(n_taps-1):(n_taps-1 + input_symbols.size)].copy()

    # Gain
    burst_segment *= gain

    # The new state of the FIR is the last n_taps-1 values of what was passed
    new_fir_state = full_accumulated_result[-(n_taps-1):].copy()

    return burst_segment, new_fir_state

###################################################################################################

def generate_ramping_float_lut(n: int, sps:int=BASEBAND_SAMPLING_FACTOR) -> NDArray[float32]:
    # We want the first and last symbol to have a constant envelope so we are not ramping during it
    # therefore we calculate using n-2 to account for this
    k = arange((n-2)*sps, dtype=float32)
    profile = 0.5 * (1.0 - cos(pi * k / (((n-2)*sps)-1)))
    lut = profile.astype(float32)
    lut[0] = 0
    lut[-1] = 1

    # prepend and postpend the full symbol period 0 at the start and 1 at the end
    lut = concatenate((zeros(sps, dtype=float32), lut))
    lut = concatenate((lut, full(sps, 1, dtype=float32)))

    return lut

###################################################################################################

def power_ramping_quantized(i_ch:NDArray[int64], q_ch:NDArray[int64],
                           burst_ramp_periods:tuple[int, int],
                           sps:int=BASEBAND_SAMPLING_FACTOR) -> tuple[NDArray[int64], NDArray[int64]]:
    """
    Performs power ramping on the quantized signal using LUTs and raised cosine shape
    """
    n = i_ch.size
    if burst_ramp_periods[0] not in VALID_START_GUARD_PERIOD_OFFSETS:
        raise ValueError(f"Passed start burst guard ramp period of {burst_ramp_periods[0]},"
                         f" expected value in: {VALID_START_GUARD_PERIOD_OFFSETS}")

    # For tetra, there are only a few defined burst delay periods,
    n_start = int(burst_ramp_periods[0] / 2) * sps
    n_end = int(burst_ramp_periods[1] / 2) * sps

    # create envelope of 1's
    envelope = full(n, (1 << NUMBER_OF_FRACTIONAL_BITS), dtype=int64)

    match int(burst_ramp_periods[0] / 2):
        case 5:
            # Discontinuous SB or NDB
            up_ramp = RAMPING_LUT_5
        case 6:
            # Continuous SB or NDB
            up_ramp = RAMPING_LUT_6
        case 17:
            # NUB or CB
            up_ramp = RAMPING_LUT_17
        case 240:
            raise NotImplementedError("Uplink linearization (LB) ramping not implemented yet")

    if int(burst_ramp_periods[0] / 2) != 0:
        envelope[:n_start] = up_ramp

    if burst_ramp_periods[1] not in VALID_END_GUARD_PERIOD_OFFSETS:
        raise ValueError(f"Passed end burst guard ramp period of {burst_ramp_periods[1]},"
                         f" expected value in: {VALID_END_GUARD_PERIOD_OFFSETS}")

    match int(burst_ramp_periods[1] / 2):
        case 4:
            # Discontinuous SB or NDB
            down_ramp = RAMPING_LUT_4
        case 5:
            # Continuous SB or NDB
            down_ramp = RAMPING_LUT_5
        case 7:
            # NUB or CB or LB
            down_ramp = RAMPING_LUT_7
        case 8:
            down_ramp = RAMPING_LUT_8

    if int(burst_ramp_periods[1] / 2) != 0:
        envelope[-n_end:] = down_ramp[::-1]

    i_ch_product = i_ch.astype(int64) * envelope.astype(int64)
    q_ch_product = q_ch.astype(int64) * envelope.astype(int64)

    i_ch_result = q17_rounding(i_ch_product.copy())
    q_ch_result = q17_rounding(q_ch_product.copy())
    return i_ch_result, q_ch_result

###################################################################################################

def power_ramping_float(i_ch:NDArray[float32], q_ch:NDArray[float32],
                           burst_ramp_periods:tuple[int, int],
                           sps:int=BASEBAND_SAMPLING_FACTOR) -> tuple[NDArray[float32], NDArray[float32]]:
    """
    Performs power ramping on the quantized signal using LUTs and raised cosine shape
    """
    n = i_ch.size
    if burst_ramp_periods[0] not in VALID_START_GUARD_PERIOD_OFFSETS:
        raise ValueError(f"Passed start burst guard ramp period of {burst_ramp_periods[0]},"
                         f" expected value in: {VALID_START_GUARD_PERIOD_OFFSETS}")
    # For tetra, there are only a few defined burst delay periods,
    n_start = int(burst_ramp_periods[0] / 2) * sps
    n_end = int(burst_ramp_periods[1] / 2) * sps

    # create envelope of 1's
    envelope = full(n, 1, dtype=float32)
    match int(burst_ramp_periods[0] / 2):
        case 5:
            # Discontinuous SB or NDB
            up_ramp = generate_ramping_float_lut(5)
        case 6:
            # Continuous SB or NDB
            up_ramp = generate_ramping_float_lut(6)
        case 17:
            # NUB or CB
            up_ramp = generate_ramping_float_lut(17)
        case 240:
            raise NotImplementedError("Uplink linearization (LB) ramping not implemented yet")

    if int(burst_ramp_periods[0] / 2) != 0:
        envelope[:n_start] = up_ramp

    if burst_ramp_periods[1] not in VALID_END_GUARD_PERIOD_OFFSETS:
        raise ValueError(f"Passed end burst guard ramp period of {burst_ramp_periods[1]},"
                         f" expected value in: {VALID_END_GUARD_PERIOD_OFFSETS}")

    match int(burst_ramp_periods[1] / 2):
        case 4:
            # Discontinuous SB or NDB
            down_ramp = generate_ramping_float_lut(4)
        case 5:
            # Continuous SB or NDB
            down_ramp = generate_ramping_float_lut(5)
        case 7:
            # NUB or CB or LB
            down_ramp = generate_ramping_float_lut(7)
        case 8:
            down_ramp = generate_ramping_float_lut(8)

    if int(burst_ramp_periods[1] / 2) != 0:
        envelope[-n_end:] = down_ramp[::-1]

    i_ch_product = i_ch.astype(float32) * envelope.astype(float32)
    q_ch_product = q_ch.astype(float32) * envelope.astype(float32)

    return i_ch_product.astype(float32), q_ch_product.astype(float32)
