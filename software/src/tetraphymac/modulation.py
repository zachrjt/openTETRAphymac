"""
modulation.py contains functions that peform modulation or demodulation of data into or out of numpy complex data
into binary modulation bits.
"""
from typing import Literal
from numpy import cumprod, complex64, pi, float64, uint8, mod, array, exp, dtype, ndarray, \
    concatenate, full
from numpy import sum as np_sum
from numpy import abs as np_abs
from numpy.typing import NDArray

# Pi/4 DQPSK LUT used to convert modulation bits into complex symbols
DQPSK_PHASE_TRANSITION_LUT = array([pi/4, 3*pi/4, -pi/4, -3*pi/4], dtype=float64)
DQPSK_TRANSITION_PHASOR_LUT = exp(1j * DQPSK_PHASE_TRANSITION_LUT).astype(complex64)


def calculate_phase_adjustment_bits(input_data: NDArray[uint8],
                                    inclusive_indices: tuple[int, int],
                                    guard_start_offset: int = 0) -> ndarray[tuple[Literal[2]], dtype[uint8]]:
    """
    Modulates input data within the inclusiveIndices accounting for a guardoffset, to determine the total
    cummulative phase and returns the correct bit pair that would set the cumulative phase to zero

    :param input_data: Binary 1's and 0's input to evaluate total Pi/4-dqpsk phase angle over
    :type input_data: NDArray[uint8]
    :param inclusiveIndices: The start and stop indices (inclusive) to evaluate the cumulative phase
    :type inclusiveIndices: tuple[int, int]
    :param guardOffset: An offset to account for an initial guard period in input data, offsets inclusive indices
    :type guardOffset: int
    :return: Returns two bits that if added into the cumulative phase results in the cumulative phase being zero
    :rtype: NDArray[uint8]
    """
    if input_data.ndim != 1:
        raise ValueError(f"Input data dimensions are: {input_data.ndim}, expected 1")
    if input_data.size % 2 != 0:    # must have even number of bits
        raise ValueError(f"Number of input burst bits is: {input_data.size}, expected even number")

    # reshape to have even and odd bits [b0, b1], also skip the first guardOffset bits to get to data
    bit_pairs = input_data[guard_start_offset:]
    bit_pairs = bit_pairs[inclusive_indices[0]*2:(inclusive_indices[1]*2)+2].reshape(-1, 2)

    # map the even odd bits into a 4-entry code to map phase transistion quickly from LUT
    coded_transistions = (bit_pairs[:, 0] << 1) | bit_pairs[:, 1]   # maps into value [b0b1, b2b3, ...]
    # grayTransistion -> 0=0b00, 1=0b01, 2=0b10, 3=0b11, lookup the phase transistion from table

    # if you get an out of range error here, then your indices are off and are
    # reading random values in unset phase adjustment bits
    dphi = DQPSK_PHASE_TRANSITION_LUT[coded_transistions]

    phi_accumulated = float64(np_sum(dphi))

    resultant_angle = mod(-phi_accumulated, (2*pi))
    if resultant_angle > pi:
        resultant_angle -= (2*pi)
    resultant_bit_pair = np_abs(DQPSK_PHASE_TRANSITION_LUT - resultant_angle).argmin()

    bits = array([(resultant_bit_pair >> 1) & 1, resultant_bit_pair & 1], dtype=uint8)

    return bits

###################################################################################################


def dqpsk_modulator(input_data: NDArray[uint8],
                    burst_ramp_periods: tuple[int, int],
                    phase_ref: complex64 = complex64(1 + 0j)) -> NDArray[complex64]:
    """
    Performs pi/4-DQPSK modulation on binary input_data arrays, with a starting phase reference for differential
    modulation starting at phase_rf, maintains a constant phase during the first burst_ramp_period[0] time as phase ref
    and then a constant phase during the last [:-burst_ramp_periods[1]] period, maintaining the last phase of the end of
    the payload.

    :param input_data: input binary values stored in a uint8 array, must have an even numbered length
    :type input_data: NDArray[uint8]
    :param burst_ramp_periods: Stores the length of the start ramp period, and end ramp period in bits
    :type burst_ramp_periods: tuple[int, int]
    :param phase_ref: Defaults to (1 + 0j), if continuous tx', pass the end phase of the prev. burst
    :type phase_ref: complex64
    :return: pi/4-DPSK modulated output unit-circle complex data, with half the length of input_data
    :rtype: NDArray[complex64]
    """

    if input_data.ndim != 1:
        raise ValueError(f"Input data dimensions are: {input_data.ndim}, expected 1")

    if input_data.size % 2 != 0:    # must have even number of bits
        raise ValueError(f"Number of input burst bits is: {input_data.size}, expected even number")

    if burst_ramp_periods[0] % 2 != 0:
        raise ValueError(f"Start ramp guard period in number of bits is: {burst_ramp_periods[0]}"
                         f", expected even number")
    if burst_ramp_periods[1] % 2 != 0:
        raise ValueError(f"End ramp guard period in number of bits is: {burst_ramp_periods[1]}"
                         f", expected even number")

    n_start = int(burst_ramp_periods[0] / 2)    # in number of symbols
    n_end = int(burst_ramp_periods[1] / 2)      # in number of symbols

    # The ramping period at the start and end shall have constant phase,
    # Therefore we exclude the ramping periods from the calculations,
    # Note that the start ramping phase is constant and equal to reference phase,
    # while the final down ramp phase is constant and equal to
    # whichever was the last phase in the useful part of the burst
    # This prevents phase discontinuity

    # reshape to have even and odd bits [b0, b1]
    useful_burst_seg = input_data[(n_start * 2): input_data.size - (n_end * 2)].copy()
    bit_pairs = useful_burst_seg.reshape(-1, 2)

    # map the even odd bits into a 4-entry code to map phase transistion quickly from LUT
    coded_transistions = (bit_pairs[:, 0] << 1) | bit_pairs[:, 1]   # maps into value [b0b1, b2b3, ...]

    # grayTransistion -> 0=0b00, 1=0b01, 2=0b10, 3=0b11, lookup the phase transistion table in phasor form
    d_phasor = DQPSK_TRANSITION_PHASOR_LUT[coded_transistions]
    burst_seg = (cumprod(d_phasor) * phase_ref).astype(complex64)

    # prepend and postpend the constant phase ramping periods if they are needed
    if n_start > 0:
        start_ramp_phase = full(n_start, phase_ref, dtype=complex64)
        burst_seg = concatenate((start_ramp_phase, burst_seg)).astype(complex64)

    if n_end > 0:
        end_phase = burst_seg[-1] if burst_seg.size > 0 else phase_ref
        end_ramp_phase = full(n_end, end_phase, dtype=complex64)
        burst_seg = concatenate((burst_seg, end_ramp_phase)).astype(complex64)

    return burst_seg
