"""
transmitter.py provides ideal and realistic non-ideal emulation of the openTETRAphymac hardware implementation.

The basis of the transmitter implementations is a semi-abstract RFTransmitter parent class
that implements inheritied methods which wrap handling of conversion of modulation bits in transmit baseband data and
are public, while leaving abstract private methods to handle baseband processing, dac conversion,
analog reconstruction, etc.

This is done so that an ideal and real implementation can provide differening implementations such as quantized
processing or ideal float processing, non-linearities, and no non-linearities.

The child implementations were designed to be used in the following fashions:
RealTransmitter is designed to support hardware development by allowing for cascaded DSP development
and simulation of EVM, ACPL, time-power ramping masks, wideband noise, etc

IdealTransmitter is designed to generate ideal transmit bursts for testing rcvr software and hardware, and BER and MER
testing.
"""
from abc import ABC, abstractmethod
from typing import ClassVar, Literal
from numpy import complex64, float64, uint8, zeros, mean, \
    int64, concatenate, full, sqrt, zeros_like, repeat, sin, cos, complex128
from numpy.random import Generator, PCG64
from numpy.typing import NDArray
from scipy.signal import bessel, sosfilt

from .tx_rx_utilities import power_ramping_float, \
    power_ramping_quantized, assert_tail_is_zero, oversample_data_quantized, oversample_data_float, \
    TX_HALFBAND1_FLOAT_COEFFICIENTS, TX_HALFBAND1_Q17_COEFFICIENTS, TX_HALFBAND2_FLOAT_COEFFICIENTS, \
    TX_HALFBAND2_Q17_COEFFICIENTS, TX_HALFBAND3_FLOAT_COEFFICIENTS, TX_HALFBAND3_Q17_COEFFICIENTS, \
    TX_LPF_FLOAT_COEFFICIENTS, TX_LPF_Q17_COEFFICIENTS, TX_RRC_FLOAT_COEFFICIENTS, TX_RRC_Q17_COEFFICIENTS, \
    dsp_fir_i_q_stream_convolve, OPENTETRAPHYMAC_HW_DAC_BIT_NUMBER, q17_rounding, NUMBER_OF_FRACTIONAL_BITS


from .constants import SUBSLOT_BIT_LENGTH, TX_BB_SAMPLING_FACTOR
from .modulation import dqpsk_modulator

VALID_RETURN_STAGE_VALUES = ("baseband", "dac", "tx")  # possible output stages from transmitter.transmitBurst(),

HALF_BASEBAND_SAMPLING_FACTOR = int(TX_BB_SAMPLING_FACTOR / 2)   # Half of the culmative baseband sampling factor
TETRA_SYMBOL_RATE = 18000                                           # The base EN 300 392-2 symbol rate

# The internal sw simulator sampling factor over the DAC rate
TRANSMIT_SIMULATION_SAMPLING_FACTOR = 10
# Internal sw simulator sampling rate, allows for capture of harmonics
TRANSMIT_SIMULATION_SAMPLE_RATE = int(TX_BB_SAMPLING_FACTOR * TETRA_SYMBOL_RATE * TRANSMIT_SIMULATION_SAMPLING_FACTOR)
# Number of base sample rampe sames used to prepend and post bend burst data to clear FIR memory states
BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT = 30

###################################################################################################


class RFTransmitter(ABC):
    """
    RFTransmitter is a base semi-abstract parent class that implements methods to
    accuractely model openTETRAphymac hardware.

    It implements baseband processing, dac conversion, and analog reconstruction, to provide a basis that can be as
    in depth in modeling non-linearities and errors as the child implementation desires.

    Its' primary method is transmit_burst, which converts input modulation bits of a burst and information regarding
    the ramping into output samples that can model non-linearity, quantization effects, and anything else the
    implementation does or does not desire, in the abstract _baseband_processing, _dac_conversion_, and
    _analog_reconstruction methods.

    It is flexible and allows for quantized Q17 stored in int64 or ideal float64 handling of data
    between abstract methods.
    """

    phase_reference = complex64(1 + 0j)

    rrc_filter_state: ClassVar[NDArray[int64 | float64]]
    lpf_filter_state: ClassVar[NDArray[int64 | float64]]
    halfband_1_filter_state: ClassVar[NDArray[int64 | float64]]
    halfband_2_filter_state: ClassVar[NDArray[int64 | float64]]
    halfband_3_filter_state: ClassVar[NDArray[int64 | float64]]

    _error_generation_state = False
    _genI_generator = Generator(PCG64())
    _genQ_generator = Generator(PCG64())

    i_ch_ofst_correction = float64(0)
    q_ch_ofst_correction = float64(0)
    i_ch_gain_correction = float64(1)
    q_ch_gain_correction = float64(1)

    _i_ch_offset_err = float64(0)
    _q_ch_offset_err = float64(0)
    _i_ch_gain_err = float64(0)
    _q_ch_gain_err = float64(0)

    _q_ch_phase_err = float(0.0174533)
    _q_ch_phase_correction = float(0)

    def __init__(self):
        pass

    @abstractmethod
    def _baseband_processing(self, symbol_complex_data: NDArray[complex64],
                             burst_ramp_periods: tuple[int, int]
                             ) -> tuple[NDArray[int64 | float64], NDArray[int64 | float64]]:
        """
        Base abstract method that converts modulation bits into symbols, performs upsampling, ramping, and filtering,
        """
        raise NotImplementedError

    @abstractmethod
    def _dac_conversion(self, i_ch: NDArray[int64 | float64], q_ch: NDArray[int64 | float64]
                        ) -> tuple[NDArray[float64], NDArray[float64]]:
        """
        Base abstract method that converts baseband processed data into dac code representation,
        """
        raise NotImplementedError

    @abstractmethod
    def _analog_reconstruction(self, i_ch: NDArray[float64],
                               q_ch: NDArray[float64]
                               ) -> NDArray[complex128]:
        """
        base abstract method that takes in DAC codes at rate Rs, converts to real floats with ZOH with
        sampling rate Rif which is x9 more than Rs, then filters with analog reconstruction filter.
        """
        # coupling
        # gain error
        # offset
        raise NotImplementedError

    def _handle_sublot_burst(self, burst_bit_seq: NDArray[uint8], burst_ramp_periods: tuple[int, int],
                             subslot_2_burst_ramp_periods: tuple[int, int] | None = None
                             ) -> tuple[NDArray[int64 | float64], NDArray[int64 | float64]]:
        """
        Helper function used to handle dealing with subslot bursts which require additional processing to stich together
        . This function handles modulating, upsampling, filtering, and power ramping individaully, returning ramped
        i and q channel data that is ready for subsequent handling in transmit_burst.

        :param self: RFTransmitter child implementation
        :param burst_bit_seq: input binary data stored in uint8 array for a burst slot, can be full or subslot burst(s)
        :type burst_bit_seq: NDArray[uint8]
        :param burst_ramp_periods: Stores the length of the start ramp period, and end ramp period in bits
        :type burst_ramp_periods: tuple[int, int]
        :param subslot_2_burst_ramp_periods: Like burst_ramp_periods, but used when a subslot burst in subslot 2 is used
        :type subslot_2_burst_ramp_periods: tuple[int, int] | None
        :return: Returns ramped int64 or float64 data, type dependent on RFTransmitter child implementation, that has
        been modulated, upsampled, filtered, and power ramped.
        :rtype: tuple[NDArray[int64 | float64], NDArray[int64 | float64]]
        """
        null_subslot_index = 0
        # multiple slot bursts or subslot bursts passed
        if burst_bit_seq.shape[1] == SUBSLOT_BIT_LENGTH:
            # Passed 2 sublots for a single burst, acceptable
            # If the blocks are only SUBLOT length, then we are transmitting a single full slot burst
            # Could be of form: [CB, empty], [LB, CB], [empty, CB], or [LB, empty]
            if burst_bit_seq.shape[0] != 2:
                raise ValueError(f"Passed {burst_bit_seq.shape[0]} subslots to transmit,"
                                 f" expected exactly 2 to handle subslot tx")

            # Determine where/if there is a null subslot
            if (burst_bit_seq[0] == 0).all():
                null_subslot_index = 0
                number_of_used_subslots = 1
            elif (burst_bit_seq[1] == 0).all():
                null_subslot_index = 1
                number_of_used_subslots = 1
            else:
                # There are no empty subslots
                number_of_used_subslots = 2

        else:
            raise ValueError(f"Passed {burst_bit_seq.shape[1]} modulation bits,"
                             f" expected 2 subslots of length {(SUBSLOT_BIT_LENGTH)}")

        # Single burst made from 1 to 2 subslots bursts (may have null/empty subslot) with odd number of bits.
        # Therefore we postpend an additional modulation bit to the end of each subslot
        # Because the last symbol is entirely zero, and we simply truncate the burst at the end
        # to only be zero for half a symbol at the ramp-downs of each subslot in order to lineup the timing overall

        # 3. Phase reference is always default since half-slot bursts are noncontinuous

        if number_of_used_subslots == 2 and subslot_2_burst_ramp_periods is not None:
            # we have two subslot bursts to modulate indepedently.
            # must increment the end guard period of the first and second subslot burst to
            # account for the need for the additional modulation bit

            # 4. Modulate the burst bits into 255 symbols
            ssb1_burst_ramp_periods = (burst_ramp_periods[0], burst_ramp_periods[1]+1)
            ssb2_burst_ramp_periods = (subslot_2_burst_ramp_periods[0], subslot_2_burst_ramp_periods[1]+1)
            sbs1_symbol_complex_data = dqpsk_modulator(concatenate((burst_bit_seq[0], zeros(1, dtype=uint8))),
                                                       ssb1_burst_ramp_periods)
            sbs2_symbol_complex_data = dqpsk_modulator(concatenate((burst_bit_seq[1], zeros(1, dtype=uint8))),
                                                       ssb2_burst_ramp_periods)

            # 5. Pass modulated symbols and guard information to baseband processing function
            i_ch_sbs_1, q_ch_sbs_1 = self._baseband_processing(sbs1_symbol_complex_data, ssb1_burst_ramp_periods)
            i_ch_sbs_2, q_ch_sbs_2 = self._baseband_processing(sbs2_symbol_complex_data, ssb2_burst_ramp_periods)

            # concatenate the two subslots, indexing such that the extra tail modulation bits are eliminated
            # verify that the tail does go to zero otherwise we get a discontinuity
            assert_tail_is_zero(i_ch_sbs_1, q_ch_sbs_1, HALF_BASEBAND_SAMPLING_FACTOR)
            assert_tail_is_zero(i_ch_sbs_2, q_ch_sbs_2, HALF_BASEBAND_SAMPLING_FACTOR)

            i_temp_ramp = concatenate((i_ch_sbs_1[:i_ch_sbs_1.size - HALF_BASEBAND_SAMPLING_FACTOR],
                                       i_ch_sbs_2[:i_ch_sbs_2.size - HALF_BASEBAND_SAMPLING_FACTOR]))
            q_temp_ramp = concatenate((q_ch_sbs_1[:q_ch_sbs_1.size - HALF_BASEBAND_SAMPLING_FACTOR],
                                       q_ch_sbs_2[:q_ch_sbs_2.size - HALF_BASEBAND_SAMPLING_FACTOR]))

        elif null_subslot_index == 0 and subslot_2_burst_ramp_periods is not None:
            # First subslot is null burst, while second subslot is real burst
            ssb2_burst_ramp_periods = (subslot_2_burst_ramp_periods[0], subslot_2_burst_ramp_periods[1]+1)
            sbs2_symbol_complex_data = dqpsk_modulator(concatenate((burst_bit_seq[1], zeros(1, dtype=uint8))),
                                                       ssb2_burst_ramp_periods)
            i_ch_sbs_2, q_ch_sbs_2 = self._baseband_processing(sbs2_symbol_complex_data, ssb2_burst_ramp_periods)

            # generate equivalent number of zeros to fill null sublot
            i_ch_sbs_1 = zeros((SUBSLOT_BIT_LENGTH+1)*64, dtype=(int64 if isinstance(self, RealTransmitter)
                                                                 else float64))
            q_ch_sbs_1 = zeros((SUBSLOT_BIT_LENGTH+1)*64, dtype=(int64 if isinstance(self, RealTransmitter)
                                                                 else float64))

            # verify that the tail does go to zero otherwise we get a discontinuity
            assert_tail_is_zero(i_ch_sbs_2, q_ch_sbs_2, HALF_BASEBAND_SAMPLING_FACTOR)

            i_temp_ramp = concatenate((i_ch_sbs_1[:i_ch_sbs_1.size - HALF_BASEBAND_SAMPLING_FACTOR],
                                       i_ch_sbs_2[:i_ch_sbs_2.size - HALF_BASEBAND_SAMPLING_FACTOR]))
            q_temp_ramp = concatenate((q_ch_sbs_1[:q_ch_sbs_1.size - HALF_BASEBAND_SAMPLING_FACTOR],
                                       q_ch_sbs_2[:q_ch_sbs_2.size - HALF_BASEBAND_SAMPLING_FACTOR]))

        elif null_subslot_index == 1:
            # Second subslot is null burst, while first subslot is real burst
            ssb1_burst_ramp_periods = (burst_ramp_periods[0], burst_ramp_periods[1]+1)
            sbs1_symbol_complex_data = dqpsk_modulator(concatenate((burst_bit_seq[0], zeros(1, dtype=uint8))),
                                                       ssb1_burst_ramp_periods)

            i_ch_sbs_1, q_ch_sbs_1 = self._baseband_processing(sbs1_symbol_complex_data, ssb1_burst_ramp_periods)

            # generate equivalent number of zeros to fill null sublot
            i_ch_sbs_2 = zeros((SUBSLOT_BIT_LENGTH+1)*64, dtype=(int64 if isinstance(self, RealTransmitter)
                                                                 else float64))
            q_ch_sbs_2 = zeros((SUBSLOT_BIT_LENGTH+1)*64, dtype=(int64 if isinstance(self, RealTransmitter)
                                                                 else float64))

            # verify that the tail does go to zero otherwise we get a discontinuity
            assert_tail_is_zero(i_ch_sbs_1, q_ch_sbs_1, HALF_BASEBAND_SAMPLING_FACTOR)

            i_temp_ramp = concatenate((i_ch_sbs_1[:i_ch_sbs_1.size - HALF_BASEBAND_SAMPLING_FACTOR],
                                       i_ch_sbs_2[:i_ch_sbs_2.size - HALF_BASEBAND_SAMPLING_FACTOR]))
            q_temp_ramp = concatenate((q_ch_sbs_1[:q_ch_sbs_1.size - HALF_BASEBAND_SAMPLING_FACTOR],
                                       q_ch_sbs_2[:q_ch_sbs_2.size - HALF_BASEBAND_SAMPLING_FACTOR]))

        else:
            raise ValueError("subslot2RampPeriod cannot be None, using the second subslot it is expected"
                             " to represent the bit interval if odd")

        return i_temp_ramp, q_temp_ramp

    def transmit_burst(self, burst_bit_seq: NDArray[uint8], burst_ramp_periods: tuple[int, int],
                       subslot_2_burst_ramp_periods: tuple[int, int] | None = None,
                       debug_return_stage: Literal["baseband"] | Literal["dac"] | Literal["tx"] = "tx"
                       ) -> tuple[NDArray[int64 | float64], NDArray[int64 | float64]] | NDArray[complex128]:
        """
        Performs the process of converting burst modulation bits into transmitted data, including baseband processing,
        DAC conversion, and analog reconstruction

        :param self: RFTransmitter child implementation
        :param burst_bit_seq: input binary data stored in uint8 array for a burst slot, can be full or subslot burst(s)
        :type burst_bit_seq: NDArray[uint8]
        :param burst_ramp_periods: Stores the length of the start ramp period, and end ramp period in bits
        :type burst_ramp_periods: tuple[int, int]
        :param subslot_2_burst_ramp_periods: Like burst_ramp_periods, but used when a subslot burst in subslot 2 is used
        :type subslot_2_burst_ramp_periods: tuple[int, int] | None
        :param debug_return_stage: A string value that allows for early return to capture outputs at
         different internal stages
        :type debug_return_stage: Literal["baseband"] | Literal["dac"] | Literal["tx"]
        :return: Dependant on debug_return_stage, but either returns int64 quantized data directly or float64 data for
         a full "tx" return at the non-debug return end point, representing the baseband signal after transmission
         with nonlinearities and errors as implemented by the RFTransmitter child
        :rtype: tuple[NDArray[int64 | float64], NDArray[int64 | float64]]
        """

        if debug_return_stage not in VALID_RETURN_STAGE_VALUES:
            raise RuntimeError(f"Passed debug return stage for transmitBurst of: {debug_return_stage} invalid,"
                               f" expected value in: {VALID_RETURN_STAGE_VALUES}")

        # 1. Determine handling of input burstbitSequence
        halfslot_state = False

        if burst_bit_seq.ndim == 2:
            halfslot_state = True
        elif burst_bit_seq.ndim == 1:
            # Passed only one full slot burst, acceptable if sufficent number of modulation bits
            if burst_bit_seq.size != (2*SUBSLOT_BIT_LENGTH):
                raise ValueError(f"Passed burstbitSequences of shape: {burst_bit_seq.shape},"
                                 f" invalid number of modulation bits, expected {2*SUBSLOT_BIT_LENGTH}")
        else:
            raise ValueError(f"Passed burstbitSequences of shape: {burst_bit_seq.shape},"
                             f" invalid number of dimensions or invalid number of modulation bits")

        if halfslot_state:
            i_temp_ramp, q_temp_ramp = self._handle_sublot_burst(burst_bit_seq, burst_ramp_periods,
                                                                 subslot_2_burst_ramp_periods)
        else:
            # Full slot burst
            # if we are not ramping up at the start of the burst, then we can assume it is continuous with
            # the previous burst and use the the internal phase reference state
            burst_ramp_state = (burst_ramp_periods[0] != 0, burst_ramp_periods[1] != 0)
            burst_phase_ref = complex64(1 + 0j) if burst_ramp_state[0] else self.phase_reference

            # 4. Modulate the burst bits into 255 symbols
            input_symbol_complex_data = dqpsk_modulator(burst_bit_seq, burst_ramp_periods, burst_phase_ref)
            # if we ramp down at the end, reset the phase reference for the next burst, if we don't ramp down then
            # it is assumed phase continuous and we set the reference state to the last phase of the burst
            self.phase_reference = input_symbol_complex_data[-1] if not burst_ramp_state[1] else complex64(1 + 0j)

            # 5. Pass modulated symbols and guard information to baseband processing function
            i_temp_ramp, q_temp_ramp = self._baseband_processing(input_symbol_complex_data, burst_ramp_periods)

        if debug_return_stage == "baseband":
            return i_temp_ramp, q_temp_ramp

        # 6. Convert to DAC representation
        i_float, q_float = self._dac_conversion(i_temp_ramp, q_temp_ramp)

        if debug_return_stage == "dac":
            return i_float, q_float
        # 7. Create tx representation

        rf_data = self._analog_reconstruction(i_float, q_float)

        return rf_data


###################################################################################################

class RealTransmitter(RFTransmitter):
    """
    Implementation of the parent RFTransmitter base class which uses quantized Q17 data in its' baseband processing
    method and adds in realistic nonlinearities based on the openTETRAphymac hardware design, such as
    dac non-linearity, I, Q channel gain and offset differences, realistic reconstruction filters, and phase noise from
    modulator LO.
    """

    rrc_filter_state = zeros(shape=(2, len(TX_RRC_Q17_COEFFICIENTS)-1), dtype=int64)
    lpf_filter_state = zeros(shape=(2, len(TX_LPF_Q17_COEFFICIENTS)-1), dtype=int64)
    halfband_1_filter_state = zeros(shape=(2, len(TX_HALFBAND1_Q17_COEFFICIENTS)-1), dtype=int64)
    halfband_2_filter_state = zeros(shape=(2, len(TX_HALFBAND2_Q17_COEFFICIENTS)-1), dtype=int64)
    halfband_3_filter_state = zeros(shape=(2, len(TX_HALFBAND3_Q17_COEFFICIENTS)-1), dtype=int64)

    def _baseband_processing(self, symbol_complex_data: NDArray[complex64],
                             burst_ramp_periods: tuple[int, int]) -> tuple[NDArray[int64], NDArray[int64]]:
        """
        Docstring for _baseband_processing

        :param self: Description
        :param symbol_complex_data: Description
        :type symbol_complex_data: NDArray[complex64]
        :param burst_ramp_periods: Description
        :type burst_ramp_periods: tuple[int, int]
        :return: Description
        :rtype: tuple[NDArray[int64], NDArray[int64]]
        """
        # 1. Determine if prepending and/or postpending zeros to flush is required
        start_offset = 0
        end_offset = 0
        # 1a. continuous with previous burst consideration:
        if burst_ramp_periods[0] != 0:
            # Since we ramp up, we are not continuous with previous data, and must flush the FIRs with prepended zeros
            insulated_input_data = concatenate((full(shape=BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT,
                                               fill_value=complex64(1 + 0j), dtype=complex64),
                                               symbol_complex_data))
            start_offset = BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT

        else:
            insulated_input_data = symbol_complex_data.copy()

        # 1b. continuous with subsequent burst consideration:
        if burst_ramp_periods[1] != 0:
            # Since we ramp down at the end, we are not continuous afterwards and
            # should flush data with postpended zeros
            insulated_input_data = concatenate((insulated_input_data, full(shape=BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT,
                                               fill_value=complex64(1 + 0j), dtype=complex64)))
            end_offset = BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT

        mag = sqrt(insulated_input_data.real[(start_offset):
                                             len(insulated_input_data.real)-(end_offset)].astype(float64)**2
                   + insulated_input_data.imag[(start_offset):
                                               len(insulated_input_data.imag)-(end_offset)].astype(float64)**2)

        print("\nPre x8 upsampling - symbol mapper ", "peakFS: ", mag.max()/(1), "rmsFS: ", sqrt(mean(mag**2))/(1))

        # 2. Upsample by x8 with zero insertions, and quantize data to Q1.17 fixed format stored in int64
        stage_1_symbols = oversample_data_quantized(insulated_input_data, 8)

        # 3. Perform RRC filtering
        stage_2_symbols = zeros_like(stage_1_symbols)
        stage_2_symbols[0], stage_2_symbols[1], self.rrc_filter_state[0], self.rrc_filter_state[1] = \
            dsp_fir_i_q_stream_convolve(stage_1_symbols[0], stage_1_symbols[1], TX_RRC_Q17_COEFFICIENTS,
                                        self.rrc_filter_state, 4)

        # 4. Perform cleanup LPF'ing
        stage_3_symbols = zeros_like(stage_2_symbols)
        stage_3_symbols[0], stage_3_symbols[1], self.lpf_filter_state[0], self.lpf_filter_state[1] = \
            dsp_fir_i_q_stream_convolve(stage_2_symbols[0], stage_2_symbols[1], TX_LPF_Q17_COEFFICIENTS,
                                        self.lpf_filter_state, 1)

        # 5. Perform x2 upsampling with zero insertions and filter - Part 1
        stage_4_symbols = zeros(shape=(2, 2*stage_3_symbols.shape[1]), dtype=int64)
        stage_4_symbols[:, ::2] = stage_3_symbols

        stage_5_symbols = zeros_like(stage_4_symbols)
        stage_5_symbols[0], stage_5_symbols[1], self.halfband_1_filter_state[0], self.halfband_1_filter_state[1] = \
            dsp_fir_i_q_stream_convolve(stage_4_symbols[0], stage_4_symbols[1], TX_HALFBAND1_Q17_COEFFICIENTS,
                                        self.halfband_1_filter_state, 2)

        # 6. Perform x2 upsampling with zero insertions and filter - Part 2
        stage_6_symbols = zeros(shape=(2, 2*stage_5_symbols.shape[1]), dtype=int64)
        stage_6_symbols[:, ::2] = stage_5_symbols

        stage_7_symbols = zeros_like(stage_6_symbols)
        stage_7_symbols[0], stage_7_symbols[1], self.halfband_2_filter_state[0], self.halfband_2_filter_state[1] = \
            dsp_fir_i_q_stream_convolve(stage_6_symbols[0], stage_6_symbols[1], TX_HALFBAND2_Q17_COEFFICIENTS,
                                        self.halfband_2_filter_state, 2)

        # 7. Perform x2 upsampling with zero insertions and filter - Part 3
        stage_8_symbols = zeros(shape=(2, 2*stage_7_symbols.shape[1]), dtype=int64)
        stage_8_symbols[:, ::2] = stage_7_symbols

        stage_9_symbols = zeros_like(stage_8_symbols)
        stage_9_symbols[0], stage_9_symbols[1], self.halfband_3_filter_state[0], self.halfband_3_filter_state[1] = \
            dsp_fir_i_q_stream_convolve(stage_8_symbols[0], stage_8_symbols[1], TX_HALFBAND3_Q17_COEFFICIENTS,
                                        self.halfband_3_filter_state, 2)

        # 8. Extract useful part of burst
        full_pad_length = int(BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT*TX_BB_SAMPLING_FACTOR)
        if burst_ramp_periods[0] != 0:
            stage_9_symbols = stage_9_symbols[:, full_pad_length:].copy()
        if burst_ramp_periods[1] != 0:
            stage_9_symbols = stage_9_symbols[:, :-full_pad_length].copy()

        # 9. Perform ramping on signal
        i_ramped_symbols, q_ramped_symbols = power_ramping_quantized(stage_9_symbols[0], stage_9_symbols[1],
                                                                     burst_ramp_periods)
        mag = sqrt(i_ramped_symbols.astype(float64)**2 + q_ramped_symbols.astype(float64)**2)
        print("Post ramping ", "peakFS: ", mag.max()/(1 << 17), "rmsFS: ", sqrt(mean(mag**2))/(1 << 17))

        return i_ramped_symbols, q_ramped_symbols

    def _dac_conversion(self, i_ch: NDArray[int64 | float64], q_ch: NDArray[int64 | float64]
                        ) -> tuple[NDArray[float64], NDArray[float64]]:
        """
        Converts baseband processed data into float representation with realistic dac non-linearity
        """
        # 1. Round Q17 data to 10bits for DAC
        i_dac_bits = q17_rounding(i_ch.copy().astype(int64),
                                  "rtz", NUMBER_OF_FRACTIONAL_BITS - (OPENTETRAPHYMAC_HW_DAC_BIT_NUMBER-1))
        q_dac_bits = q17_rounding(q_ch.copy().astype(int64),
                                  "rtz", NUMBER_OF_FRACTIONAL_BITS - (OPENTETRAPHYMAC_HW_DAC_BIT_NUMBER-1))

        # 2. Convert to float64 values
        i_float_data = i_dac_bits.astype(float64) / float(1 << 9)
        q_float_data = q_dac_bits.astype(float64) / float(1 << 9)

        # 3. Now we can add non-linearities which will be modelled as a uniform source as described by
        #    10.1109/IC3I.2014.7019821 Taheri S. M, and Mohammadi B.
        if not self._error_generation_state:
            # Generate channel offset errors
            self._i_ch_offset_err = self._genI_generator.uniform(-0.001, +0.001, 1)[0]
            self._q_ch_offset_err = self._genQ_generator.uniform(-0.001, +0.001, 1)[0]
            # Generate gain offset errors
            self._i_ch_gain_err = self._genI_generator.uniform(-0.02, +0.02, 1)[0]
            self._q_ch_gain_err = self._genQ_generator.uniform(-0.02, +0.02, 1)[0]

        a = (0.1-0.06)/(2**OPENTETRAPHYMAC_HW_DAC_BIT_NUMBER - 1) * 1
        b = (0.1+0.06)/(2**OPENTETRAPHYMAC_HW_DAC_BIT_NUMBER - 1) * 1

        # Add INL-DNL error and channel offset error
        i_float_data += self._genI_generator.uniform(a, b, len(i_float_data))
        i_float_data += self._i_ch_offset_err

        q_float_data += self._genI_generator.uniform(a, b, len(q_float_data))
        q_float_data += self._q_ch_offset_err

        # # Add gain error (from DAC and resistor mismatch)
        i_float_data += self._i_ch_gain_err*i_float_data
        q_float_data += self._q_ch_gain_err*q_float_data

        # Add crosstalk between channels
        q_copy = q_float_data.copy() * (10 ** (-95/10))
        q_float_data += (i_float_data.copy()) * (10 ** (-95/10))
        i_float_data += q_copy

        # Apply offset and gain corrections done in mixer and/or DAC
        i_float_data += self.i_ch_ofst_correction
        i_float_data *= self.i_ch_gain_correction
        q_float_data += self.q_ch_ofst_correction
        q_float_data *= self.q_ch_gain_correction

        return i_float_data, q_float_data

    def _analog_reconstruction(self, i_ch: NDArray[float64],
                               q_ch: NDArray[float64]
                               ) -> NDArray[complex128]:
        """
        Takes in DAC codes at rate Rs, converts to real floats with ZOH with
        sampling rate Rif which is x8 more than Rs, then filters with analog reconstruction filter.

        Then converts I and Q channels to real data modeling phase error in quadrature modulation
        """
        # 1. ZOH at TRANSMIT_SIMULATION_SAMPLING_FACTOR to the simulation sample rate
        i_float_data = repeat(i_ch, TRANSMIT_SIMULATION_SAMPLING_FACTOR)
        q_float_data = repeat(q_ch, TRANSMIT_SIMULATION_SAMPLING_FACTOR)

        # 2. Perform analog reconstruction filtering
        # First Generate bessel function coefficents
        sos = bessel(9, 100E3, btype='lowpass', analog=False,
                     fs=float(TRANSMIT_SIMULATION_SAMPLE_RATE), output="sos")
        # Apply filtering to each channel individually
        analog_i = sosfilt(sos, i_float_data)
        analog_q = sosfilt(sos, q_float_data)

        # TODO: Add phase noise simulation

        # 3. Apply phase error
        analog_i = analog_i - analog_q*sin(self._q_ch_phase_err)
        analog_q = analog_q*cos(self._q_ch_phase_err)

        rf_signal = zeros_like(analog_i, dtype=complex128)
        rf_signal = (analog_i + 1j*analog_q).astype(complex128)

        return rf_signal

###################################################################################################


class IdealTransmitter(RFTransmitter):
    """
    Implementation of the parent RFTransmitter base class which uses non-quantized float data in its'
    baseband processing method and adds very little non-linearities and errors to output transmission
    data asides from AWGN noise. However, still uses the same order of filters as its' RealTransmitter alternative
    """

    rrc_filter_state = zeros(shape=(2, len(TX_RRC_FLOAT_COEFFICIENTS)-1), dtype=float64)
    lpf_filter_state = zeros(shape=(2, len(TX_LPF_FLOAT_COEFFICIENTS)-1), dtype=float64)
    halfband_1_filter_state = zeros(shape=(2, len(TX_HALFBAND1_FLOAT_COEFFICIENTS)-1), dtype=float64)
    halfband_2_filter_state = zeros(shape=(2, len(TX_HALFBAND2_FLOAT_COEFFICIENTS)-1), dtype=float64)
    halfband_3_filter_state = zeros(shape=(2, len(TX_HALFBAND3_FLOAT_COEFFICIENTS)-1), dtype=float64)

    def _baseband_processing(self, symbol_complex_data: NDArray[complex64],
                             burst_ramp_periods: tuple[int, int]) -> tuple[NDArray[float64], NDArray[float64]]:
        """
        Converts modulation bits into symbols, performs upsampling, ramping, and filtering using ideal float data
        """
        # 1. Determine if prepending and/or postpending zeros to flush is required

        # 1a. continuous with previous burst consideration:
        # 1. Determine if prepending and/or postpending zeros to flush is required
        start_offset = 0
        end_offset = 0
        # 1a. continuous with previous burst consideration:
        if burst_ramp_periods[0] != 0:
            # Since we ramp up, we are not continuous with previous data, and must flush the FIRs with prepended zeros
            insulated_input_data = concatenate((full(shape=BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT,
                                               fill_value=complex64(1 + 0j), dtype=complex64),
                                               symbol_complex_data))
            start_offset = BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT

        else:
            insulated_input_data = symbol_complex_data.copy()

        # 1b. continuous with subsequent burst consideration:
        if burst_ramp_periods[1] != 0:
            # Since we ramp down at the end, we are not continuous afterwards and
            # should flush data with postpended zeros
            insulated_input_data = concatenate((insulated_input_data, full(shape=BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT,
                                               fill_value=complex64(1 + 0j), dtype=complex64)))
            end_offset = BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT

        mag = sqrt(insulated_input_data.real[(start_offset):
                                             len(insulated_input_data.real)-(end_offset)].astype(float64)**2
                   + insulated_input_data.imag[(start_offset):
                                               len(insulated_input_data.imag)-(end_offset)].astype(float64)**2)

        print("\nPre x8 upsampling - symbol mapper ", "peakFS: ", mag.max()/(1), "rmsFS: ", sqrt(mean(mag**2))/(1))

        # 2. Upsample by x8 with zero insertions, and quantize data to Q1.17 fixed format stored in float64
        stage_1_symbols = oversample_data_float(insulated_input_data, 8)

        # 3. Perform RRC filtering
        stage_2_symbols = zeros_like(stage_1_symbols)
        stage_2_symbols[0], stage_2_symbols[1], self.rrc_filter_state[0], self.rrc_filter_state[1] = \
            dsp_fir_i_q_stream_convolve(stage_1_symbols[0], stage_1_symbols[1], TX_RRC_FLOAT_COEFFICIENTS,
                                        self.rrc_filter_state, 4)

        # 4. Perform cleanup LPF'ing
        stage_3_symbols = zeros_like(stage_2_symbols)
        stage_3_symbols[0], stage_3_symbols[1], self.lpf_filter_state[0], self.lpf_filter_state[1] = \
            dsp_fir_i_q_stream_convolve(stage_2_symbols[0], stage_2_symbols[1], TX_LPF_FLOAT_COEFFICIENTS,
                                        self.lpf_filter_state, 1)

        # 5. Perform x2 upsampling with zero insertions and filter - Part 1
        stage_4_symbols = zeros(shape=(2, 2*stage_3_symbols.shape[1]), dtype=float64)
        stage_4_symbols[:, ::2] = stage_3_symbols

        stage_5_symbols = zeros_like(stage_4_symbols)
        stage_5_symbols[0], stage_5_symbols[1], self.halfband_1_filter_state[0], self.halfband_1_filter_state[1] = \
            dsp_fir_i_q_stream_convolve(stage_4_symbols[0], stage_4_symbols[1], TX_HALFBAND1_FLOAT_COEFFICIENTS,
                                        self.halfband_1_filter_state, 2)

        # 6. Perform x2 upsampling with zero insertions and filter - Part 2
        stage_6_symbols = zeros(shape=(2, 2*stage_5_symbols.shape[1]), dtype=float64)
        stage_6_symbols[:, ::2] = stage_5_symbols

        stage_7_symbols = zeros_like(stage_6_symbols)
        stage_7_symbols[0], stage_7_symbols[1], self.halfband_2_filter_state[0], self.halfband_2_filter_state[1] = \
            dsp_fir_i_q_stream_convolve(stage_6_symbols[0], stage_6_symbols[1], TX_HALFBAND2_FLOAT_COEFFICIENTS,
                                        self.halfband_2_filter_state, 2)

        # 7. Perform x2 upsampling with zero insertions and filter - Part 3
        stage_8_symbols = zeros(shape=(2, 2*stage_7_symbols.shape[1]), dtype=float64)
        stage_8_symbols[:, ::2] = stage_7_symbols

        stage_9_symbols = zeros_like(stage_8_symbols)
        stage_9_symbols[0], stage_9_symbols[1], self.halfband_3_filter_state[0], self.halfband_3_filter_state[1] = \
            dsp_fir_i_q_stream_convolve(stage_8_symbols[0], stage_8_symbols[1], TX_HALFBAND3_FLOAT_COEFFICIENTS,
                                        self.halfband_3_filter_state, 2)

        # 8. Extract useful part of burst
        full_pad_length = int(BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT*TX_BB_SAMPLING_FACTOR)
        if burst_ramp_periods[0] != 0:
            stage_9_symbols = stage_9_symbols[:, full_pad_length:].copy()
        if burst_ramp_periods[1] != 0:
            stage_9_symbols = stage_9_symbols[:, :-full_pad_length].copy()

        # 9. Perform ramping on signal
        i_ramped_symbols, q_ramped_symbols = power_ramping_float(stage_9_symbols[0], stage_9_symbols[1],
                                                                 burst_ramp_periods)
        mag = sqrt(i_ramped_symbols.astype(float64)**2 + q_ramped_symbols.astype(float64)**2)
        print("Post ramping ", "peakFS: ", mag.max(), "rmsFS: ", sqrt(mean(mag**2)))

        return i_ramped_symbols, q_ramped_symbols

    def _dac_conversion(self, i_ch: NDArray[int64 | float64], q_ch: NDArray[int64 | float64]
                        ) -> tuple[NDArray[float64], NDArray[float64]]:
        """
        Converts baseband processed data into dac code representation, does not model DAC non linearities however.
        """
        # 1. We have no errors in te ideal form
        i_float_data = i_ch.copy().astype(float64)
        q_float_data = q_ch.copy().astype(float64)

        return i_float_data, q_float_data

    def _analog_reconstruction(self, i_ch: NDArray[float64],
                               q_ch: NDArray[float64]
                               ) -> NDArray[complex128]:
        """
        Takes in DAC codes at rate Rs, converts to real floats with ZOH with
        sampling rate Rif which is x8 more than Rs, then filters with analog reconstruction filter.

        Does not model gain or offset errors, coupling, phase noise or LO leakage
        """
        # 1. ZOH at TRANSMIT_SIMULATION_SAMPLING_FACTOR to the simulation sample rate
        i_float_data = repeat(i_ch, TRANSMIT_SIMULATION_SAMPLING_FACTOR)
        q_float_data = repeat(q_ch, TRANSMIT_SIMULATION_SAMPLING_FACTOR)

        # 2. Perform analog reconstruction filtering
        # First Generate bessel function coefficents
        sos = bessel(9, 100E3, btype='lowpass', analog=False,
                     fs=float(TRANSMIT_SIMULATION_SAMPLE_RATE), output="sos")
        # Apply filtering to each channel individually
        analog_i = sosfilt(sos, i_float_data)
        analog_q = sosfilt(sos, q_float_data)

        # TODO: Add phase noise simulation

        # 3. Convert to complex signal

        rf_signal = zeros_like(analog_i, dtype=complex128)
        rf_signal[:] = (analog_i + 1j*analog_q).astype(complex128)

        return rf_signal
