from abc import ABC, abstractmethod
from typing import ClassVar, Literal
from numpy import complex64, float64, uint8, array, zeros, float32, mean, \
    int64, concatenate, full, sqrt
from numpy.typing import NDArray

from .tx_rx_utilities import dsp_fir_float_stream, dsp_fir_quantized_stream, power_ramping_float,\
    power_ramping_quantized, assert_tail_is_zero, oversample_data_quantized, oversample_data_float
from .constants import SUBSLOT_BIT_LENGTH, BASEBAND_SAMPLING_FACTOR
from .modulation import dqpsk_modulator

VALID_RETURN_STAGE_VALUES = ("baseband", "dac", "tx") # possible output stages from transmitter.transmitBurst(),

HALF_BASEBAND_SAMPLING_FACTOR = int(BASEBAND_SAMPLING_FACTOR / 2)   # Half of the culmative baseband sampling factor
TETRA_SYMBOL_RATE = 18000                                           # The base EN 300 392-2 symbol rate

TRANSMIT_SIMULATION_SAMPLING_FACTOR = 10    # The internal sw simulator sampling factor over the DAC rate
TRANSMIT_SIMULATION_SAMPLE_RATE = int(TETRA_SYMBOL_RATE * TETRA_SYMBOL_RATE * BASEBAND_SAMPLING_FACTOR)
                                            # Internal ^ sw simulator sampling rate, allows for capture of harmonics

RRC_Q1_17_COEFFICIENTS = array(
    [94, 42, 69, 77, 61, 26, -21, -66, -96, -100, -73, -21, 45, 108, 150, 157, 123, 52, -39, -128,
    -190, -206, -168, -81, 33, 144, 218, 226, 156, 19, -156, -319, -416, -399, -242, 47, 419, 794,
    1068, 1141, 934, 420, -361, -1297, -2210, -2881, -3084, -2625, -1385, 653, 3382, 6582, 9936,
    13080, 15648, 17328, 17912, 17328, 15648, 13080, 9936, 6582, 3382, 653, -1385, -2625, -3084,
    -2881, -2210, -1297, -361, 420, 934, 1141, 1068, 794, 419, 47, -242, -399, -416, -319, -156,
    19, 156, 226, 218, 144, 33, -81, -168, -206, -190, -128, -39, 52, 123, 157, 150, 108, 45,
    -21, -73, -100, -96, -66, -21, 26, 61, 77, 69, 42, 4], dtype=int64)

RRC_FLOAT_COEFFICIENTS = array(
    [3.2547283E-05, 3.2207786E-04, 5.2808010E-04, 5.8573583E-04, 4.6815889E-04, 1.9697665E-04,
    -1.5991251E-04, -5.0379767E-04, -7.3058502E-04, -7.5993093E-04, -5.6027190E-04, -1.6242533E-04,
    3.4303279E-04, 8.2534668E-04, 1.1462410E-03, 1.1978352E-03, 9.3581964E-04, 3.9825984E-04,
    -2.9658416E-04, -9.7493920E-04, -1.4502126E-03, -1.5738290E-03, -1.2810810E-03, -6.1880576E-04,
    2.5482883E-04, 1.1016943E-03, 1.6614646E-03, 1.7233356E-03, 1.1939828E-03, 1.4279077E-04,
    -1.1910767E-03, -2.4372363E-03, -3.1748905E-03, -3.0411342E-03, -1.8429373E-03, 3.5534150E-04,
    3.1950125E-03, 6.0541783E-03, 8.1505440E-03, 8.7031256E-03, 7.1246121E-03, 3.2053748E-03,
    -2.7531665E-03, -9.8923352E-03, -1.6859306E-02, -2.1981161E-02, -2.3528602E-02, -2.0027600E-02,
    -1.0563596E-02, 4.9805501E-03, 2.5803484E-02, 5.0213095E-02, 7.5808890E-02, 9.9796265E-02,
    1.1938435E-01, 1.3220197E-01, 1.3666072E-01, 1.3220197E-01, 1.1938435E-01, 9.9796265E-02,
    7.5808890E-02, 5.0213095E-02, 2.5803484E-02, 4.9805501E-03, -1.0563596E-02, -2.0027600E-02,
    -2.3528602E-02, -2.1981161E-02, -1.6859306E-02, -9.8923352E-03, -2.7531665E-03, 3.2053748E-03,
    7.1246121E-03, 8.7031256E-03, 8.1505440E-03, 6.0541783E-03, 3.1950125E-03, 3.5534150E-04,
    -1.8429373E-03, -3.0411342E-03, -3.1748905E-03, -2.4372363E-03, -1.1910767E-03, 1.4279077E-04,
    1.1939828E-03, 1.7233356E-03, 1.6614646E-03, 1.1016943E-03, 2.5482883E-04, -6.1880576E-04,
    -1.2810810E-03, -1.5738290E-03, -1.4502126E-03, -9.7493920E-04, -2.9658416E-04, 3.9825984E-04,
    9.3581964E-04, 1.1978352E-03, 1.1462410E-03, 8.2534668E-04, 3.4303279E-04, -1.6242533E-04,
    -5.6027190E-04, -7.5993093E-04, -7.3058502E-04, -5.0379767E-04, -1.5991251E-04, 1.9697665E-04,
    4.6815889E-04, 5.8573583E-04, 5.2808010E-04, 3.2207786E-04, 3.2547283E-05], dtype=float32)

FIR_LPF_Q1_17_COEFFICIENTS = array(
    [40, 136, 205, 218, 156, 27, -136, -283, -361, -329, -179, 57, 312, 503, 555, 429, 140,
    -240, -595, -804, -776, -485, 9, 570, 1022, 1199, 1004, 447, -339, -1128, -1659, -1720,
    -1216, -226, 1003, 2104, 2694, 2483, 1388, -411, -2476, -4191, -4904, -4088, -1501, 2723,
    8064, 13701, 18671, 22078, 23290, 22078, 18671, 13701, 8064, 2723, -1501, -4088, -4904,
    -4191, -2476, -411, 1388, 2483, 2694, 2104, 1003, -226, -1216, -1720, -1659, -1128, -339,
    447, 1004, 1199, 1022, 570, 9, -485, -776, -804, -595, -240, 140, 429, 555, 503, 312, 57,
    -179, -329, -361, -283, -136, 27, 156, 218, 205, 136, 40], dtype=int64)

FIR_LPF_FLOAT_COEFFICIENTS = array(
    [3.0866607E-04, 1.0378698E-03, 1.5670853E-03, 1.6598026E-03, 1.1896548E-03, 2.0867007E-04,
    -1.0375803E-03, -2.1627121E-03, -2.7524120E-03, -2.5066389E-03, -1.3648995E-03, 4.3268623E-04,
    2.3801184E-03, 3.8384833E-03, 4.2354544E-03, 3.2718693E-03, 1.0650634E-03, -1.8293252E-03,
    4.5416570E-03, -6.1367146E-03, -5.9191459E-03, -3.7006680E-03, 6.5840668E-05, 4.3479592E-03,
    7.7964842E-03, 9.1487500E-03, 7.6578711E-03, 3.4103186E-03, -2.5865371E-03, -8.6053726E-03,
    -1.2660975E-02, -1.3118963E-02, -9.2759097E-03, -1.7231343E-03, 7.6487811E-03, 1.6053667E-02,
    2.0551295E-02, 1.8945906E-02, 1.0593299E-02, -3.1324285E-03, -1.8888838E-02, -3.1977766E-02,
    -3.7411781E-02, -3.1189190E-02, -1.1454434E-02, 2.0773036E-02, 6.1524669E-02, 1.0452736E-01,
    1.4244613E-01, 1.6844460E-01, 1.7769138E-01, 1.6844460E-01, 1.4244613E-01, 1.0452736E-01,
    6.1524669E-02, 2.0773036E-02, -1.1454434E-02, -3.1189190E-02, -3.7411781E-02, -3.1977766E-02,
    -1.8888838E-02, -3.1324285E-03, 1.0593299E-02, 1.8945906E-02, 2.0551295E-02, 1.6053667E-02,
    7.6487811E-03, -1.7231343E-03, -9.2759097E-03, -1.3118963E-02, -1.2660975E-02, -8.6053726E-03,
    -2.5865371E-03, 3.4103186E-03, 7.6578711E-03, 9.1487500E-03, 7.7964842E-03, 4.3479592E-03,
    6.5840668E-05, -3.7006680E-03, -5.9191459E-03, -6.1367146E-03, -4.5416570E-03, -1.8293252E-03,
    1.0650634E-03, 3.2718693E-03, 4.2354544E-03, 3.8384833E-03, 2.3801184E-03, 4.3268623E-04,
    -1.3648995E-03, -2.5066389E-03, -2.7524120E-03, -2.1627121E-03, -1.0375803E-03, 2.0867007E-04,
    1.1896548E-03, 1.6598026E-03, 1.5670853E-03, 1.0378698E-03, 3.0866607E-04], dtype=float32)

FIR_HALFBAND1_Q1_17_COEFFICIENTS = array(
    [-145, 0, 193, 0, -323, 0, 555, 0, -914, 0, 1434, 0, -2170, 0, 3221, 0, -4805, 0,
    7491, 0, -13392, 0, 41587, 65606, 41587, 0, -13392, 0, 7491, 0, -4805, 0, 3221, 0,
    -2170, 0, 1434, 0, -914, 0, 555, 0, -323, 0, 193, 0, -145], dtype=int64)

FIR_HALFBAND1_FLOAT_COEFFICIENTS = array(
    [-1.1083445E-03, 0.0000000E+00, 1.4727359E-03, 0.0000000E+00, -2.4647851E-03, 0.0000000E+00,
    4.2366369E-03, 0.0000000E+00, -6.9756542E-03, 0.0000000E+00, 1.0942169E-02, 0.0000000E+00,
    -1.6552123E-02, 0.0000000E+00, 2.4572962E-02, 0.0000000E+00, -3.6657065E-02, 0.0000000E+00,
    5.7154625E-02, 0.0000000E+00, -1.0217133E-01, 0.0000000E+00, 3.1728380E-01, 5.0053275E-01,
    3.1728380E-01, 0.0000000E+00, -1.0217133E-01, 0.0000000E+00, 5.7154625E-02, 0.0000000E+00,
    -3.6657065E-02, 0.0000000E+00, 2.4572962E-02, 0.0000000E+00, -1.6552123E-02, 0.0000000E+00,
    1.0942169E-02, 0.0000000E+00, -6.9756542E-03, 0.0000000E+00, 4.2366369E-03, 0.0000000E+00,
    -2.4647851E-03, 0.0000000E+00, 1.4727359E-03, 0.0000000E+00, -1.1083445E-03], dtype=float32)

FIR_HALFBAND2_Q1_17_COEFFICIENTS = array(
    [-304, 0, 711, 0, -2084, 0, 5063, 0, -11725, 0, 41035, 65681, 41035, 0, -11725, 0,
    5063, 0, -2084, 0, 711, 0, -304], dtype=int64)

FIR_HALFBAND2_FLOAT_COEFFICIENTS = array(
    [-2.3200984E-03, 0.0000000E+00, 5.4240586E-03, 0.0000000E+00, -1.5900960E-02, 0.0000000E+00,
    3.8630295E-02, 0.0000000E+00, -8.9455216E-02, 0.0000000E+00, 3.1306928E-01, 5.0110528E-01,
    3.1306928E-01, 0.0000000E+00, -8.9455216E-02, 0.0000000E+00, 3.8630295E-02, 0.0000000E+00,
    -1.5900960E-02, 0.0000000E+00, 5.4240586E-03, 0.0000000E+00, -2.3200984E-03], dtype=float32)

FIR_HALFBAND3_Q1_17_COEFFICIENTS = array(
    [0, 1181, 0, -7504, 0, 39118, 65482, 39118, 0, -7504, 0, 1181, 0], dtype=int64)

FIR_HALFBAND3_FLOAT_COEFFICIENTS = array(
    [0.0000000E+00, 9.0088874E-03, 0.0000000E+00, -5.7248430E-02, 0.0000000E+00, 2.9844614E-01,
    4.9958680E-01, 2.9844614E-01, 0.0000000E+00, -5.7248430E-02, 0.0000000E+00, 9.0088874E-03,
    0.0000000E+00], dtype=float32)

BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT = 30


###################################################################################################
class Transmitter(ABC):

    phase_reference = complex64(1 + 0j)

    rrc_filter_state: ClassVar[NDArray]
    lpf_filter_state: ClassVar[NDArray]
    halfband_1_filter_state: ClassVar[NDArray]
    halfband_2_filter_state: ClassVar[NDArray]
    halfband_3_filter_state: ClassVar[NDArray]

    def __init__(self):
        pass

    @abstractmethod
    def _baseband_processing(self, symbol_complex_data:NDArray[complex64],
                            burst_ramp_periods:tuple[int, int]
                            ) -> tuple[NDArray[int64 | float32], NDArray[int64 | float32]]:
        """
        Converts modulation bits into symbols, performs upsampling, ramping, and filtering
        """
        raise NotImplementedError

    @abstractmethod
    def _dac_conversion(self):
        """
        Converts baseband processed data into dac code representation, 
        """
        raise NotImplementedError

    @abstractmethod
    def _analog_reconstruction(self):
        """
        Takes in DAC codes at rate Rs, converts to real floats with ZOH with 
        sampling rate Rif which is x9 more than Rs, then filters with analog reconstruction filter.
        """
        # coupling
        # gain error
        # offset
        raise NotImplementedError

    def transmit_burst(self, burst_bit_seq:NDArray[uint8], burst_ramp_periods:tuple[int, int],
                      subslot_2_burst_ramp_periods:tuple[int, int] | None = None,
                      debug_return_stage: Literal["baseband"] | Literal["dac"] | Literal["tx"] ="baseband"
                      ) -> tuple[NDArray, NDArray]:

        if debug_return_stage not in VALID_RETURN_STAGE_VALUES:
            raise RuntimeError(f"Passed debug return stage for transmitBurst of: {debug_return_stage} invalid,"
                               f" expected value in: {VALID_RETURN_STAGE_VALUES}")

        #1. Determine handling of input burstbitSequence

        number_of_used_subslots = 0
        halfslot = False
        null_subslot_index = None

        if burst_bit_seq.ndim == 2:
            # multiple slot bursts or subslot bursts passed
            if burst_bit_seq.shape[1] == SUBSLOT_BIT_LENGTH:
                # Passed 2 sublots for a single burst, acceptable
                # If the blocks are only SUBLOT length, then we are transmitting a single full slot burst
                # Could be of form: [CB, empty], [LB, CB], [empty, CB], or [LB, empty]
                if burst_bit_seq.shape[0] != 2:
                    raise ValueError(f"Passed {burst_bit_seq.shape[0]} subslots to transmit,"
                                     f" expected exactly 2 to handle subslot tx")
                else:
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

                    halfslot = True
            else:
                raise ValueError(f"Passed {burst_bit_seq.shape[1]} modulation bits,"
                                 f" expected 2 subslots of length {(SUBSLOT_BIT_LENGTH)}")
        elif burst_bit_seq.ndim == 1 and burst_bit_seq.size == (2*SUBSLOT_BIT_LENGTH):
            # Passed only one full slot burst, acceptable
            number_of_used_subslots = 1
        else:
            raise ValueError(f"Passed burstbitSequences of shape: {burst_bit_seq.shape},"
                             f" invalid number of dimensions or invalid number of modulation bits")

        # Allocate an output array for the burst data
        # outputBBSignal = empty(shape=(1, (TIMESLOT_SYMBOL_LENGTH * BASEBAND_SAMPLING_FACTOR * TETRA_SYMBOL_RATE)),
        #                        dtype=complex64)

        #2. Check if half slot
        if halfslot:
            # Single burst made from 1 to 2 subslots bursts (may have null/empty subslot) with odd number of bits.
            # Therefore we postpend an additional modulation bit to the end of each subslot
            # Because the last symbol is entirely zero, and we simply truncate the burst at the end
            # to only be zero for half a symbol at the ramp-downs of each subslot in order to lineup the timing overall

            #3. Determine the state of the phase reference usage, here it will always be the default case
            burst_phase_ref = complex64(1 + 0j)
            if number_of_used_subslots == 2 and subslot_2_burst_ramp_periods is not None:
                # we have two subslot bursts to modulate indepedently.
                # must increment the end guard period of the first and second subslot burst to
                # account for the need for the additional modulation bit

                #4. Modulate the burst bits into 255 symbols
                ssb1_burst_ramp_periods = (burst_ramp_periods[0], burst_ramp_periods[1]+1)
                ssb2_burst_ramp_periods = (subslot_2_burst_ramp_periods[0], subslot_2_burst_ramp_periods[1]+1)
                sbs1_symbol_complex_data = dqpsk_modulator(concatenate((burst_bit_seq[0], zeros(1, dtype=uint8))),
                                                     ssb1_burst_ramp_periods,
                                                     burst_phase_ref)
                sbs2_symbol_complex_data = dqpsk_modulator(concatenate((burst_bit_seq[1], zeros(1, dtype=uint8))),
                                                     ssb2_burst_ramp_periods,
                                                     burst_phase_ref)

                #5. Pass modulated symbols and guard information to baseband processing function
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
                                                     ssb2_burst_ramp_periods,
                                                     burst_phase_ref)
                i_ch_sbs_2, q_ch_sbs_2 = self._baseband_processing(sbs2_symbol_complex_data, ssb2_burst_ramp_periods)

                # generate equivalent number of zeros to fill null sublot
                i_ch_sbs_1 = zeros((SUBSLOT_BIT_LENGTH+1)*64, dtype=(int64 if isinstance(self, RealTransmitter)
                                                                      else float32))
                q_ch_sbs_1 = zeros((SUBSLOT_BIT_LENGTH+1)*64, dtype=(int64 if isinstance(self, RealTransmitter)
                                                                      else float32))

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
                                                     ssb1_burst_ramp_periods,
                                                     burst_phase_ref)
                i_ch_sbs_1, q_ch_sbs_1 = self._baseband_processing(sbs1_symbol_complex_data, ssb1_burst_ramp_periods)

                # generate equivalent number of zeros to fill null sublot
                i_ch_sbs_2 = zeros((SUBSLOT_BIT_LENGTH+1)*64, dtype=(int64 if isinstance(self, RealTransmitter)
                                                                      else float32))
                q_ch_sbs_2 = zeros((SUBSLOT_BIT_LENGTH+1)*64, dtype=(int64 if isinstance(self, RealTransmitter)
                                                                      else float32))

                # verify that the tail does go to zero otherwise we get a discontinuity
                assert_tail_is_zero(i_ch_sbs_1, q_ch_sbs_1, HALF_BASEBAND_SAMPLING_FACTOR)

                i_temp_ramp = concatenate((i_ch_sbs_1[:i_ch_sbs_1.size - HALF_BASEBAND_SAMPLING_FACTOR],
                                         i_ch_sbs_2[:i_ch_sbs_2.size - HALF_BASEBAND_SAMPLING_FACTOR]))
                q_temp_ramp = concatenate((q_ch_sbs_1[:q_ch_sbs_1.size - HALF_BASEBAND_SAMPLING_FACTOR],
                                         q_ch_sbs_2[:q_ch_sbs_2.size - HALF_BASEBAND_SAMPLING_FACTOR]))

            else:
                raise ValueError("subslot2RampPeriod cannot be None, using the second subslot it is expected"
                                 +" to represent the bit interval if odd")

        else:
            # Full slot burst
            # if we are not ramping up at the start of the burst, then we can assume it is continuous with
            # the previous burst and use the the internal phase reference state
            burst_ramp_state = (burst_ramp_periods[0] != 0, burst_ramp_periods[1] != 0)
            burst_phase_ref = complex64(1 + 0j) if burst_ramp_state[0] else self.phase_reference

            #4. Modulate the burst bits into 255 symbols
            input_symbol_complex_data = dqpsk_modulator(burst_bit_seq, burst_ramp_periods, burst_phase_ref)
            # if we ramp down at the end, reset the phase reference for the next burst, if we don't ramp down then
            # it is assumed phase continuous and we set the reference state to the last phase of the burst
            self.phase_reference = input_symbol_complex_data[-1] if not burst_ramp_state[1] else complex64(1 + 0j)

            #5. Pass modulated symbols and guard information to baseband processing function
            i_temp_ramp, q_temp_ramp = self._baseband_processing(input_symbol_complex_data, burst_ramp_periods)

        if debug_return_stage == VALID_RETURN_STAGE_VALUES[0]:
            return i_temp_ramp, q_temp_ramp


        #6. Convert to DAC representation and perform ZOH at higher sampling rate
        return i_temp_ramp, q_temp_ramp


###################################################################################################

class RealTransmitter(Transmitter):

    rrc_filter_state = zeros(shape=(2,len(RRC_Q1_17_COEFFICIENTS)-1), dtype=int64)
    lpf_filter_state = zeros(shape=(2,len(FIR_LPF_Q1_17_COEFFICIENTS)-1), dtype=int64)
    halfband_1_filter_state = zeros(shape=(2,len(FIR_HALFBAND1_Q1_17_COEFFICIENTS)-1), dtype=int64)
    halfband_2_filter_state = zeros(shape=(2,len(FIR_HALFBAND2_Q1_17_COEFFICIENTS)-1), dtype=int64)
    halfband_3_filter_state = zeros(shape=(2,len(FIR_HALFBAND3_Q1_17_COEFFICIENTS)-1), dtype=int64)

    def _baseband_processing(self, symbol_complex_data:NDArray[complex64],
                            burst_ramp_periods:tuple[int, int]) -> tuple[NDArray[int64], NDArray[int64]]:
        """
        Converts modulation bits into symbols, performs upsampling, ramping, and filtering using quantized data
        """
        #1. Determine if prepending and/or postpending zeros to flush is required
        start_offset = 0
        end_offset = 0
        #1a. continuous with previous burst consideration:
        if burst_ramp_periods[0] != 0:
            # Since we ramp up, we are not continuous with previous data, and must flush the FIRs with prepended zeros
            insulated_input_data = concatenate((full(shape=BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT,
                                                   fill_value=complex64(1 + 0j), dtype=complex64),
                                                   symbol_complex_data))
            start_offset = BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT
        else:
            insulated_input_data = symbol_complex_data.copy()

        #1b. continuous with subsequent burst consideration:
        if burst_ramp_periods[1] != 0:
            # Since we ramp down at the end, we are not continuous afterwards and
            # should flush data with postpended zeros
            insulated_input_data = concatenate((insulated_input_data, full(shape=BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT,
                                                                       fill_value=complex64(1 + 0j), dtype=complex64)))
            end_offset = BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT

        i_temp = insulated_input_data.real.astype(int64, copy=True)
        q_temp = insulated_input_data.imag.astype(int64, copy=True)

        mag = sqrt(i_temp[(start_offset):len(i_temp)-(end_offset)].astype(float64)**2
                   + q_temp[(start_offset):len(q_temp)-(end_offset)].astype(float64)**2)
        print("Pre x8 upsampling - symbol mapper ", "peakFS: ", mag.max()/(1), "rmsFS: ", sqrt(mean(mag**2))/(1))

        #2. Upsample by x8 with zero insertions, and quantize data to Q1.17 fixed format stored in float32
        upsampled_input_data = oversample_data_quantized(insulated_input_data, 8)


        #3. Perform RRC filtering
        i_stage_1_symbols = upsampled_input_data[0].copy()
        q_stage_1_symbols = upsampled_input_data[1].copy()

        i_stage_2_symbols, self.rrc_filter_state[0] = dsp_fir_quantized_stream(i_stage_1_symbols,
                                                                                RRC_Q1_17_COEFFICIENTS,
                                                                                self.rrc_filter_state[0], gain=4)
        q_stage_2_symbols, self.rrc_filter_state[1] = dsp_fir_quantized_stream(q_stage_1_symbols,
                                                                                RRC_Q1_17_COEFFICIENTS,
                                                                                self.rrc_filter_state[1], gain=4)

        mag = sqrt(i_stage_2_symbols[(start_offset*8):len(i_stage_2_symbols)-(end_offset*8)].astype(float64)**2
                   + q_stage_2_symbols[(start_offset*8):len(q_stage_2_symbols)-(end_offset*8)].astype(float64)**2)
        print("Post RRC ", "peakFS: ", mag.max()/(1<<17), "rmsFS: ", sqrt(mean(mag**2))/(1<<17))

        #4. Perform cleanup LPF'ing
        i_stage_3_symbols, self.lpf_filter_state[0] = dsp_fir_quantized_stream(i_stage_2_symbols,
                                                                              FIR_LPF_Q1_17_COEFFICIENTS,
                                                                              self.lpf_filter_state[0])
        q_stage_3_symbols, self.lpf_filter_state[1] = dsp_fir_quantized_stream(q_stage_2_symbols,
                                                                              FIR_LPF_Q1_17_COEFFICIENTS,
                                                                              self.lpf_filter_state[1])

        mag = sqrt(i_stage_3_symbols[(start_offset*8):len(i_stage_3_symbols)-(end_offset*8)].astype(float64)**2
                   + q_stage_3_symbols[(start_offset*8):len(q_stage_3_symbols)-(end_offset*8)].astype(float64)**2)
        print("Post Cleanup ", "peakFS: ", mag.max()/(1<<17), "rmsFS: ", sqrt(mean(mag**2))/(1<<17))

        #5. Perform x2 upsampling with zero insertions and filter - Part 1
        i_stage_4_symbols = zeros(shape=(2*i_stage_3_symbols.size), dtype=int64)
        i_stage_4_symbols[::2] = i_stage_3_symbols
        i_stage_5_symbols, self.halfband_1_filter_state[0] = dsp_fir_quantized_stream(i_stage_4_symbols,
                                                                              FIR_HALFBAND1_Q1_17_COEFFICIENTS,
                                                                              self.halfband_1_filter_state[0], gain=2)

        q_stage_4_symbols = zeros(shape=(2*q_stage_3_symbols.size), dtype=int64)
        q_stage_4_symbols[::2] = q_stage_3_symbols
        q_stage_5_symbols, self.halfband_1_filter_state[1] = dsp_fir_quantized_stream(q_stage_4_symbols,
                                                                              FIR_HALFBAND1_Q1_17_COEFFICIENTS,
                                                                              self.halfband_1_filter_state[1], gain=2)

        mag = sqrt(i_stage_5_symbols[(start_offset*16):len(i_stage_5_symbols)-(end_offset*16)].astype(float64)**2
                   + q_stage_5_symbols[(start_offset*16):len(q_stage_5_symbols)-(end_offset*16)].astype(float64)**2)
        print("Post halfband-1 ", "peakFS: ", mag.max()/(1<<17), "rmsFS: ", sqrt(mean(mag**2))/(1<<17))

        #6. Perform x2 upsampling with zero insertions and filter - Part 2
        i_stage_6_symbols = zeros(shape=(2*i_stage_5_symbols.size), dtype=int64)
        i_stage_6_symbols[::2] = i_stage_5_symbols
        i_stage_7_symbols, self.halfband_2_filter_state[0] = dsp_fir_quantized_stream(i_stage_6_symbols,
                                                                              FIR_HALFBAND2_Q1_17_COEFFICIENTS,
                                                                              self.halfband_2_filter_state[0], gain=2)

        q_stage_6_symbols = zeros(shape=(2*q_stage_5_symbols.size), dtype=int64)
        q_stage_6_symbols[::2] = q_stage_5_symbols
        q_stage_7_symbols, self.halfband_2_filter_state[1] = dsp_fir_quantized_stream(q_stage_6_symbols,
                                                                              FIR_HALFBAND2_Q1_17_COEFFICIENTS,
                                                                              self.halfband_2_filter_state[1], gain=2)

        mag = sqrt(i_stage_7_symbols[(start_offset*32):len(i_stage_7_symbols)-(end_offset*32)].astype(float64)**2
                   + q_stage_7_symbols[(start_offset*32):len(q_stage_7_symbols)-(end_offset*32)].astype(float64)**2)
        print("Post halfband-2 ", "peakFS: ", mag.max()/(1<<17), "rmsFS: ", sqrt(mean(mag**2))/(1<<17))

        #7. Perform x2 upsampling with zero insertions and filter - Part 3
        i_stage_8_symbols = zeros(shape=(2*i_stage_7_symbols.size), dtype=int64)
        i_stage_8_symbols[::2] = i_stage_7_symbols
        i_stage_9_symbols, self.halfband_3_filter_state[0] = dsp_fir_quantized_stream(i_stage_8_symbols,
                                                                              FIR_HALFBAND3_Q1_17_COEFFICIENTS,
                                                                              self.halfband_3_filter_state[0], gain=2)


        q_stage_8_symbols = zeros(shape=(2*q_stage_7_symbols.size), dtype=int64)
        q_stage_8_symbols[::2] = q_stage_7_symbols
        q_stage_9_symbols, self.halfband_3_filter_state[1] = dsp_fir_quantized_stream(q_stage_8_symbols,
                                                                              FIR_HALFBAND3_Q1_17_COEFFICIENTS,
                                                                              self.halfband_3_filter_state[1], gain=2)

        mag = sqrt(i_stage_9_symbols[(start_offset*64):len(i_stage_9_symbols)-(end_offset*64)].astype(float64)**2
                   + q_stage_9_symbols[(start_offset*64):len(q_stage_9_symbols)-(end_offset*64)].astype(float64)**2)
        print("Post halfband-3 ", "peakFS: ", mag.max()/(1<<17), "rmsFS: ", sqrt(mean(mag**2))/(1<<17))

        #8. Extract useful part of burst
        full_pad_length = int(BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT*BASEBAND_SAMPLING_FACTOR)
        if burst_ramp_periods[0] != 0:
            i_output_symbols = i_stage_9_symbols[full_pad_length:].copy()
            q_output_symbols = q_stage_9_symbols[full_pad_length:].copy()
        else:
            i_output_symbols = i_stage_9_symbols.copy()
            q_output_symbols = q_stage_9_symbols.copy()
        if burst_ramp_periods[1] != 0:
            i_output_symbols = i_output_symbols[:-full_pad_length].copy()
            q_output_symbols = q_output_symbols[:-full_pad_length].copy()
        #9. Perform ramping on signal
        i_ramped_symbols, q_ramped_symbols = power_ramping_quantized(i_output_symbols, q_output_symbols,
                                                                  burst_ramp_periods)
        mag = sqrt(i_ramped_symbols.astype(float64)**2 + q_ramped_symbols.astype(float64)**2)
        print("Post ramping ", "peakFS: ", mag.max()/(1<<17), "rmsFS: ", sqrt(mean(mag**2))/(1<<17))
        return i_ramped_symbols, q_ramped_symbols

    def _dac_conversion(self):
        """
        Converts baseband processed data into dac code representation, 
        """
        raise NotImplementedError

    def _analog_reconstruction(self):
        """
        Takes in DAC codes at rate Rs, converts to real floats with ZOH with 
        sampling rate Rif which is x8 more than Rs, then filters with analog reconstruction filter.
        """
        # coupling
        # gain error
        # offset
        raise NotImplementedError



###################################################################################################

class IdealTransmitter(Transmitter):

    rrc_filter_state = zeros(shape=(2,len(RRC_FLOAT_COEFFICIENTS)-1), dtype=float32)
    lpf_filter_state = zeros(shape=(2,len(FIR_LPF_FLOAT_COEFFICIENTS)-1), dtype=float32)
    halfband_1_filter_state = zeros(shape=(2,len(FIR_HALFBAND1_FLOAT_COEFFICIENTS)-1), dtype=float32)
    halfband_2_filter_state = zeros(shape=(2,len(FIR_HALFBAND2_FLOAT_COEFFICIENTS)-1), dtype=float32)
    halfband_3_filter_state = zeros(shape=(2,len(FIR_HALFBAND3_FLOAT_COEFFICIENTS)-1), dtype=float32)

    def _baseband_processing(self, symbol_complex_data:NDArray[complex64],
                            burst_ramp_periods:tuple[int, int]) -> tuple[NDArray[float32], NDArray[float32]]:
        """
        Converts modulation bits into symbols, performs upsampling, ramping, and filtering using float data
        """
        #1. Determine if prepending and/or postpending zeros to flush is required

        #1a. continuous with previous burst consideration:
        if burst_ramp_periods[0] != 0:
            # Since we ramp up, we are not continuous with previous data, and must flush the FIRs with prepended zeros
            insulated_input_data = concatenate((full(shape=BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT,
                                                   fill_value=complex64(1 + 0j), dtype=complex64), symbol_complex_data))
        else:
            insulated_input_data = symbol_complex_data.copy()
        #1b. continuous with subsequent burst consideration:
        if burst_ramp_periods[1] != 0:
            # Since we ramp down at the end, we are not continuous afterwards and
            # should flush data with postpended zeros
            insulated_input_data = concatenate((insulated_input_data, full(shape=BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT,
                                                                       fill_value=complex64(1 + 0j), dtype=complex64)))

        #2. Upsample by x8 with zero insertions, and quantize data to Q1.17 fixed format stored in float32
        upsampled_input_data = oversample_data_float(insulated_input_data, 8)

        #3. Perform RRC filtering
        i_stage_1_symbols = upsampled_input_data[0].copy()
        q_stage_1_symbols = upsampled_input_data[1].copy()

        i_stage_2_symbols, self.rrc_filter_state[0] = dsp_fir_float_stream(i_stage_1_symbols,
                                                                          RRC_FLOAT_COEFFICIENTS,
                                                                          self.rrc_filter_state[0], gain=4)
        q_stage_2_symbols, self.rrc_filter_state[1] = dsp_fir_float_stream(q_stage_1_symbols,
                                                                          RRC_FLOAT_COEFFICIENTS,
                                                                          self.rrc_filter_state[1], gain=4)

        #4. Perform cleanup LPF'ing
        i_stage_3_symbols, self.lpf_filter_state[0] = dsp_fir_float_stream(i_stage_2_symbols,
                                                                          FIR_LPF_FLOAT_COEFFICIENTS,
                                                                          self.lpf_filter_state[0])
        q_stage_3_symbols, self.lpf_filter_state[1] = dsp_fir_float_stream(q_stage_2_symbols,
                                                                          FIR_LPF_FLOAT_COEFFICIENTS,
                                                                          self.lpf_filter_state[1])

        #5. Perform x2 upsampling with zero insertions and filter - Part 1
        i_stage_4_symbols = zeros(shape=(2*i_stage_3_symbols.size), dtype=float32)
        i_stage_4_symbols[::2] = i_stage_3_symbols
        i_stage_5_symbols = zeros(shape=(2*i_stage_4_symbols.size), dtype=float32)
        i_stage_5_symbols[::2], self.halfband_1_filter_state[0] = dsp_fir_float_stream(i_stage_4_symbols,
                                                                               FIR_HALFBAND1_FLOAT_COEFFICIENTS,
                                                                               self.halfband_1_filter_state[0], gain=2)

        q_stage_4_symbols = zeros(shape=(2*q_stage_3_symbols.size), dtype=float32)
        q_stage_4_symbols[::2] = q_stage_3_symbols
        q_stage_5_symbols = zeros(shape=(2*q_stage_4_symbols.size), dtype=float32)
        q_stage_5_symbols[::2], self.halfband_1_filter_state[1] = dsp_fir_float_stream(q_stage_4_symbols,
                                                                               FIR_HALFBAND1_FLOAT_COEFFICIENTS,
                                                                               self.halfband_1_filter_state[1], gain=2)

        #6. Perform x2 upsampling with zero insertions and filter - Part 2
        i_stage_6_symbols = zeros(shape=(2*i_stage_5_symbols.size), dtype=float32)
        i_stage_6_symbols[::2], self.halfband_2_filter_state[0] = dsp_fir_float_stream(i_stage_5_symbols,
                                                                               FIR_HALFBAND2_FLOAT_COEFFICIENTS,
                                                                               self.halfband_2_filter_state[0], gain=2)

        q_stage_6_symbols = zeros(shape=(2*q_stage_5_symbols.size), dtype=float32)
        q_stage_6_symbols[::2], self.halfband_2_filter_state[1] = dsp_fir_float_stream(q_stage_5_symbols,
                                                                               FIR_HALFBAND2_FLOAT_COEFFICIENTS,
                                                                               self.halfband_2_filter_state[1], gain=2)
        #7. Perform x2 upsampling with zero insertions and filter - Part 3
        i_stage_7_symbols= zeros(shape=(i_stage_6_symbols.size), dtype=float32)
        i_stage_7_symbols, self.halfband_3_filter_state[0] = dsp_fir_float_stream(i_stage_6_symbols,
                                                                          FIR_HALFBAND3_FLOAT_COEFFICIENTS,
                                                                          self.halfband_3_filter_state[0], gain=2)

        q_stage_7_symbols = zeros(shape=(q_stage_6_symbols.size), dtype=float32)
        q_stage_7_symbols, self.halfband_3_filter_state[1] = dsp_fir_float_stream(q_stage_6_symbols,
                                                                          FIR_HALFBAND3_FLOAT_COEFFICIENTS,
                                                                          self.halfband_3_filter_state[1], gain=2)

        #8. Extract useful part of burst
        full_pad_length = int(BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT*BASEBAND_SAMPLING_FACTOR)
        if burst_ramp_periods[0] != 0:
            i_output_symbols = i_stage_7_symbols[full_pad_length:].copy()
            q_output_symbols = q_stage_7_symbols[full_pad_length:].copy()
        else:
            i_output_symbols = i_stage_7_symbols.copy()
            q_output_symbols = q_stage_7_symbols.copy()

        if burst_ramp_periods[1] != 0:
            i_output_symbols = i_output_symbols[:-full_pad_length].copy()
            q_output_symbols = q_output_symbols[:-full_pad_length].copy()


        #9. Perform ramping on signal

        i_ramped_symbols, q_ramped_symbols = power_ramping_float(i_output_symbols, q_output_symbols,
                                                                burst_ramp_periods)

        return i_ramped_symbols, q_ramped_symbols


    def _dac_conversion(self):
        """
        Converts baseband processed data into dac code representation, 
        """
        raise NotImplementedError

    def _analog_reconstruction(self):
        """
        Takes in DAC codes at rate Rs, converts to real floats with ZOH with 
        sampling rate Rif which is x8 more than Rs, then filters with analog reconstruction filter.
        """
        # coupling
        # gain error
        # offset
        raise NotImplementedError
