"""
tx_rx_utilities.py contains common functions for transmitter.py and reciever.py, including fir dsp block calculations,
power ramping, saving and reading .iq data files, etc.
"""
from typing import Literal
from pathlib import Path
from sys import byteorder
from numpy import array, complex64, pi, float64, ravel, zeros, int16, vstack, \
    fromfile, int64, concatenate, right_shift, clip, arange, cos, full, where, \
    left_shift, sqrt
from numpy import abs as np_abs
from numpy import max as np_max
from numpy import round as np_round
from numpy.typing import NDArray
from scipy.signal import convolve as sp_convolve

from .constants import TX_BB_SAMPLING_FACTOR

NUMBER_OF_FRACTIONAL_BITS = 17        # Relates to Q17 fixed point representation used
PLUTOSDR_DAC_BIT_NUMBER = 12          # Number of bits in pluto sdr
OPENTETRAPHYMAC_HW_DAC_BIT_NUMBER = 10    # Number of bits in openTETRAphymac hw implementation (AD9115)

VALID_ROUNDING_METHODS = ('rti', 'rtz', 'truncate', 'unbiased')
VALID_START_GUARD_PERIOD_OFFSETS = [0, 10, 12, 34, 120]
VALID_END_GUARD_PERIOD_OFFSETS = [0, 8, 10, 14, 16]

TX_RRC_Q17_COEFFICIENTS = array(
    [6, 60, 98, 109, 87, 37, -30, -93, -135, -141, -104, -30, 64, 153, 212, 222, 173, 74, -55, -181,
     -269, -292, -237, -115, 47, 204, 308, 319, 221, 26, -221, -452, -589, -564, -342, 66, 592, 1122,
     1511, 1613, 1321, 594, -510, -1834, -3125, -4075, -4361, -3712, -1958, 923, 4783, 9308, 14052,
     18499, 22130, 24505, 25332, 24505, 22130, 18499, 14052, 9308, 4783, 923, -1958, -3712, -4361,
     -4075, -3125, -1834, -510, 594, 1321, 1613, 1511, 1122, 592, 66, -342, -564, -589, -452, -221,
     26, 221, 319, 308, 204, 47, -115, -237, -292, -269, -181, -55, 74, 173, 222, 212, 153, 64, -30,
     -104, -141, -135, -93, -30, 37, 87, 109, 98, 60, 6], dtype=int64)

TX_RRC_FLOAT_COEFFICIENTS = array(
    [4.6028807E-05, 4.5548688E-04, 7.4681803E-04, 8.2835555E-04, 6.6207664E-04, 2.7856705E-04,
     -2.2615044E-04, -7.1247749E-04, -1.0332032E-03, -1.0747046E-03, -7.9234410E-04, -2.2970411E-04,
     4.8512162E-04, 1.1672165E-03, 1.6210295E-03, 1.6939947E-03, 1.3234488E-03, 5.6322449E-04,
     -4.1943332E-04, -1.3787722E-03, -2.0509101E-03, -2.2257303E-03, -1.8117221E-03, -8.7512349E-04,
     3.6038237E-04, 1.5580310E-03, 2.3496656E-03, 2.4371645E-03, 1.6885466E-03, 2.0193664E-04,
     -1.6844368E-03, -3.4467725E-03, -4.4899732E-03, -4.3008132E-03, -2.6063069E-03, 5.0252874E-04,
     4.5184297E-03, 8.5619008E-03, 1.1526610E-02, 1.2308078E-02, 1.0075723E-02, 4.5330846E-03,
     -3.8935654E-03, -1.3989874E-02, -2.3842659E-02, -3.1086056E-02, -3.3274468E-02, -2.8323304E-02,
     -1.4939181E-02, 7.0435614E-03, 3.6491636E-02, 7.1012035E-02, 1.0720996E-01, 1.4113323E-01,
     1.6883495E-01, 1.8696181E-01, 1.9326745E-01, 1.8696181E-01, 1.6883495E-01, 1.4113323E-01,
     1.0720996E-01, 7.1012035E-02, 3.6491636E-02, 7.0435614E-03, -1.4939181E-02, -2.8323304E-02,
     -3.3274468E-02, -3.1086056E-02, -2.3842659E-02, -1.3989874E-02, -3.8935654E-03, 4.5330846E-03,
     1.0075723E-02, 1.2308078E-02, 1.1526610E-02, 8.5619008E-03, 4.5184297E-03, 5.0252874E-04,
     -2.6063069E-03, -4.3008132E-03, -4.4899732E-03, -3.4467725E-03, -1.6844368E-03, 2.0193664E-04,
     1.6885466E-03, 2.4371645E-03, 2.3496656E-03, 1.5580310E-03, 3.6038237E-04, -8.7512349E-04,
     -1.8117221E-03, -2.2257303E-03, -2.0509101E-03, -1.3787722E-03, -4.1943332E-04, 5.6322449E-04,
     1.3234488E-03, 1.6939947E-03, 1.6210295E-03, 1.1672165E-03, 4.8512162E-04, -2.2970411E-04,
     -7.9234410E-04, -1.0747046E-03, -1.0332032E-03, -7.1247749E-04, -2.2615044E-04, 2.7856705E-04,
     6.6207664E-04, 8.2835555E-04, 7.4681803E-04, 4.5548688E-04, 4.6028807E-05], dtype=float64)

TX_LPF_Q17_COEFFICIENTS = array(
    [40, 136, 205, 218, 156, 27, -136, -283, -361, -329, -179, 57, 312, 503, 555, 429, 140,
     -240, -595, -804, -776, -485, 9, 570, 1022, 1199, 1004, 447, -339, -1128, -1659, -1720,
     -1216, -226, 1003, 2104, 2694, 2483, 1388, -411, -2476, -4191, -4904, -4088, -1501, 2723,
     8064, 13701, 18671, 22078, 23290, 22078, 18671, 13701, 8064, 2723, -1501, -4088, -4904,
     -4191, -2476, -411, 1388, 2483, 2694, 2104, 1003, -226, -1216, -1720, -1659, -1128, -339,
     447, 1004, 1199, 1022, 570, 9, -485, -776, -804, -595, -240, 140, 429, 555, 503, 312, 57,
     -179, -329, -361, -283, -136, 27, 156, 218, 205, 136, 40], dtype=int64)

TX_LPF_FLOAT_COEFFICIENTS = array(
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
     1.1896548E-03, 1.6598026E-03, 1.5670853E-03, 1.0378698E-03, 3.0866607E-04], dtype=float64)

TX_HALFBAND1_Q17_COEFFICIENTS = array(
    [-145, 0, 193, 0, -323, 0, 555, 0, -914, 0, 1434, 0, -2170, 0, 3221, 0, -4805, 0,
     7491, 0, -13392, 0, 41587, 65606, 41587, 0, -13392, 0, 7491, 0, -4805, 0, 3221, 0,
     -2170, 0, 1434, 0, -914, 0, 555, 0, -323, 0, 193, 0, -145], dtype=int64)

TX_HALFBAND1_FLOAT_COEFFICIENTS = array(
    [-1.1083445E-03, 0.0000000E+00, 1.4727359E-03, 0.0000000E+00, -2.4647851E-03, 0.0000000E+00,
     4.2366369E-03, 0.0000000E+00, -6.9756542E-03, 0.0000000E+00, 1.0942169E-02, 0.0000000E+00,
     -1.6552123E-02, 0.0000000E+00, 2.4572962E-02, 0.0000000E+00, -3.6657065E-02, 0.0000000E+00,
     5.7154625E-02, 0.0000000E+00, -1.0217133E-01, 0.0000000E+00, 3.1728380E-01, 5.0053275E-01,
     3.1728380E-01, 0.0000000E+00, -1.0217133E-01, 0.0000000E+00, 5.7154625E-02, 0.0000000E+00,
     -3.6657065E-02, 0.0000000E+00, 2.4572962E-02, 0.0000000E+00, -1.6552123E-02, 0.0000000E+00,
     1.0942169E-02, 0.0000000E+00, -6.9756542E-03, 0.0000000E+00, 4.2366369E-03, 0.0000000E+00,
     -2.4647851E-03, 0.0000000E+00, 1.4727359E-03, 0.0000000E+00, -1.1083445E-03], dtype=float64)

TX_HALFBAND2_Q17_COEFFICIENTS = array(
    [-304, 0, 711, 0, -2084, 0, 5063, 0, -11725, 0, 41035, 65681, 41035, 0, -11725, 0,
     5063, 0, -2084, 0, 711, 0, -304], dtype=int64)

TX_HALFBAND2_FLOAT_COEFFICIENTS = array(
    [-2.3200984E-03, 0.0000000E+00, 5.4240586E-03, 0.0000000E+00, -1.5900960E-02, 0.0000000E+00,
     3.8630295E-02, 0.0000000E+00, -8.9455216E-02, 0.0000000E+00, 3.1306928E-01, 5.0110528E-01,
     3.1306928E-01, 0.0000000E+00, -8.9455216E-02, 0.0000000E+00, 3.8630295E-02, 0.0000000E+00,
     -1.5900960E-02, 0.0000000E+00, 5.4240586E-03, 0.0000000E+00, -2.3200984E-03], dtype=float64)

TX_HALFBAND3_Q17_COEFFICIENTS = array(
    [0, 1181, 0, -7504, 0, 39118, 65482, 39118, 0, -7504, 0, 1181, 0], dtype=int64)

TX_HALFBAND3_FLOAT_COEFFICIENTS = array(
    [0.0000000E+00, 9.0088874E-03, 0.0000000E+00, -5.7248430E-02, 0.0000000E+00, 2.9844614E-01,
     4.9958680E-01, 2.9844614E-01, 0.0000000E+00, -5.7248430E-02, 0.0000000E+00, 9.0088874E-03,
     0.0000000E+00], dtype=float64)

###################################################################################################


def generate_ramping_lut_quantized(n: int, sps: int = TX_BB_SAMPLING_FACTOR) -> NDArray[int64]:
    """
    This function generates a Q17 quantized ramp-up LUT using a raised cosine function. The first sps+1 number of points
    are held constant at 0 and the last sps+1 number of points are held at ~1.

    :param n: The ramping period interval in number of symbols
    :type n: int
    :param sps: Defaults to TX_BB_SAMPLING_FACTOR, the sampling rate used for the resultant lut, the
     relationship results is that the function returns n*sps number of ramping points
    :type sps: int
    :return: The Q17 quantized ramping lut, with length (n*sps), where the last sps and
     first sps are held constant as 0 and 1 (1 aprox. due to fixed point Q17 representation) respective
    :rtype: NDArray[int64]
    """

    # We want the first and last symbol to have a constant envelope so we are not ramping during it
    # therefore we calculate using n-2 to account for this
    k = arange((n-2)*sps, dtype=int64)
    profile = 0.5 * (1.0 - cos(pi * k / (((n-2)*sps)-1)))
    lut = np_round(profile * (1 << NUMBER_OF_FRACTIONAL_BITS)).astype(int64)
    lut[0] = 0
    lut[-1] = (1 << NUMBER_OF_FRACTIONAL_BITS)

    # prepend and postpend the full symbol period 0 at the start and 1 at the end
    lut = concatenate((zeros(sps, dtype=int64), lut))
    lut = concatenate((lut, full(sps, (1 << NUMBER_OF_FRACTIONAL_BITS), dtype=int64)))

    return lut


# Generate the ramping luts at runtime for use later on
RAMPING_LUT_4 = generate_ramping_lut_quantized(4)
RAMPING_LUT_5 = generate_ramping_lut_quantized(5)
RAMPING_LUT_6 = generate_ramping_lut_quantized(6)
RAMPING_LUT_7 = generate_ramping_lut_quantized(7)
RAMPING_LUT_8 = generate_ramping_lut_quantized(8)
RAMPING_LUT_17 = generate_ramping_lut_quantized(17)

###################################################################################################


def generate_ramping_float_lut(n: int, sps: int = TX_BB_SAMPLING_FACTOR) -> NDArray[float64]:
    """
    This function generates a float ramp-up LUT using a raised cosine function. The first sps+1 number of points
    are held constant at 0 and the last sps+1 number of points are held at 1.

    :param n: The ramping period interval in number of symbols
    :type n: int
    :param sps: Defaults to TX_BB_SAMPLING_FACTOR, the sampling rate used for the resultant lut, the
     relationship results is that the function returns n*sps number of ramping points
    :type sps: int
    :return: The float ramping lut, with length (n*sps), where the last sps and
     first sps are held constant as 1 and 0 respective
    :rtype: NDArray[float64]
    """

    # We want the first and last symbol to have a constant envelope so we are not ramping during it
    # therefore we calculate using n-2 to account for this
    k = arange((n-2)*sps, dtype=float64)
    profile = 0.5 * (1.0 - cos(pi * k / (((n-2)*sps)-1)))
    lut = profile.astype(float64)
    lut[0] = 0
    lut[-1] = 1

    # prepend and postpend the full symbol period 0 at the start and 1 at the end
    lut = concatenate((zeros(sps, dtype=float64), lut))
    lut = concatenate((lut, full(sps, 1, dtype=float64)))

    return lut

###################################################################################################


def save_burst_iqfile(input_data: NDArray[int64] | NDArray[float64] | NDArray[int16], filepath: str,
                      dac_bits: int = PLUTOSDR_DAC_BIT_NUMBER, msb_aligned: bool = True,
                      endian: Literal["big"] | Literal["little"] = "little"):
    """
    Function that saves burst data to a .iq file specified by "filepath" to be read by GNU radio for an SDR.
    The output is saved in 16bits interleaved I and Q as : In | Qn | In+1 | Qn+1 | ...

    Allows for configuration of the number of DAC bits in the SDR, MSB or LSB aligned with 16bits, and
    little and big endian formats.

    Arguments are default for the PlutoSDR via GNUradio with only needing a LPF for reconstruction to remove
    higher order images.

    :param input_data: Input burst data either in quantized int64 Q17 format, float64 format, or DAC code int16 values.
    :type input_data: NDArray[int64] | NDArray[float64] | NDArray[int16]
    :param filepath: string file path that can be absolute or relative, that can be accepted by PathLib via Path()
    :type filepath: str
    :param dac_bits: Number of bits in the target SDR DAC, defaults to 12, function rounds input_data to meet this,
     allowing for direct reading of data without additional processing.
    :type dac_bits: int
    :param msb_aligned: Controls how the dac_bits number of bits are aligned with a 16 bit value, True implies that the
     upper MSB bits are used for the data, leaving 16-dac_bits number of lower bits zero, while False leaves upper bits
     zero instead
    :type msb_aligned: bool
    :param endian: Controls if data is written to file as little or big endian, sw like GNUradio uses the system type
     which is typcally little, however, for some applications like vector signal generators big endian is used.
    :type endian: Literal["big"] | Literal["little"]
    """

    # PlutoSDR can read data saved as 16-bits but in 12bit format, MSB aligned
    # i.e. 12 bits but shift left 4 times
    if input_data.ndim != 2:
        raise RuntimeError(f"IQ data to save to file has passed dimensions of {input_data.ndim}, expected 2")
    # now we need to interleave the data such that it goes i_ch,q_ch|i_ch,q_ch
    temp = ravel(vstack((input_data[0], input_data[1])), order="F")
    # peform conversion and rounding to dacBit number and align as needed
    if input_data.dtype == int64:
        # add rounding offset
        shift = NUMBER_OF_FRACTIONAL_BITS - (dac_bits-1)
        round_word = int64(1 << ((shift)-1))
        temp = right_shift((temp + round_word), (NUMBER_OF_FRACTIONAL_BITS - (dac_bits-1))).astype(int16)

    elif input_data.dtype == int16:
        # format would be the 10bit dac values for the openTETRAphymac hw
        hw_dac_num = OPENTETRAPHYMAC_HW_DAC_BIT_NUMBER-1
        if (temp.max() > ((1 << hw_dac_num)-1) or temp.min() < -(1 << hw_dac_num)):
            raise ValueError(f"Expected 10bit value with maximum + value of "
                             f"{((1 << OPENTETRAPHYMAC_HW_DAC_BIT_NUMBER-1)-1)}, found {np_max(temp)}")
        temp = left_shift(temp, (dac_bits-OPENTETRAPHYMAC_HW_DAC_BIT_NUMBER)).astype(int16)

    elif input_data.dtype == float64:
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
                           f" expected np.int64, np.int16 or np.float64")
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

def read_burst_iqfile(filepath: str, output_type: Literal["int64"] | Literal["float64"] = "int64",
                      dac_bits: int = PLUTOSDR_DAC_BIT_NUMBER, msb_aligned: bool = False,
                      endian: Literal["big"] | Literal["little"] = "little"
                      ) -> tuple[NDArray[int64], NDArray[int64]] | tuple[NDArray[float64], NDArray[float64]]:
    """
    Function that reads burst data from a .iq file specified by "filepath" from an SDR or from the save_burst_iqfile
    function. Expectes I and Q data to be interleaved in 16bit values as: In | Qn | In+1 | Qn+1 | ...

    Allows for configuration of the number of origin number ofDAC bits, MSB or LSB alignment in 16bits,
    and reading from little or big endian formats.

    Arguments are default for reading back data from save_burst_iqfile.

    :param filepath: target string file path that can be absolute or relative,
     that can be accepted by PathLib via Path()
    :type filepath: str
    :param output_type: Descriptor that configures the return to either be Q17 quantized int64 values or float64 values
    :type output_type: Literal["int64"] | Literal["float64"]
    :param dac_bits: Number of bits of the creator SDR DAC, defaults to 12
    :type dac_bits: int
    :param msb_aligned: Controls wether the dac_bits number of bits are aligned with a 16 bit value,
     True implies that the upper MSB bits are used for the data, leaving 16-dac_bits number of lower bits zero,
     while False (LSB) leaves upper bits zero instead
    :type msb_aligned: bool
    :param endian: Controls if data was written to file as little or big endian, sw like GNUradio uses the system type
     which is typcally little, however, for some applications like vector signal generators big endian is used.
    :type endian: Literal["big"] | Literal["little"]
    :return: The output data read from the target file, either int64 or float64 as configured by output_type
    :rtype: tuple[NDArray[int64], NDArray[int64]] | tuple[NDArray[float64], NDArray[float64]]
    """

    # PlutoSDR can read data saved as 16-bits but in 12bit format, MSB aligned
    # i.e. 12 bits but shift left 4 times
    if output_type not in ("int64", "float64"):
        raise ValueError(f"Passed output data type is: {output_type} expected int64 or float64")

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
        i_copy = i_signed.astype(float64)
        i_copy /= scale
        q_copy = q_signed.astype(float64)
        q_copy /= scale
    return i_copy, q_copy

###################################################################################################


def oversample_data_quantized(input_data: NDArray[complex64], over_sample_rate: int) -> NDArray[int64]:
    """
    Function performs over_sample_rate upsampling (with zero insertions) and quantization of unit-circle mapped
    input_data complex symbol data into Q17 quantized signed integers. Returning a 2-dimensional output array of
    int64 data

    :param input_data: Input array of complex64 symbol data, must be mapped such that maximum magnitude is 1
    :type input_data: NDArray[complex64]
    :param over_sample_rate: Integer oversampling rate, typical usage is x8
    :type over_sample_rate: int
    :return: Q17 quantized I channel and Q channel data, upsampled with zero insertions stored in int64 2-dimensional
     numpy array as [Ichannel:[], Qchannel:[]]
    :rtype: NDArray[int64]
    """

    output_data = zeros(shape=(2, input_data.size*over_sample_rate), dtype=int64)
    # because the modulation mapped only on the unit circle, the max value is 1, and min is zero,
    # the culmative baseband processing gain ~ one, so we do not need to scale to prevent overflow in this case
    temp_i_val = np_round((input_data.real * (1 << NUMBER_OF_FRACTIONAL_BITS)))
    temp_q_val = np_round((input_data.imag * (1 << NUMBER_OF_FRACTIONAL_BITS)))
    output_data[0][0::over_sample_rate] = temp_i_val.astype(int64)
    output_data[1][0::over_sample_rate] = temp_q_val.astype(int64)
    return output_data

###################################################################################################


def oversample_data_float(input_data: NDArray[complex64], over_sample_rate: int) -> NDArray[float64]:
    """
    Function performs over_sample_rate upsampling (with zero insertions) and conversion of unit-circle mapped
    input_data complex symbol data into non-complex floats. Returning a 2-dimensional output array of
    float64 data

    :param input_data: Input array of complex64 symbol data, must be mapped such that maximum magnitude is 1
    :type input_data: NDArray[complex64]
    :param over_sample_rate: Integer oversampling rate, typical usage is x8
    :type over_sample_rate: int
    :return: Float I channel and Q channel data, upsampled with zero insertions stored in float64 2-dimensional
     numpy array as [Ichannel:[], Qchannel:[]]
    :rtype: NDArray[float64]
    """

    output_data = zeros(shape=(2, input_data.size*over_sample_rate), dtype=float64)
    temp_i_val = input_data.real.astype(float64)
    temp_q_val = input_data.imag.astype(float64)
    output_data[0][0::over_sample_rate] = temp_i_val.astype(float64)
    output_data[1][0::over_sample_rate] = temp_q_val.astype(float64)
    return output_data

###################################################################################################


def assert_tail_is_zero(i_ch: NDArray[int64 | float64],
                        q_ch: NDArray[int64 | float64], samples: int) -> None:
    """
    This function verifies that the sum magnitude of the power of last 'samples" number of values
    in i_ch and q_ch components is zero. If it is not, it raises a RuntimeError, since power ramping must have been
    misconfigured and will result in spectral splatter.

    :param i_ch: Ramped i_ch data to verify
    :type i_ch: NDArray[int64 | float64]
    :param q_ch: Ramped q_ch data to verify
    :type q_ch: NDArray[int64 | float64]
    :param samples: The integer number of last samples to check as i/q_ch[-samples:]
    :type samples: int
    :raises RuntimeError: If the sum of the tail of the channel components squared is not zero,
         then the power does not ramp down to zero and therefore power ramping has been misconfigured.
    """

    tail_p = i_ch[-samples:].astype(int64)**2 + q_ch[-samples:].astype(int64)**2
    if np_max(tail_p) != 0:
        raise RuntimeError("Burst tail is not fully gated to zero.")

###################################################################################################


def q17_rounding(accumulated_results: NDArray[int64],
                 rounding: Literal["rti"] | Literal["rtz"] | Literal['unbiased'] | Literal['truncate'] = "rti",
                 right_shift_number: int = NUMBER_OF_FRACTIONAL_BITS):
    """
    Performs fixed point rounding on int64 object data, typically used for rounding down convolution
    accumulation result down to Q17 however can support rounding down to any number of bits.

    :param accumulated_results: int64 data that is in 2's compliments to be rounded down
    :type accumulated_results: NDArray[int64]
    :param rounding: The type of rounding, "rti", "rtz" and "truncate are the supported biased methods for ECP5 DSP
     blocks. Unbiased rounding is for sw usage and uses bankers rounding.
    :type rounding: Literal["rti"] | Literal["rtz"] | Literal['unbiased'] | Literal['truncate']
    :param right_shift_number: The number bits to round down, defaults to Q17 rounding e.g. 17
    :type right_shift_number: int
    """

    if rounding not in VALID_ROUNDING_METHODS:
        raise ValueError(f"Rounding method passed of {rounding} invalid, expected type in: {VALID_ROUNDING_METHODS}")

    if right_shift_number == 0:
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
            trunc = right_shift(a, right_shift_number)  # floor
            rem = a & int64((1 << right_shift_number) - 1)  # remainder bits
            half = int64(1 << (right_shift_number - 1))

            gt = rem > half
            eq = rem == half

            # if exactly halfway, np_round up only if we are even value
            inc = gt | (eq & ((trunc & 1) == 1))

            trunc = trunc + inc.astype(int64)
            return trunc * sign
    return right_shift(y, right_shift_number)

###################################################################################################


def dsp_fir_i_q_stream_convolve(i_ch: NDArray[int64 | float64], q_ch: NDArray[int64 | float64],
                                h_coefficients: NDArray[int64 | float64],
                                input_fir_state: NDArray[int64 | float64] | None = None,
                                gain: Literal[1] | Literal[2] | Literal[4] | Literal[8] = 1
                                ) -> tuple[NDArray[int64 | float64], NDArray[int64 | float64],
                                           NDArray[int64 | float64], NDArray[int64 | float64]]:
    """
    Handles performing FIR filter convolution process with continuous state control. Allows for "gain" by reducing
    rounding amount or adding gain after convolution, to account for loss of amplitude due to oversampling
    and zero insertion. Handles Q17 int64 values or float64 values

    Returns isolated input convolution result, as well as a FIR memory states that can be used when transmiting
    continuous bursts for the subseqent burst.

    :param i_ch: I channel Float or Q17 formated input data to be filtered, should include isolation flushes/pads
     if burst is desired to be isolated or isolate subsequent burst
    :type i_ch: NDArray[int64 | float64]
    :param q_ch: Q channel Float or Q17 formated input data to be filtered, should include isolation flushes/pads
     if burst is desired to be isolated or isolate subsequent burst.
    :type q_ch: NDArray[int64 | float64]
    :param h_coefficients: Float or Q17 formated filter coefficents, can be of any order, DC gain of filter should be
     <=1 to prevent overflow from occuring.
    :type h_coefficients: NDArray[int64 | float64]
    :param input_fir_state: The FIR memory state, typically taken as a return value of the preceding burst, Defaults to
     None in the case there is no preceeding burst or no FIR memory is desired to be tracked. If not none, should be
     2-dim first row should be for i_ch fir state and 2nd row for q_ch fir state
    :type input_fir_state: NDArray[int64 | float64]
    :param gain: Gain of block taken after convolution during rounding of accumulated result, limited to 1, 2, 4, or 8
    :type gain: Literal[1] | Literal[2] | Literal[4] | Literal[8]
    :return: Returns tuple packed as: (i_ch result, q_ch result, i_ch_fir_state, q_ch_fir_state)
    :rtype: tuple[NDArray[int64 | float64], NDArray[int64 | float64] | None,
    NDArray[int64 | float64], NDArray[int64 | float64] | None]
    """

    data_type = i_ch.dtype
    if data_type != q_ch.dtype:
        raise RuntimeError(f"Mismatched i and q data types passed, q_ch type: {q_ch.dtype}"
                           f", expected {data_type} to match")

    if data_type != h_coefficients.dtype:
        raise RuntimeError(f"Mismatched i and h coefficents types passed, h_coefficents: {h_coefficients.dtype}"
                           f", expected {data_type} to match")

    if input_fir_state is not None and data_type != input_fir_state.dtype:
        raise RuntimeError(f"Mismatched i and input_fir_state_1 types passed, h_coefficents: {input_fir_state.dtype}"
                           f", expected {data_type} to match")

    if gain not in [1, 2, 4, 8]:
        raise ValueError(f"Passed gain value: {gain}, is not 1 or multiple of 2.")

    n_taps = h_coefficients.size

    i_ch_output = zeros(shape=i_ch.shape, dtype=data_type)
    i_fir_state = zeros(shape=n_taps, dtype=data_type)
    q_ch_output = zeros(shape=q_ch.shape, dtype=data_type)
    q_fir_state = zeros(shape=n_taps, dtype=data_type)

    for i, ch in enumerate((i_ch, q_ch)):

        if input_fir_state is not None:
            if input_fir_state.ndim != 2:
                # if we only passed i_ch, then input_fir_state may just be 1-dimensional
                fir_state = input_fir_state.copy() if i == 1 else zeros(shape=(n_taps-1), dtype=data_type)
            else:
                if input_fir_state.shape[1] != (n_taps - 1):
                    raise ValueError(f"Length of FIR state passed is {input_fir_state.shape[1]},"
                                     f" expected {(n_taps - 1)} based on h_coefficients passed.")
                fir_state = input_fir_state[i].copy()
        else:
            fir_state = zeros(shape=(n_taps-1), dtype=data_type)
        # Prepend the state values, burst isolation flushes are handled externally to the function
        input_data_extended: NDArray[int64 | float64]
        input_data_extended = concatenate((fir_state, ch))

        # Stream the convolution results, note that prepended values come from state/memory of FIR from previous burst
        full_accumulated_result = sp_convolve(input_data_extended, h_coefficients,
                                              method="direct", mode="full").astype(data_type)

        # Data-aligned segment:
        burst_segment = full_accumulated_result[(n_taps-1):(n_taps-1 + i_ch.size)].copy()
        # The rounding implemented is not pure unbiased rounding due to behaviour with negative values,
        # however it provides greater precision than pure truncation and follow ECP5 DSP slices implementation

        if gain == 1:
            shift_value = NUMBER_OF_FRACTIONAL_BITS
        else:
            shift_value = int(NUMBER_OF_FRACTIONAL_BITS - (gain // 2))

        if burst_segment.dtype == int64:
            output_accumulated_result = \
                q17_rounding(burst_segment, right_shift_number=shift_value)  # type: ignore[arg-type]

            # Saturate output incase of clipping, it is not expected for this to occur due to the FIR gains and such
            output_accumulated_result = clip(output_accumulated_result, -(1 << NUMBER_OF_FRACTIONAL_BITS),
                                             (1 << NUMBER_OF_FRACTIONAL_BITS)-1).astype(int64)
        else:
            output_accumulated_result = burst_segment * gain
        # The new state of the FIR is the last n_taps-1 values of what was passed
        new_fir_state = input_data_extended[-(n_taps-1):].copy()

        if i == 1:
            i_ch_output = output_accumulated_result
            i_fir_state = new_fir_state
        else:
            q_ch_output = output_accumulated_result
            q_fir_state = new_fir_state

    return i_ch_output, q_ch_output, i_fir_state, q_fir_state

###################################################################################################


def dsp_fir_quantized_stream(input_symbols: NDArray[int64], h_coefficients: NDArray[int64],
                             input_fir_state: NDArray[int64] | None = None,
                             gain: Literal[1] | Literal[2] | Literal[4] | Literal[8] = 1
                             ) -> tuple[NDArray[int64], NDArray[int64]]:
    """
    Handles performing FIR filter convolution process with continuous state control, models the 54bit accumulation
    result stored int64 and subsequent rounding to Q17. Allows for "gain" by reducing rounding amount, to account for
    loss of amplitude due to oversampling and zero insertion.

    Returns isolated input convolution result, as well as a FIR memory state that can be used when transmiting
    continuous bursts for the subseqent burst.

    :param input_symbols: Q17 formated input data to be filtered, should include isolation flushes/pads if burst
     isolation from FIR memory or flushing of memory for subsequent bursts is required.
    :type input_symbols: NDArray[int64]
    :param h_coefficients: Q17 formated filter coefficents, can be of any order, DC gain of filter should be <=1 to
     prevent overflow from occuring.
    :type h_coefficients: NDArray[int64]
    :param input_fir_state: The FIR memory state, typically taken as a return value of the preceding burst, Defaults to
     None in the case there is no preceeding burst or no FIR memory is desired to be tracked.
    :type input_fir_state: NDArray[int64] | None
    :param gain: Gain of block taken after convolution during rounding of accumulated result, limited to 1, 2, 4, or 8
    :type gain: Literal[1] | Literal[2] | Literal[4] | Literal[8]
    :return: Returns (result of convolution of input_symbols, a new fir_state of the memory of the block for use with
     subsequent blocks)
    :rtype: tuple[NDArray[int64], NDArray[int64]]
    """

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
    output_accumulated_result = clip(output_accumulated_result, -(1 << NUMBER_OF_FRACTIONAL_BITS),
                                     (1 << NUMBER_OF_FRACTIONAL_BITS)-1).astype(int64)

    # The new state of the FIR is the last n_taps-1 values of what was passed
    new_fir_state = input_data_extended[-(n_taps-1):].copy()

    return output_accumulated_result, new_fir_state

###################################################################################################


def dsp_fir_float_stream(input_symbols: NDArray[float64], h_coefficients: NDArray[float64],
                         input_fir_state_1: NDArray[float64] | None = None,
                         gain: Literal[1] | Literal[2] | Literal[4] | Literal[8] = 1
                         ) -> tuple[NDArray[float64], NDArray[float64]]:
    """
    Handles performing FIR filter convolution process with continuous state control, using float64 values.
    Allows for "gain" by reducing rounding amount, to account for loss of amplitude due to oversampling
    and zero insertion.

    Returns isolated input convolution result, as well as a FIR memory state that can be used when transmiting
    continuous bursts for the subseqent burst.

    :param input_symbols: Float input data to be filtered, should include isolation flushes/pads if burst
     isolation from FIR memory or flushing of memory for subsequent bursts is required.
    :type input_symbols: NDArray[float64]
    :param h_coefficients: Float filter coefficents, can be of any order, DC gain of filter should be <=1 to
     prevent overflow from occuring.
    :type h_coefficients: NDArray[float64]
    :param input_fir_state: The FIR memory state, typically taken as a return value of the preceding burst, Defaults to
     None in the case there is no preceeding burst or no FIR memory is desired to be tracked.
    :type input_fir_state: NDArray[float64] | None
    :param gain: Gain of block taken after convolution, limited to 1, 2, 4, or 8
    :type gain: Literal[1] | Literal[2] | Literal[4] | Literal[8]
    :return: Returns (result of convolution of input_symbols, a new fir_state of the memory of the block for use with
     subsequent blocks)
    :rtype: tuple[NDArray[float64], NDArray[float64]]
    """

    if gain not in [1, 2, 4, 8]:
        raise ValueError(f"Passed gain value: {gain}, is not 1 or multiple of 2.")

    n_taps = h_coefficients.size
    if input_fir_state_1 is not None:
        if input_fir_state_1.size != (n_taps - 1):
            raise ValueError(f"Length of FIR state passed is {input_fir_state_1.size},"
                             f" expected {(n_taps - 1)} based on h_coefficients passed.")
    else:
        input_fir_state_1 = zeros(shape=(n_taps-1), dtype=float64)

    # Prepend the state values, burst isolation flushes are handled externally to the function
    input_data_extended = concatenate((input_fir_state_1, input_symbols))

    # Stream the convolution results, note that prepended values come from state/memory of FIR from previous burst
    full_accumulated_result = sp_convolve(input_data_extended, h_coefficients,
                                          method="direct", mode="full").astype(float64)

    # Data-aligned segment:
    burst_segment = full_accumulated_result[(n_taps-1):(n_taps-1 + input_symbols.size)].copy()

    # Gain
    burst_segment *= gain

    # The new state of the FIR is the last n_taps-1 values of what was passed
    new_fir_state = full_accumulated_result[-(n_taps-1):].copy()

    return burst_segment, new_fir_state

###################################################################################################


def power_ramping_quantized(i_ch: NDArray[int64], q_ch: NDArray[int64],
                            burst_ramp_periods: tuple[int, int],
                            sps: int = TX_BB_SAMPLING_FACTOR) -> tuple[NDArray[int64], NDArray[int64]]:
    """
    Performs amplitude ramping, either up and/or down, on individual I and Q channels in Q17 quantized formats
    using a raised cosine LUTs. Ramping duration is dependent on the passed burst_ramp_periods,
    which is in number bits of the original sample rate, and the upsample rate which corresponds to the sample rate
    of the passed i_ch and q_ch input data.

    If no up or down ramping is desired, passing 0 as the first or second element of burst_ramp_periods tuple
    respectively, will prevent ramping.

    :param i_ch: Input Q17 i_ch data to be ramped, at rate sps, stored in int64 numpy array
    :type i_ch: NDArray[int64]
    :param q_ch: Input Q17 q_ch data to be ramped, at rate sps, stored in int64 numpy array
    :type q_ch: NDArray[int64]
    :param burst_ramp_periods: Tuple showing the (start, end) ramping periods in number of bits at the base sample rate
    :type burst_ramp_periods: tuple[int, int]
    :param sps: The sample rate factor over the base/original sampling rate that i_ch, and q_ch are operating at
    :type sps: int
    :return: Returns ramped i_ch and q_ch data in Q17 format, using the settings passed in burst_ramp_periods and sps
    :rtype: tuple[NDArray[int64], NDArray[int64]]
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

    up_ramp = zeros(2, dtype=int64)
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
        case _:
            if burst_ramp_periods[0] != 0:
                raise ValueError(f"Invalid passed burst ramp period of {burst_ramp_periods[0]}")

    if burst_ramp_periods[0] != 0:
        envelope[:n_start] = up_ramp

    if burst_ramp_periods[1] not in VALID_END_GUARD_PERIOD_OFFSETS:
        raise ValueError(f"Passed end burst guard ramp period of {burst_ramp_periods[1]},"
                         f" expected value in: {VALID_END_GUARD_PERIOD_OFFSETS}")

    down_ramp = zeros(2, dtype=int64)
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
        case _:
            if burst_ramp_periods[1] != 0:
                raise ValueError(f"Invalid passed burst ramp period of {burst_ramp_periods[1]}")

    if burst_ramp_periods[1] != 0:
        envelope[-n_end:] = down_ramp[::-1]

    i_ch_product = i_ch.astype(int64) * envelope.astype(int64)
    q_ch_product = q_ch.astype(int64) * envelope.astype(int64)

    i_ch_result = q17_rounding(i_ch_product.copy())
    q_ch_result = q17_rounding(q_ch_product.copy())
    return i_ch_result, q_ch_result

###################################################################################################


def power_ramping_float(i_ch: NDArray[float64], q_ch: NDArray[float64],
                        burst_ramp_periods: tuple[int, int],
                        sps: int = TX_BB_SAMPLING_FACTOR) -> tuple[NDArray[float64], NDArray[float64]]:
    """
    Performs amplitude ramping, either up and/or down, on individual I and Q channels in float formats
    using a raised cosine LUTs. Ramping duration is dependent on the passed burst_ramp_periods,
    which is in number bits of the original sample rate, and the upsample rate which corresponds to the sample rate
    of the passed i_ch and q_ch input data.

    If no up or down ramping is desired, passing 0 as the first or second element of burst_ramp_periods tuple
    respectively, will prevent ramping.

    :param i_ch: Input float i_ch data to be ramped, at rate sps, stored in int64 numpy array
    :type i_ch: NDArray[float64]
    :param q_ch: Input float q_ch data to be ramped, at rate sps, stored in int64 numpy array
    :type q_ch: NDArray[float64]
    :param burst_ramp_periods: Tuple showing the (start, end) ramping periods in number of bits at the base sample rate
    :type burst_ramp_periods: tuple[int, int]
    :param sps: The sample rate factor over the base/original sampling rate that i_ch, and q_ch are operating at
    :type sps: int
    :return: Returns ramped i_ch and q_ch data in float format, using the settings passed in burst_ramp_periods and sps
    :rtype: tuple[NDArray[float64], NDArray[float64]]
    """

    n = i_ch.size
    if burst_ramp_periods[0] not in VALID_START_GUARD_PERIOD_OFFSETS:
        raise ValueError(f"Passed start burst guard ramp period of {burst_ramp_periods[0]},"
                         f" expected value in: {VALID_START_GUARD_PERIOD_OFFSETS}")
    # For tetra, there are only a few defined burst delay periods,
    n_start = int(burst_ramp_periods[0] / 2) * sps
    n_end = int(burst_ramp_periods[1] / 2) * sps

    # create envelope of 1's
    envelope = full(n, 1, dtype=float64)

    up_ramp = zeros(2, dtype=float64)
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
        case _:
            if burst_ramp_periods[0] != 0:
                raise ValueError(f"Invalid passed burst ramp period of {burst_ramp_periods[0]}")

    if burst_ramp_periods[0] != 0:
        envelope[:n_start] = up_ramp

    if burst_ramp_periods[1] not in VALID_END_GUARD_PERIOD_OFFSETS:
        raise ValueError(f"Passed end burst guard ramp period of {burst_ramp_periods[1]},"
                         f" expected value in: {VALID_END_GUARD_PERIOD_OFFSETS}")

    down_ramp = zeros(2, dtype=float64)
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
        case _:
            if burst_ramp_periods[1] != 0:
                raise ValueError(f"Invalid passed burst ramp period of {burst_ramp_periods[1]}")

    if burst_ramp_periods[1] != 0:
        envelope[-n_end:] = down_ramp[::-1]

    i_ch_product = i_ch.astype(float64) * envelope.astype(float64)
    q_ch_product = q_ch.astype(float64) * envelope.astype(float64)

    return i_ch_product.astype(float64), q_ch_product.astype(float64)
