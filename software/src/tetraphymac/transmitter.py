# ZT - 2026
# Based on EN 300 392-2 V2.4.2              
from numpy import complex64, pi, float64, uint8, array, abs, zeros, float32, max, int16, vstack, rint, mean
from numpy import int64, uint8, empty, concatenate, round, right_shift, clip, arange, cos, full, where, left_shift, ravel, fromfile, sqrt
from numpy.typing import NDArray
from scipy.signal import convolve
from abc import ABC, abstractmethod
from typing import ClassVar, Tuple, List
from .constants import SUBSLOT_BIT_LENGTH, TIMESLOT_SYMBOL_LENGTH
from .modulation import dqpskModulator
from pathlib import Path
from sys import byteorder

GUARD_PERIOD_RAMP_BLANKING_SYMBOL_INTERVAL = 2
NUMBER_OF_FRACTIONAL_BITS = 17
PLUTOSDR_DAC_BIT_NUMBER = 12
OPENTETRAPHYMAC_HW_DAC_NUMBER = 10

BASEBAND_SAMPLING_FACTOR = 64
HALF_BASEBAND_SAMPLING_FACTOR = int(BASEBAND_SAMPLING_FACTOR / 2)
TETRA_SYMBOL_RATE = 18000

TRANSMIT_SIMULATION_SAMPLE_RATE = 10_368_000
TRANSMIT_SIMULATION_SAMPLING_FACTOR = 9

RRC_Q1_17_COEFFICIENTS = array([94, 42, 69, 77, 61, 26, -21, -66, -96, -100, -73, -21, 45, 108, 150, 157, 123, 52, -39, -128, 
                                -190, -206, -168, -81, 33, 144, 218, 226, 156, 19, -156, -319, -416, -399, -242, 47, 419, 794, 
                                1068, 1141, 934, 420, -361, -1297, -2210, -2881, -3084, -2625, -1385, 653, 3382, 6582, 9936, 
                                13080, 15648, 17328, 17912, 17328, 15648, 13080, 9936, 6582, 3382, 653, -1385, -2625, -3084, 
                                -2881, -2210, -1297, -361, 420, 934, 1141, 1068, 794, 419, 47, -242, -399, -416, -319, -156, 
                                19, 156, 226, 218, 144, 33, -81, -168, -206, -190, -128, -39, 52, 123, 157, 150, 108, 45, 
                                -21, -73, -100, -96, -66, -21, 26, 61, 77, 69, 42, 4], dtype=int64)

RRC_FLOAT_COEFFICIENTS = array([3.2547283E-05, 3.2207786E-04, 5.2808010E-04, 5.8573583E-04, 4.6815889E-04, 1.9697665E-04, 
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

FIR_LPF_Q1_17_COEFFICIENTS = array([40, 136, 205, 218, 156, 27, -136, -283, -361, -329, -179, 57, 312, 503, 555, 429, 140, 
                                    -240, -595, -804, -776, -485, 9, 570, 1022, 1199, 1004, 447, -339, -1128, -1659, -1720, 
                                    -1216, -226, 1003, 2104, 2694, 2483, 1388, -411, -2476, -4191, -4904, -4088, -1501, 2723, 
                                    8064, 13701, 18671, 22078, 23290, 22078, 18671, 13701, 8064, 2723, -1501, -4088, -4904, 
                                    -4191, -2476, -411, 1388, 2483, 2694, 2104, 1003, -226, -1216, -1720, -1659, -1128, -339, 
                                    447, 1004, 1199, 1022, 570, 9, -485, -776, -804, -595, -240, 140, 429, 555, 503, 312, 57, 
                                    -179, -329, -361, -283, -136, 27, 156, 218, 205, 136, 40], dtype=int64)

FIR_LPF_FLOAT_COEFFICIENTS = array([3.0866607E-04, 1.0378698E-03, 1.5670853E-03, 1.6598026E-03, 1.1896548E-03, 2.0867007E-04, 
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

FIR_HALFBAND1_Q1_17_COEFFICIENTS = array([-145, 0, 193, 0, -323, 0, 555, 0, -914, 0, 1434, 0, -2170, 0, 3221, 0, -4805, 0, 
                                          7491, 0, -13392, 0, 41587, 65606, 41587, 0, -13392, 0, 7491, 0, -4805, 0, 3221, 0, 
                                          -2170, 0, 1434, 0, -914, 0, 555, 0, -323, 0, 193, 0, -145], dtype=int64)

FIR_HALFBAND1_FLOAT_COEFFICIENTS = array([-1.1083445E-03, 0.0000000E+00, 1.4727359E-03, 0.0000000E+00, -2.4647851E-03, 0.0000000E+00, 
                                          4.2366369E-03, 0.0000000E+00, -6.9756542E-03, 0.0000000E+00, 1.0942169E-02, 0.0000000E+00, 
                                          -1.6552123E-02, 0.0000000E+00, 2.4572962E-02, 0.0000000E+00, -3.6657065E-02, 0.0000000E+00, 
                                          5.7154625E-02, 0.0000000E+00, -1.0217133E-01, 0.0000000E+00, 3.1728380E-01, 5.0053275E-01, 
                                          3.1728380E-01, 0.0000000E+00, -1.0217133E-01, 0.0000000E+00, 5.7154625E-02, 0.0000000E+00, 
                                          -3.6657065E-02, 0.0000000E+00, 2.4572962E-02, 0.0000000E+00, -1.6552123E-02, 0.0000000E+00, 
                                          1.0942169E-02, 0.0000000E+00, -6.9756542E-03, 0.0000000E+00, 4.2366369E-03, 0.0000000E+00, 
                                          -2.4647851E-03, 0.0000000E+00, 1.4727359E-03, 0.0000000E+00, -1.1083445E-03], dtype=float32)

FIR_HALFBAND2_Q1_17_COEFFICIENTS = array([-304, 0, 711, 0, -2084, 0, 5063, 0, -11725, 0, 41035, 65681, 41035, 0, -11725, 0, 
                                          5063, 0, -2084, 0, 711, 0, -304], dtype=int64)

FIR_HALFBAND2_FLOAT_COEFFICIENTS = array([-2.3200984E-03, 0.0000000E+00, 5.4240586E-03, 0.0000000E+00, -1.5900960E-02, 0.0000000E+00, 
                                          3.8630295E-02, 0.0000000E+00, -8.9455216E-02, 0.0000000E+00, 3.1306928E-01, 5.0110528E-01, 
                                          3.1306928E-01, 0.0000000E+00, -8.9455216E-02, 0.0000000E+00, 3.8630295E-02, 0.0000000E+00, 
                                          -1.5900960E-02, 0.0000000E+00, 5.4240586E-03, 0.0000000E+00, -2.3200984E-03], dtype=float32)

FIR_HALFBAND3_Q1_17_COEFFICIENTS = array([0, 1181, 0, -7504, 0, 39118, 65482, 39118, 0, -7504, 0, 1181, 0], dtype=int64)

FIR_HALFBAND3_FLOAT_COEFFICIENTS = array([0.0000000E+00, 9.0088874E-03, 0.0000000E+00, -5.7248430E-02, 0.0000000E+00, 2.9844614E-01, 
                                          4.9958680E-01, 2.9844614E-01, 0.0000000E+00, -5.7248430E-02, 0.0000000E+00, 9.0088874E-03, 
                                          0.0000000E+00], dtype=float32)

BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT = 30

VALID_ROUNDING_METHODS = ('rti', 'rtz', 'truncate', 'unbiased')
VALID_START_GUARD_PERIOD_OFFSETS = [10, 12, 34, 120]
VALID_END_GUARD_PERIOD_OFFSETS = [8, 10, 14, 16]

RAMPING_LUT_4 = []
RAMPING_LUT_5 = []
RAMPING_LUT_6 = []
RAMPING_LUT_7 = []
RAMPING_LUT_8 = []
RAMPING_LUT_17 = []
RAMPING_LUT_240 = []

VALID_RETURN_STAGE_VALUES = ("baseband", "dac", "tx") # possible output stages from transmitter.transmitBurst(), 
# baseband is int64 or float 32 representation of digital baseband, dac is 10bit or 24 bit code, tx is float output data after noise and filtering


###################################################################################################

def saveBurstasIQ(inputData: NDArray[int64] | NDArray[float32] | NDArray[int16], filepath:str,
                  dacBits:int=PLUTOSDR_DAC_BIT_NUMBER, MSBaligned:bool=True, endian:str="big"):
    # PlutoSDR can read data saved as 16-bits but in 12bit format, MSB aligned
    # i.e. 12 bits but shift left 4 times
    if inputData.ndim != 2:
        raise RuntimeError(f"IQ data to save to file has passed dimensions of {inputData.ndim}, expected 2")
    
    # now we need to interleave the data such that it goes I,Q|I,Q
    temp = ravel(vstack((inputData[0], inputData[1])), order="F")
    
    # peform conversion and rounding to dacBit number and aligned as needed
    if inputData.dtype == int64:
        # add rounding offset
        shift = NUMBER_OF_FRACTIONAL_BITS - (dacBits-1)
        rndWord = int64(1 << ((shift)-1))
        temp = right_shift((temp + rndWord), (NUMBER_OF_FRACTIONAL_BITS - (dacBits-1))).astype(int16)

    elif inputData.dtype == int16:
        # format would be the 10bit dac values for the openTETRAphymac hw
        if temp.max() > ((1 << OPENTETRAPHYMAC_HW_DAC_NUMBER-1)-1) or temp.min() < -(1<<(OPENTETRAPHYMAC_HW_DAC_NUMBER-1)):
            raise ValueError(f"Expected 10bit value with maximum + value of {((1 << OPENTETRAPHYMAC_HW_DAC_NUMBER-1)-1)}, found {max(temp)}")
        temp = left_shift(temp, (dacBits-OPENTETRAPHYMAC_HW_DAC_NUMBER)).astype(int16)

    elif inputData.dtype == float32:
        # normalize data by the complex magnitude of the channels
        scale = (1 << (dacBits - 1)) - 1
        peak = float(max(sqrt(inputData[0].astype(float64)**2 + inputData[1].astype(float64)**2)))
        if peak == 0:
            temp = zeros(shape=temp.shape, dtype=float64)
            raise RuntimeWarning(f"Peak complex magnitude of passed waveform to save to file is zero")
        else:
            temp = temp.astype(float64) / peak
        temp = round(temp * scale).astype(int16)
    else:
        raise RuntimeError(f"Datatype of passed IQ data to save to file is: {inputData.dtype}, expected np.int64, np.int16 or np.float32")
    
    # 12 bits but in 16bit format MSB aligned
    if MSBaligned:
        shift = 16 - dacBits
        temp = left_shift(temp, shift).astype(int16)

    # Check endianess
    if byteorder == endian:
        pass
    else:
        temp = temp.byteswap()

    # open file and save data
    filePath = Path(filepath)
    filePath.touch(exist_ok=True)
    try:
        temp.tofile(filePath)
    except Exception as e:
        print(e)

###################################################################################################

def readIQData(filepath:str, outputType:str="int64", dacBits:int=PLUTOSDR_DAC_BIT_NUMBER, 
               MSBaligned:bool=False, endian:str="big"):
    # PlutoSDR can read data saved as 16-bits but in 12bit format, MSB aligned
    # i.e. 12 bits but shift left 4 times
    
    if outputType not in ("int64", "float32"):
        raise ValueError(f"Passed output data type is: {outputType} expected int64 or float32")

    filePath = Path(filepath)
    temp = fromfile(filePath, dtype=int16)

    # Match endianess to system
    if byteorder == endian:
        pass
    else:
        temp = temp.byteswap()

    I_data = temp[0::2]
    Q_data = temp[1::2]

    # 12 bits is typically LSB aligned in 16bit
    if MSBaligned:
        shift = 16 - dacBits
        I_signed = (I_data >> shift).astype(int16)
        Q_signed = (Q_data >> shift).astype(int16)
    else:
        sign = int16(1 << (dacBits-1))
        mask = int16((1 << dacBits) - 1)
        I_unsigned = (I_data & mask).astype(int16)
        Q_unsigned = (Q_data & mask).astype(int16)
        I_signed = ((I_unsigned ^ sign) - sign).astype(int16)
        Q_signed = ((Q_unsigned ^ sign) - sign).astype(int16)

    # Convert data to desired output format
    if outputType == "int64":
        # We converting to Q17 format
        shift = NUMBER_OF_FRACTIONAL_BITS - (dacBits-1)
        I_copy = left_shift(int64(I_signed), shift).astype(int64)
        Q_copy = left_shift(int64(Q_signed), shift).astype(int64)

    else:
        scale = float(1 << (dacBits-1))
        I_copy = (float32(I_signed) / scale).astype(float32)
        Q_copy = (float32(Q_signed) / scale).astype(float32)
    
    return I_copy, Q_copy

###################################################################################################

def oversampleDataQuantized(inputData: NDArray[complex64], overSampleRate:int) -> NDArray[int64]:
    outputData = zeros(shape=(2, inputData.size*overSampleRate), dtype=int64)
    # because the modulation mapped only on the unit circle, the max value is 1, and min is zero, 
    # the gain of the filters and processing is slightly under one, so we do not need to scale to prevent overflow in this case
    tempIvalues = round((inputData.real * (1 << NUMBER_OF_FRACTIONAL_BITS)))
    tempQvalues = round((inputData.imag * (1 << NUMBER_OF_FRACTIONAL_BITS)))
    outputData[0][0::overSampleRate] = tempIvalues.astype(int64)
    outputData[1][0::overSampleRate] = tempQvalues.astype(int64)
    return outputData

###################################################################################################

def assertTailGoesToZero(I, Q, samples):
    tailP = I[-samples:].astype(int64)**2 + Q[-samples:].astype(int64)**2
    if max(tailP) != 0:
        raise RuntimeError("Burst tail is not fully gated to zero.")

###################################################################################################

def oversampleDataFloat(inputData: NDArray[complex64], overSampleRate:int) -> NDArray[float32]:
    outputData = zeros(shape=(2, inputData.size*overSampleRate), dtype=float32)
    tempIvalues = inputData.real.astype(float32)
    tempQvalues = inputData.imag.astype(float32)
    outputData[0][0::overSampleRate] = tempIvalues.astype(float32)
    outputData[1][0::overSampleRate] = tempQvalues.astype(float32)
    return outputData

###################################################################################################

def _q17Rounding(multiplierResult:NDArray[int64], rounding:str = "rti", rightShift:int=NUMBER_OF_FRACTIONAL_BITS):
    """
    Performs rounding on int64 object after accumulation during convolution
    """
    if rounding not in VALID_ROUNDING_METHODS:
        raise ValueError(f"Rounding method passed of {rounding} invalid, expected type in: {VALID_ROUNDING_METHODS}")

    if right_shift == 0:
        raise ValueError(f"Right shift value of: {right_shift}, invalid")
    
    y = multiplierResult.astype(int64, copy=True)
    match rounding:
        case "rti":
            # ECP5 FPGA style Round to Infinity by adding 2^16 and right shifting by 17 bits
            y += int64(1 << (rightShift-1))
        case "rtz":
            # ECP5 FPGA style Round to Zero by adding (2^16 - 1)
            y += int64((1 << (rightShift-1))-1)
        case "truncate":
            pass
        case "unbiased":
            # Software only implementation of Bankers rounding, unbiased
            sign = where(y < 0, int64(-1), int64(1))
            a = abs(y)
            trunc = right_shift(a, rightShift) # floor
            rem = a & int64((1 << rightShift) - 1) # remainder bits
            half = int64(1 << (rightShift - 1))

            gt = rem > half
            eq = rem == half

            # if exactly halfway, round up only if we are even value
            inc = gt | (eq & ((trunc & 1) == 1))

            trunc = trunc + inc.astype(int64)
            return trunc * sign
    
    return right_shift(y, rightShift)

###################################################################################################

def _dspBlockFIRQuantizedStream(inputSymbols:NDArray[int64], hCoef:NDArray[int64], 
                               inputState: NDArray[int64] | None = None,
                               gain:int=1) -> Tuple[NDArray[int64], NDArray[int64]]:
    '''
    Handles the convolution process with continous state control, 
    but models the 54bit accumulation results and subsequent truncation and rounding down to Q1.17
    '''
    if gain not in [1, 2, 4, 8]:
        raise ValueError(f"Passed gain value: {gain}, is not 1 or multiple of 2.")

    Ntaps = hCoef.size
    if inputState is not None:
        if inputState.size != (Ntaps - 1):
            raise ValueError(f"Length of FIR state passed is {inputState.size}, expected {(Ntaps - 1)} based on hCoef passed.")
    else:
        inputState = zeros(shape=(Ntaps-1), dtype=int64)

    # Prepend the state values, burst isolation flushes are handled externally to the function
    inputDataExt = concatenate((inputState, inputSymbols))

    # Stream the convolution results, note that prepended values come from state/memory of FIR from previous burst
    fullAccumulated = convolve(inputDataExt, hCoef, method="direct", mode="full").astype(int64)

    # Data-aligned segment:
    burstSegment = fullAccumulated[(Ntaps-1):(Ntaps-1 + inputSymbols.size)].copy()
    # The rounding implemented is not pure unbiased rounding due to behaviour with negative values, however it provides
    # greater precision than pure truncation and are the only methods available on the ECP5 DSP slices
    
    if gain == 1:
        shiftValue = NUMBER_OF_FRACTIONAL_BITS
    else:
        shiftValue = int(NUMBER_OF_FRACTIONAL_BITS - (gain // 2))

    outputAccumulated = _q17Rounding(burstSegment, rightShift=shiftValue)

    # Saturate output incase of clipping, it is not expected for this to occur due to the FIR gains and such
    outputAccumulated = clip(outputAccumulated, -(1<<NUMBER_OF_FRACTIONAL_BITS), (1<<NUMBER_OF_FRACTIONAL_BITS)-1).astype(int64)

    # The new state of the FIR is the last Ntaps-1 values of what was passed
    newState = inputDataExt[-(Ntaps-1):].copy()

    return outputAccumulated, newState

###################################################################################################

def _dspBlockFIRFloatStream(inputSymbols:NDArray[float32], hCoef:NDArray[float32], 
                               inputState: NDArray[float32] | None = None,
                               gain:int=1) -> Tuple[NDArray[float32], NDArray[float32]]:
    '''
    Handles the convolution process with continous state control but uses float32 values
    '''
    if gain not in [1, 2, 4, 8]:
        raise ValueError(f"Passed gain value: {gain}, is not 1 or multiple of 2.")
    
    Ntaps = hCoef.size
    if inputState is not None:
        if inputState.size != (Ntaps - 1):
            raise ValueError(f"Length of FIR state passed is {inputState.size}, expected {(Ntaps - 1)} based on hCoef passed.")
    else:
        inputState = zeros(shape=(Ntaps-1), dtype=float32)

    # Prepend the state values, burst isolation flushes are handled externally to the function
    inputDataExt = concatenate((inputState, inputSymbols))

    # Stream the convolution results, note that prepended values come from state/memory of FIR from previous burst
    fullAccumulated = convolve(inputDataExt, hCoef, method="direct", mode="full").astype(float32)

    # Data-aligned segment:
    burstSegment = fullAccumulated[(Ntaps-1):(Ntaps-1 + inputSymbols.size)].copy()

    # Gain
    burstSegment *= gain

    # The new state of the FIR is the last Ntaps-1 values of what was passed
    newState = fullAccumulated[-(Ntaps-1):].copy()

    return burstSegment, newState

###################################################################################################

def _raisedCosineLUTQuantized(N: int, sps:int=BASEBAND_SAMPLING_FACTOR) -> NDArray[int64]:
    # We want the last symbol to constant envelope, so we are not ramping during it
    # therefore we calculate using N-2 to account for this
    n = arange((N-2)*sps, dtype=int64)
    profile = 0.5 * (1.0 - cos(pi * n / (((N-2)*sps)-1)))
    lut = round(profile * (1 << NUMBER_OF_FRACTIONAL_BITS)).astype(int64)
    lut[0] = 0
    lut[-1] = (1 << NUMBER_OF_FRACTIONAL_BITS)

    # prepend and postpend the full symbol period 0 at the start and 1 at the end
    lut = concatenate((zeros(sps, dtype=int64), lut))
    lut = concatenate((lut, full(sps, (1 << NUMBER_OF_FRACTIONAL_BITS), dtype=int64)))

    return lut

###################################################################################################

def _raisedCosineFloat(N: int, sps:int=BASEBAND_SAMPLING_FACTOR) -> NDArray[float32]:
    # We want the last symbol to constant envelope, so we are not ramping during it
    # therefore we calculate using N-2 to account for this
    n = arange((N-2)*sps, dtype=float32)
    profile = 0.5 * (1.0 - cos(pi * n / (((N-2)*sps)-1)))
    lut = profile.astype(float32)
    lut[0] = 0
    lut[-1] = 1

    # prepend and postpend the full symbol period 0 at the start and 1 at the end
    lut = concatenate((zeros(sps, dtype=float32), lut))
    lut = concatenate((lut, full(sps, 1, dtype=float32)))

    return lut

###################################################################################################

def _powerRampingQuantized(I:NDArray[int64], Q:NDArray[int64], 
                           burstGuardRampPeriods:List, 
                           sps:int=BASEBAND_SAMPLING_FACTOR) -> Tuple[NDArray[int64], NDArray[int64]]:
    """
    Performs power ramping on the quantized signal using LUTs and raised cosine shape
    """
    N = I.size
    if burstGuardRampPeriods[0] not in VALID_START_GUARD_PERIOD_OFFSETS:
        raise ValueError(f"Passed start burst guard ramp period of {burstGuardRampPeriods[0]}, expected value in: {VALID_START_GUARD_PERIOD_OFFSETS}")
    
    # For tetra, there are only a few defined burst delay periods,
    Nstart = int(burstGuardRampPeriods[0] / 2) * sps
    Nend = int(burstGuardRampPeriods[1] / 2) * sps

    # create envelope of 1's
    envelope = full(N, (1 << NUMBER_OF_FRACTIONAL_BITS), dtype=int64)
    
    match int(burstGuardRampPeriods[0] / 2):
        case 5:
            # Discontinuous SB or NDB
            upRamp = RAMPING_LUT_5
        case 6:
            # Continuous SB or NDB
            upRamp = RAMPING_LUT_6
        case 17:
            # NUB or CB
            upRamp = RAMPING_LUT_17
        case 240:
            raise NotImplementedError(f"Uplink linearization (LB) ramping not implemented yet")
    
    envelope[:Nstart] = upRamp

    if burstGuardRampPeriods[1] not in VALID_END_GUARD_PERIOD_OFFSETS:
        raise ValueError(f"Passed end burst guard ramp period of {burstGuardRampPeriods[1]}, expected value in: {VALID_END_GUARD_PERIOD_OFFSETS}")
 
    match int(burstGuardRampPeriods[1] / 2):
        case 4:
            # Discontinuous SB or NDB
            downRamp = RAMPING_LUT_4
        case 5:
            # Continuous SB or NDB
            downRamp = RAMPING_LUT_5
        case 7:
            # NUB or CB or LB
            downRamp = RAMPING_LUT_7
        case 8:
            downRamp = RAMPING_LUT_8

    envelope[-Nend:] = downRamp[::-1]

    productI = I.astype(int64) * envelope.astype(int64)
    productQ = Q.astype(int64) * envelope.astype(int64)

    resultI = _q17Rounding(productI.copy())
    resultQ = _q17Rounding(productQ.copy())

    return resultI, resultQ


###################################################################################################

def _powerRampingFloat(I:NDArray[float32], Q:NDArray[float32], 
                           burstGuardRampPeriods:List, 
                           sps:int=BASEBAND_SAMPLING_FACTOR) -> Tuple[NDArray[float32], NDArray[float32]]:
    """
    Performs power ramping on the quantized signal using LUTs and raised cosine shape
    """
    N = I.size
    if burstGuardRampPeriods[0] not in VALID_START_GUARD_PERIOD_OFFSETS:
        raise ValueError(f"Passed start burst guard ramp period of {burstGuardRampPeriods[0]}, expected value in: {VALID_START_GUARD_PERIOD_OFFSETS}")
    
    # For tetra, there are only a few defined burst delay periods,
    Nstart = int(burstGuardRampPeriods[0] / 2) * sps
    Nend = int(burstGuardRampPeriods[1] / 2) * sps

    # create envelope of 1's
    envelope = full(N, 1, dtype=float32)
    
    match int(burstGuardRampPeriods[0] / 2):
        case 5:
            # Discontinuous SB or NDB
            upRamp = _raisedCosineFloat(5)
        case 6:
            # Continuous SB or NDB
            upRamp = _raisedCosineFloat(6)
        case 17:
            # NUB or CB
            upRamp = _raisedCosineFloat(17)
        case 240:
            raise NotImplementedError(f"Uplink linearization (LB) ramping not implemented yet")
    
    envelope[:Nstart] = upRamp

    if burstGuardRampPeriods[1] not in VALID_END_GUARD_PERIOD_OFFSETS:
        raise ValueError(f"Passed end burst guard ramp period of {burstGuardRampPeriods[1]}, expected value in: {VALID_END_GUARD_PERIOD_OFFSETS}")
 
    match int(burstGuardRampPeriods[1] / 2):
        case 4:
            # Discontinuous SB or NDB
            downRamp = _raisedCosineFloat(4)
        case 5:
            # Continuous SB or NDB
            downRamp = _raisedCosineFloat(5)
        case 7:
            # NUB or CB or LB
            downRamp = _raisedCosineFloat(7)
        case 8:
            downRamp = _raisedCosineFloat(8)

    envelope[-Nend:] = downRamp[::-1]

    productI = I.astype(float32) * envelope.astype(float32)
    productQ = Q.astype(float32) * envelope.astype(float32)

    return productI.astype(float32), productQ.astype(float32)


###################################################################################################
class Transmitter(ABC):
    phaseReference = complex64(1 + 0j)

    rrcFilterState: ClassVar[NDArray]
    lpfFilterState: ClassVar[NDArray]
    halfband1State: ClassVar[NDArray]
    halfband2State: ClassVar[NDArray]
    halfband3State: ClassVar[NDArray]

    def __init__(self):
        global RAMPING_LUT_4
        global RAMPING_LUT_5
        global RAMPING_LUT_6
        global RAMPING_LUT_7
        global RAMPING_LUT_8
        global RAMPING_LUT_17
        if len(RAMPING_LUT_4) == 0:
            # Fill out the LUTs for usage when ramping
            RAMPING_LUT_4 = _raisedCosineLUTQuantized(4)
            RAMPING_LUT_5 = _raisedCosineLUTQuantized(5)
            RAMPING_LUT_6 = _raisedCosineLUTQuantized(6)
            RAMPING_LUT_7 = _raisedCosineLUTQuantized(7)
            RAMPING_LUT_8 = _raisedCosineLUTQuantized(8)
            RAMPING_LUT_17 = _raisedCosineLUTQuantized(17)

    @abstractmethod
    def _basebandProcessing(self, inputComplexSymbols:NDArray[complex64], burstGuardRampPeriods:List) -> Tuple[NDArray, NDArray]:
        """
        Converts modulation bits into symbols, performs upsampling, ramping, and filtering
        """
        raise NotImplementedError

    @abstractmethod
    def _dacConversion(self):
        """
        Converts baseband processed data into dac code representation, 
        """
        raise NotImplementedError

    @abstractmethod
    def _analogReconstruction(self):
        """
        Takes in DAC codes at rate Rs, converts to real floats with ZOH with 
        sampling rate Rif which is x9 more than Rs, then filters with analog reconstruction filter.
        """
        # coupling
        # gain error
        # offset
        raise NotImplementedError
    
    def transmitBurst(self, burstbitSequence:NDArray[uint8], burstGuardRampPeriods:List, 
                      subslot2RampPeriods:List | None = None, debugReturnStage:str="baseband") -> Tuple[NDArray, NDArray]:
        """
        """
        if debugReturnStage not in VALID_RETURN_STAGE_VALUES:
            raise RuntimeError(f"Passed debug return stage for transmitBurst of: {debugReturnStage} invalid, expected value in: {VALID_RETURN_STAGE_VALUES}")

        #1. Determine handling of input burstbitSequence

        nBlocks = 0
        halfSlotUsage = False
        nullSubslot = None

        if burstbitSequence.ndim == 2:
            # multiple slot bursts or subslot bursts passed
            if burstbitSequence.shape[1] == SUBSLOT_BIT_LENGTH:
                # Passed 2 sublots for a single burst, acceptable
                # If the blocks are only SUBLOT length, then we are transmitting a single full slot burst
                # Could be of form: [CB, empty], [LB, CB], [empty, CB], or [LB, empty]
                if burstbitSequence.shape[0] != 2:
                    raise ValueError(f"Passed {burstbitSequence.shape[0]} subslots to transmit, expected exactly 2 to handle subslot tx")
                else:
                    # Determine where/if there is a null subslot
                    if (burstbitSequence[0] == 0).all():
                        nullSubslot = 0
                        nBlocks = 1
                    elif (burstbitSequence[1] == 0).all():
                        nullSubslot = 1
                        nBlocks = 1
                    else:
                        # There are no empty subslots
                        nBlocks = 2
                    
                    halfSlotUsage = True
            else:
                raise ValueError(f"Passed {burstbitSequence.shape[1]} modulation bits, expected 2 subslots of length {(SUBSLOT_BIT_LENGTH)}")
        
        elif burstbitSequence.ndim == 1 and burstbitSequence.size == (2*SUBSLOT_BIT_LENGTH):
            # Passed only one full slot burst, acceptable
            nBlocks = 1
        else:
            raise ValueError(f"Passed burstbitSequences of shape: {burstbitSequence.shape}, invalid number of dimensions or invalid number of modulation bits")

        # Allocate an output array for the burst data
        outputBBSignal = empty(shape=(1, (TIMESLOT_SYMBOL_LENGTH * BASEBAND_SAMPLING_FACTOR * TETRA_SYMBOL_RATE)), dtype=complex64)

        #2. Check if half slot
        if halfSlotUsage:
            # Single burst made from 1 or 2 subslots, need to add prepend or postpend extra modulation bit to ensure even number of bits
            # In the case of subslot, we need to manage the different cases, but generaly we modulate and ramp each half slot individually
            # However, both subslots share 1 common symbol overlapping the end of SUB1 and start of SUB2, 
            # but we know because of ramping this will always be zero so it is not the biggest deal to stitch together

            #3. Determine the state of the phase reference usage, here it will always be the default case
            burstPhaseReference = complex64(1 + 0j)
            if nBlocks == 2 and subslot2RampPeriods is not None:
                # we have two subslot bursts to modulate indepedently. nut 
                burstGuardRampPeriods[1] += 1 # must increment the end guard period of the first subslot burst to account for the need for the additional modulation bit
                subslot2RampPeriods[1] += 1   # must increment the end guard period of the first subslot burst to account for the need for the additional modulation bit

                #4. Modulate the burst bits into 255 symbols
                sbs1ComplexSymbolsN = dqpskModulator(concatenate((burstbitSequence[0], zeros(1, dtype=uint8))), burstGuardRampPeriods, burstPhaseReference)
                sbs2ComplexSymbolsN = dqpskModulator(concatenate((burstbitSequence[1], zeros(1, dtype=uint8))), subslot2RampPeriods, burstPhaseReference)

                #5. Pass modulated symbols and guard information to baseband processing function
                Isbs1, Qsbs1 = self._basebandProcessing(sbs1ComplexSymbolsN, burstGuardRampPeriods)
                Isbs2, Qsbs2 = self._basebandProcessing(sbs2ComplexSymbolsN, subslot2RampPeriods)

                # concatenate the two subslot, indexing such that the extra tail modulation bits are eliminated
                # is in subslot 1 and the other 32/64 is in subslot 2
                assertTailGoesToZero(Isbs1, Qsbs1, HALF_BASEBAND_SAMPLING_FACTOR)
                assertTailGoesToZero(Isbs2, Qsbs2, HALF_BASEBAND_SAMPLING_FACTOR)

                Itempramp = concatenate((Isbs1[:Isbs1.size - HALF_BASEBAND_SAMPLING_FACTOR], Isbs2[:Isbs2.size - HALF_BASEBAND_SAMPLING_FACTOR]))
                Qtempramp = concatenate((Qsbs1[:Qsbs1.size - HALF_BASEBAND_SAMPLING_FACTOR], Qsbs2[:Qsbs2.size - HALF_BASEBAND_SAMPLING_FACTOR]))

            elif nullSubslot == 0 and subslot2RampPeriods is not None:
                # First subslot is null burst, while second subslot is real burst
                subslot2RampPeriods[1] += 1
                sbs2ComplexSymbolsN = dqpskModulator(concatenate((burstbitSequence[1], zeros(1, dtype=uint8))), subslot2RampPeriods, burstPhaseReference)
                Isbs2, Qsbs2 = self._basebandProcessing(sbs2ComplexSymbolsN, subslot2RampPeriods)

                # generate equivalent number of zeros to fill null sublot
                Isbs1 = zeros((SUBSLOT_BIT_LENGTH+1)*64, dtype=(int64 if type(self) is realTransmitter else float32))
                Qsbs1 = zeros((SUBSLOT_BIT_LENGTH+1)*64, dtype=(int64 if type(self) is realTransmitter else float32))

                assertTailGoesToZero(Isbs2, Qsbs2, HALF_BASEBAND_SAMPLING_FACTOR)

                Itempramp = concatenate((Isbs1[:Isbs1.size - HALF_BASEBAND_SAMPLING_FACTOR], Isbs2[:Isbs2.size - HALF_BASEBAND_SAMPLING_FACTOR]))
                Qtempramp = concatenate((Qsbs1[:Qsbs1.size - HALF_BASEBAND_SAMPLING_FACTOR], Qsbs2[:Qsbs2.size - HALF_BASEBAND_SAMPLING_FACTOR]))

            elif nullSubslot == 1:
                # Second subslot is null burst, while first subslot is real burst
                # must index the end guard period of the real burst to account for the need for an additional modulation bit
                burstGuardRampPeriods[1] += 1
                sbs1ComplexSymbolsN = dqpskModulator(concatenate((burstbitSequence[0], zeros(1, dtype=uint8))), burstGuardRampPeriods, burstPhaseReference)
                Isbs1, Qsbs1 = self._basebandProcessing(sbs1ComplexSymbolsN, burstGuardRampPeriods)

                # generate equivalent number of zeros to fill null sublot
                Isbs2 = zeros((SUBSLOT_BIT_LENGTH+1)*64, dtype=(int64 if type(self) is realTransmitter else float32))
                Qsbs2 = zeros((SUBSLOT_BIT_LENGTH+1)*64, dtype=(int64 if type(self) is realTransmitter else float32))

                assertTailGoesToZero(Isbs1, Qsbs1, HALF_BASEBAND_SAMPLING_FACTOR)

                Itempramp = concatenate((Isbs1[:Isbs1.size - HALF_BASEBAND_SAMPLING_FACTOR], Isbs2[:Isbs2.size - HALF_BASEBAND_SAMPLING_FACTOR]))
                Qtempramp = concatenate((Qsbs1[:Qsbs1.size - HALF_BASEBAND_SAMPLING_FACTOR], Qsbs2[:Qsbs2.size - HALF_BASEBAND_SAMPLING_FACTOR]))

            else:
                raise ValueError(f"subslot2RampPeriod cannot be None, using the second subslot it is expected to represent the bit interval if odd")

        else:
            # Full slot burst
            # if we are not ramping up at the start of the burst, then we can assume it is continous with a previous burst and use the the internal phase reference state
            burstRampUpDownState = (burstGuardRampPeriods[0] != 0, burstGuardRampPeriods[1] != 0)
            burstPhaseReference = complex64(1 + 0j) if burstRampUpDownState[0] else self.phaseReference
            
            #4. Modulate the burst bits into 255 symbols
            inputComplexSymbolsN = dqpskModulator(burstbitSequence, burstGuardRampPeriods, burstPhaseReference)
            # if we ramp down at the end, reset the phase reference for the next burst, if we don't ramp down then it is assume phase continous and we set the reference to the last symbol phase of the burst
            self.phaseReference = inputComplexSymbolsN[-1] if not burstRampUpDownState[1] else complex64(1 + 0j)

            #5. Pass modulated symbols and guard information to baseband processing function
            Itempramp, Qtempramp = self._basebandProcessing(inputComplexSymbolsN, burstGuardRampPeriods)
        
        if debugReturnStage == VALID_RETURN_STAGE_VALUES[0]:
            return Itempramp, Qtempramp


        #6. Convert to DAC representation and perform ZOH at higher sampling rate
        
        return Itempramp, Qtempramp


###################################################################################################

class realTransmitter(Transmitter):


    rrcFilterState = zeros(shape=(2,len(RRC_Q1_17_COEFFICIENTS)-1), dtype=int64)
    lpfFilterState = zeros(shape=(2,len(FIR_LPF_Q1_17_COEFFICIENTS)-1), dtype=int64)
    halfband1State = zeros(shape=(2,len(FIR_HALFBAND1_Q1_17_COEFFICIENTS)-1), dtype=int64)
    halfband2State = zeros(shape=(2,len(FIR_HALFBAND2_Q1_17_COEFFICIENTS)-1), dtype=int64)
    halfband3State = zeros(shape=(2,len(FIR_HALFBAND3_Q1_17_COEFFICIENTS)-1), dtype=int64)
    
    def _basebandProcessing(self, inputComplexSymbols:NDArray[complex64], burstGuardRampPeriods:List) -> Tuple[NDArray[int64], NDArray[int64]]:
        """
        Converts modulation bits into symbols, performs upsampling, ramping, and filtering using quantized data
        """
        #1. Determine if prepending and/or postpending zeros to flush is required
        sOffset = 0
        eOffset = 0
        #1a. Continous with previous burst consideration:
        if burstGuardRampPeriods[0] != 0:
            # Since we ramp up, we are not continous with previous data, and must flush the FIRs with prepended zeros
            processedInputData = concatenate((full(shape=BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT, fill_value=complex64(1 + 0j), dtype=complex64), inputComplexSymbols))
            sOffset = BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT
        else:
            processedInputData = inputComplexSymbols.copy()
        
        #1b. Continous with subsequent burst consideration:
        if burstGuardRampPeriods[1] != 0:
            # Since we ramp down at the end, we are not continous afterwards and should flush data with postpended zeros
            processedInputData = concatenate((processedInputData, full(shape=BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT, fill_value=complex64(1 + 0j), dtype=complex64)))
            eOffset = BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT

        i_temp = processedInputData.real.astype(int64, copy=True)
        q_temp = processedInputData.imag.astype(int64, copy=True)

        mag = sqrt(i_temp[(sOffset):len(i_temp)-(eOffset)].astype(float64)**2 + q_temp[(sOffset):len(q_temp)-(eOffset)].astype(float64)**2)
        print("Pre x8 upsampling - symbol mapper ", "peakFS: ", mag.max()/(1), "rmsFS: ", sqrt(mean(mag**2))/(1))

        #2. Upsample by x8 with zero insertions, and quantize data to Q1.17 fixed format stored in float32
        upSampledInputData = oversampleDataQuantized(processedInputData, 8)


        #3. Perform RRC filtering
        I_stage1Symbols = upSampledInputData[0].copy()
        Q_stage1Symbols = upSampledInputData[1].copy()
       
        I_stage2Symbols, self.rrcFilterState[0] = _dspBlockFIRQuantizedStream(I_stage1Symbols, RRC_Q1_17_COEFFICIENTS, self.rrcFilterState[0], gain=4)
        Q_stage2Symbols, self.rrcFilterState[1] = _dspBlockFIRQuantizedStream(Q_stage1Symbols, RRC_Q1_17_COEFFICIENTS, self.rrcFilterState[1], gain=4)

        mag = sqrt(I_stage2Symbols[(sOffset*8):len(I_stage2Symbols)-(eOffset*8)].astype(float64)**2 + Q_stage2Symbols[(sOffset*8):len(Q_stage2Symbols)-(eOffset*8)].astype(float64)**2)
        print("Post RRC ", "peakFS: ", mag.max()/(1<<17), "rmsFS: ", sqrt(mean(mag**2))/(1<<17))

        #4. Perform cleanup LPF'ing
        I_stage3Symbols, self.lpfFilterState[0] = _dspBlockFIRQuantizedStream(I_stage2Symbols, FIR_LPF_Q1_17_COEFFICIENTS, self.lpfFilterState[0])
        Q_stage3Symbols, self.lpfFilterState[1] = _dspBlockFIRQuantizedStream(Q_stage2Symbols, FIR_LPF_Q1_17_COEFFICIENTS, self.lpfFilterState[1])

        mag = sqrt(I_stage3Symbols[(sOffset*8):len(I_stage3Symbols)-(eOffset*8)].astype(float64)**2 + Q_stage3Symbols[(sOffset*8):len(Q_stage3Symbols)-(eOffset*8)].astype(float64)**2)
        print("Post Cleanup ", "peakFS: ", mag.max()/(1<<17), "rmsFS: ", sqrt(mean(mag**2))/(1<<17))

        #5. Perform x2 upsampling with zero insertions and filter - Part 1
        I_stage4Symbols = zeros(shape=(2*I_stage3Symbols.size), dtype=int64)
        I_stage4Symbols[::2] = I_stage3Symbols
        I_stage5Symbols, self.halfband1State[0] = _dspBlockFIRQuantizedStream(I_stage4Symbols, FIR_HALFBAND1_Q1_17_COEFFICIENTS, self.halfband1State[0], gain=2)

        Q_stage4Symbols = zeros(shape=(2*Q_stage3Symbols.size), dtype=int64)
        Q_stage4Symbols[::2] = Q_stage3Symbols
        Q_stage5Symbols, self.halfband1State[1] = _dspBlockFIRQuantizedStream(Q_stage4Symbols, FIR_HALFBAND1_Q1_17_COEFFICIENTS, self.halfband1State[1], gain=2)

        mag = sqrt(I_stage5Symbols[(sOffset*16):len(I_stage5Symbols)-(eOffset*16)].astype(float64)**2 + Q_stage5Symbols[(sOffset*16):len(Q_stage5Symbols)-(eOffset*16)].astype(float64)**2)
        print("Post halfband-1 ", "peakFS: ", mag.max()/(1<<17), "rmsFS: ", sqrt(mean(mag**2))/(1<<17))
        
        #6. Perform x2 upsampling with zero insertions and filter - Part 2
        I_stage6Symbols = zeros(shape=(2*I_stage5Symbols.size), dtype=int64)
        I_stage6Symbols[::2] = I_stage5Symbols
        I_stage7Symbols, self.halfband2State[0] = _dspBlockFIRQuantizedStream(I_stage6Symbols, FIR_HALFBAND2_Q1_17_COEFFICIENTS, self.halfband2State[0], gain=2)

        Q_stage6Symbols = zeros(shape=(2*Q_stage5Symbols.size), dtype=int64)
        Q_stage6Symbols[::2] = Q_stage5Symbols
        Q_stage7Symbols, self.halfband2State[1] = _dspBlockFIRQuantizedStream(Q_stage6Symbols, FIR_HALFBAND2_Q1_17_COEFFICIENTS, self.halfband2State[1], gain=2)
        
        mag = sqrt(I_stage7Symbols[(sOffset*32):len(I_stage7Symbols)-(eOffset*32)].astype(float64)**2 + Q_stage7Symbols[(sOffset*32):len(Q_stage7Symbols)-(eOffset*32)].astype(float64)**2)
        print("Post halfband-2 ", "peakFS: ", mag.max()/(1<<17), "rmsFS: ", sqrt(mean(mag**2))/(1<<17))
        
        #7. Perform x2 upsampling with zero insertions and filter - Part 3
        I_stage8Symbols = zeros(shape=(2*I_stage7Symbols.size), dtype=int64)
        I_stage8Symbols[::2] = I_stage7Symbols
        I_stage9Symbols, self.halfband3State[0] = _dspBlockFIRQuantizedStream(I_stage8Symbols, FIR_HALFBAND3_Q1_17_COEFFICIENTS, self.halfband3State[0], gain=2)


        Q_stage8Symbols = zeros(shape=(2*Q_stage7Symbols.size), dtype=int64)
        Q_stage8Symbols[::2] = Q_stage7Symbols
        Q_stage9Symbols, self.halfband3State[1] = _dspBlockFIRQuantizedStream(Q_stage8Symbols, FIR_HALFBAND3_Q1_17_COEFFICIENTS, self.halfband3State[1], gain=2)

        mag = sqrt(I_stage9Symbols[(sOffset*64):len(I_stage9Symbols)-(eOffset*64)].astype(float64)**2 + Q_stage9Symbols[(sOffset*64):len(Q_stage9Symbols)-(eOffset*64)].astype(float64)**2)
        print("Post halfband-3 ", "peakFS: ", mag.max()/(1<<17), "rmsFS: ", sqrt(mean(mag**2))/(1<<17))

        #8. Extract useful part of burst
        if burstGuardRampPeriods[0] != 0:
            I_outputSymbols = I_stage9Symbols[(BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT*BASEBAND_SAMPLING_FACTOR):].copy()
            Q_outputSymbols = Q_stage9Symbols[(BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT*BASEBAND_SAMPLING_FACTOR):].copy()
        else:
            I_outputSymbols = I_stage9Symbols.copy()
            Q_outputSymbols = Q_stage9Symbols.copy()
        
        if burstGuardRampPeriods[1] != 0:
            I_outputSymbols = I_outputSymbols[:-(BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT*BASEBAND_SAMPLING_FACTOR)].copy()
            Q_outputSymbols = Q_outputSymbols[:-(BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT*BASEBAND_SAMPLING_FACTOR)].copy()

        #9. Perform ramping on signal

        I_rampedSymbols, Q_rampedSymbols = _powerRampingQuantized(I_outputSymbols, Q_outputSymbols, burstGuardRampPeriods)

        mag = sqrt(I_rampedSymbols.astype(float64)**2 + Q_rampedSymbols.astype(float64)**2)
        print("Post ramping ", "peakFS: ", mag.max()/(1<<17), "rmsFS: ", sqrt(mean(mag**2))/(1<<17))


        return I_rampedSymbols, Q_rampedSymbols


    def _dacConversion(self):
        """
        Converts baseband processed data into dac code representation, 
        """
        pass

    def _analogReconstruction(self):
        """
        Takes in DAC codes at rate Rs, converts to real floats with ZOH with 
        sampling rate Rif which is x8 more than Rs, then filters with analog reconstruction filter.
        """
        # coupling
        # gain error
        # offset
        pass
    


###################################################################################################

class idealTransmitter(Transmitter):
    rrcFilterState = zeros(shape=(2,len(RRC_FLOAT_COEFFICIENTS)-1), dtype=float32)
    lpfFilterState = zeros(shape=(2,len(FIR_LPF_FLOAT_COEFFICIENTS)-1), dtype=float32)
    halfband1State = zeros(shape=(2,len(FIR_HALFBAND1_FLOAT_COEFFICIENTS)-1), dtype=float32)
    halfband2State = zeros(shape=(2,len(FIR_HALFBAND2_FLOAT_COEFFICIENTS)-1), dtype=float32)
    halfband3State = zeros(shape=(2,len(FIR_HALFBAND3_FLOAT_COEFFICIENTS)-1), dtype=float32)
    
    def _basebandProcessing(self, inputComplexSymbols:NDArray[complex64], burstGuardRampPeriods:List) -> Tuple[NDArray[float32], NDArray[float32]]:
        """
        Converts modulation bits into symbols, performs upsampling, ramping, and filtering using float data
        """
        #1. Determine if prepending and/or postpending zeros to flush is required

        #1a. Continous with previous burst consideration:
        if burstGuardRampPeriods[0] != 0:
            # Since we ramp up, we are not continous with previous data, and must flush the FIRs with prepended zeros
            processedInputData = concatenate((full(shape=BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT, fill_value=complex64(1 + 0j), dtype=complex64), inputComplexSymbols))
        else:
            processedInputData = inputComplexSymbols.copy()
        
        #1b. Continous with subsequent burst consideration:
        if burstGuardRampPeriods[1] != 0:
            # Since we ramp down at the end, we are not continous afterwards and should flush data with postpended zeros
            processedInputData = concatenate((processedInputData, full(shape=BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT, fill_value=complex64(1 + 0j), dtype=complex64)))

        #2. Upsample by x8 with zero insertions, and quantize data to Q1.17 fixed format stored in float32
        upSampledInputData = oversampleDataFloat(processedInputData, 8)

        #3. Perform RRC filtering
        I_stage1Symbols = upSampledInputData[0].copy()
        Q_stage1Symbols = upSampledInputData[1].copy()

        I_stage2Symbols, self.rrcFilterState[0] = _dspBlockFIRFloatStream(I_stage1Symbols, RRC_FLOAT_COEFFICIENTS, self.rrcFilterState[0], gain=4)
        Q_stage2Symbols, self.rrcFilterState[1] = _dspBlockFIRFloatStream(Q_stage1Symbols, RRC_FLOAT_COEFFICIENTS, self.rrcFilterState[1], gain=4)

        #4. Perform cleanup LPF'ing
        I_stage3Symbols, self.lpfFilterState[0] = _dspBlockFIRFloatStream(I_stage2Symbols, FIR_LPF_FLOAT_COEFFICIENTS, self.lpfFilterState[0])
        Q_stage3Symbols, self.lpfFilterState[1] = _dspBlockFIRFloatStream(Q_stage2Symbols, FIR_LPF_FLOAT_COEFFICIENTS, self.lpfFilterState[1])

        #5. Perform x2 upsampling with zero insertions and filter - Part 1
        I_stage4Symbols = zeros(shape=(2*I_stage3Symbols.size), dtype=float32)
        I_stage4Symbols[::2] = I_stage3Symbols
        I_stage5Symbols = zeros(shape=(2*I_stage4Symbols.size), dtype=float32)
        I_stage5Symbols[::2], self.halfband1State[0] = _dspBlockFIRFloatStream(I_stage4Symbols, FIR_HALFBAND1_FLOAT_COEFFICIENTS, self.halfband1State[0], gain=2)

        Q_stage4Symbols = zeros(shape=(2*Q_stage3Symbols.size), dtype=float32)
        Q_stage4Symbols[::2] = Q_stage3Symbols
        Q_stage5Symbols = zeros(shape=(2*Q_stage4Symbols.size), dtype=float32)
        Q_stage5Symbols[::2], self.halfband1State[1] = _dspBlockFIRFloatStream(Q_stage4Symbols, FIR_HALFBAND1_FLOAT_COEFFICIENTS, self.halfband1State[1], gain=2)

        #6. Perform x2 upsampling with zero insertions and filter - Part 2
        I_stage6Symbols = zeros(shape=(2*I_stage5Symbols.size), dtype=float32)
        I_stage6Symbols[::2], self.halfband2State[0] = _dspBlockFIRFloatStream(I_stage5Symbols, FIR_HALFBAND2_FLOAT_COEFFICIENTS, self.halfband2State[0], gain=2)

        Q_stage6Symbols = zeros(shape=(2*Q_stage5Symbols.size), dtype=float32)
        Q_stage6Symbols[::2], self.halfband2State[1] = _dspBlockFIRFloatStream(Q_stage5Symbols, FIR_HALFBAND2_FLOAT_COEFFICIENTS, self.halfband2State[1], gain=2)
        
        
        #7. Perform x2 upsampling with zero insertions and filter - Part 3
        I_stage7Symbols= zeros(shape=(I_stage6Symbols.size), dtype=float32)
        I_stage7Symbols, self.halfband3State[0] = _dspBlockFIRFloatStream(I_stage6Symbols, FIR_HALFBAND3_FLOAT_COEFFICIENTS, self.halfband3State[0], gain=2)

        Q_stage7Symbols = zeros(shape=(Q_stage6Symbols.size), dtype=float32)
        Q_stage7Symbols, self.halfband3State[1] = _dspBlockFIRFloatStream(Q_stage6Symbols, FIR_HALFBAND3_FLOAT_COEFFICIENTS, self.halfband3State[1], gain=2)

        #8. Extract useful part of burst
        if burstGuardRampPeriods[0] != 0:
            I_outputSymbols = I_stage7Symbols[(BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT*BASEBAND_SAMPLING_FACTOR):].copy()
            Q_outputSymbols = Q_stage7Symbols[(BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT*BASEBAND_SAMPLING_FACTOR):].copy()
        else:
            I_outputSymbols = I_stage7Symbols.copy()
            Q_outputSymbols = Q_stage7Symbols.copy()
        
        if burstGuardRampPeriods[1] != 0:
            I_outputSymbols = I_outputSymbols[:-(BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT*BASEBAND_SAMPLING_FACTOR)].copy()
            Q_outputSymbols = Q_outputSymbols[:-(BASE_SAMPLE_RATE_ZERO_FIR_FLUSH_COUNT*BASEBAND_SAMPLING_FACTOR)].copy()


        #9. Perform ramping on signal

        I_rampedSymbols, Q_rampedSymbols = _powerRampingFloat(I_outputSymbols, Q_outputSymbols, burstGuardRampPeriods)

        return I_rampedSymbols, Q_rampedSymbols


    def _dacConversion(self):
        """
        Converts baseband processed data into dac code representation, 
        """
        pass

    def _analogReconstruction(self):
        """
        Takes in DAC codes at rate Rs, converts to real floats with ZOH with 
        sampling rate Rif which is x8 more than Rs, then filters with analog reconstruction filter.
        """
        # coupling
        # gain error
        # offset
        pass

###################################################################################################

