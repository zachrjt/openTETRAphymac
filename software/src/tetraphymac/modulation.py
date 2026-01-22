# ZT - 2026
# Based on EN 300 392-2 V2.4.2
NUM_STATES = 8                                 
ALPHA = 0.35                                          
RAISED_COSINE_FILTER_SPAN = 4
from numpy import cumprod, complex64, pi, float64, uint8, mod, array, abs, sum, exp, zeros, int16, float32
from numpy.typing import NDArray
import scipy as sp
from abc import ABC, abstractmethod

DQPSK_PHASE_TRANSITION_LUT = array([pi/4, 3*pi/4, -pi/4, -3*pi/4], dtype=float64)
DQPSK_TRANSITION_PHASOR_LUT = exp(1j * DQPSK_PHASE_TRANSITION_LUT).astype(complex64)


def calculatePhaseAdjustmentBits(inputData: NDArray[uint8], 
                                 inclusiveIndices:tuple[int, int],
                                 guardOffset:int=0) -> NDArray[uint8]:
    assert inputData.ndim == 1
    assert inputData.size % 2 == 0 # must have even number of bits

    # reshape to have even and odd bits [b0, b1], also skip the first guardOffset bits to get to data 
    bitPairs = inputData[guardOffset:]
    bitPairs = bitPairs[inclusiveIndices[0]*2:(inclusiveIndices[1]*2)+2].reshape(-1,2)

    # map the even odd bits into a 4-entry code to map phase transistion quickly from LUT
    codedTransistions = (bitPairs[:, 0] << 1 ) | bitPairs[:, 1] # maps into value [b0b1, b2b3, ...]
    
    # grayTransistion -> 0=0b00, 1=0b01, 2=0b10, 3=0b11, lookup the phase transistion from table
    dphi = DQPSK_PHASE_TRANSITION_LUT[codedTransistions] # if you get an out of range here, then your indices are off, 
                                                         # reading random (0-255) values in the unset phase adjustment bits

    phiAccumulated = float64(sum(dphi))

    resultantAngle = mod(-phiAccumulated, (2*pi))
    if resultantAngle > pi:
        resultantAngle -= (2*pi)
    
    resultantBitPair = abs(DQPSK_PHASE_TRANSITION_LUT - resultantAngle).argmin()

    bits = array([(resultantBitPair >> 1) & 1, resultantBitPair & 1],dtype=uint8)

    return bits

###################################################################################################

def dqpskModulator(inputData: NDArray[uint8], 
                   phaseRef: complex64 = complex64(1 + 0j)) -> NDArray[complex64]:
    
    assert inputData.ndim == 1
    assert inputData.size % 2 == 0 # must have even number of bits,

    # reshape to have even and odd bits [b0, b1]
    bitPairs = inputData.reshape(-1,2)
    # map the even odd bits into a 4-entry code to map phase transistion quickly from LUT
    codedTransistions = (bitPairs[:, 0] << 1 ) | bitPairs[:, 1] # maps into value [b0b1, b2b3, ...]
    
    # grayTransistion -> 0=0b00, 1=0b01, 2=0b10, 3=0b11, lookup the phase transistion table in phasor form
    dPhasor = DQPSK_TRANSITION_PHASOR_LUT[codedTransistions]
    
    return (cumprod(dPhasor) * phaseRef).astype(complex64)

###################################################################################################

def generateBurstIQData():
    # TODO: implement the burst sampling
    # 1. zero pad, oversample, manage guard period IQ data for ramping if enabled
    # 2. RRC filter 
    # 3. Apply ramping envelope as required to end and start


    raise NotImplementedError

# discrete transmitter
# quantizes the data to 10bits
# performs limited filtering to emulate actual Tx chain in terms of DSP slices
# converts quantized results into complex floats afterwards for reception
# continous transmitter
# does not quantize data
# filtering is ideal
# no I, Q impairments


class Transmitter(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def oversampleData(self):
        # 8x oversampling
        pass

    @abstractmethod
    def pulseShapeData(self):
        # RRC filter TBD design
        pass

    @abstractmethod
    def matchDACRate(self):
        # interpolate using 4x CIC
        # compensating FIR filter
        pass

    @abstractmethod
    def DAC(self):
        # convert digitzed data to continous
        # add in errors
        # result is complex 64
        pass




class discreteTransmitter(Transmitter):
    pass

class continousTransmitter(Transmitter):
    pass



def oversampleData(inputData: NDArray[complex64], overSampleRate:int):
    outputData = zeros(shape=(2, inputData.size*overSampleRate), dtype=int16)
    # because the modulation mapped only on the unit circle, the max value is 1, and min is zero
    # for this module, we quantize scaling by 0.5 then multiplying by our maximum value to provide headroom in later calculations
    tempIvalues = (inputData.real* ((2 ** 15) - 1)) / 2
    tempQvalues = (inputData.imag* ((2 ** 15) - 1)) / 2
    outputData[0][0::overSampleRate] = tempIvalues.astype(int16)
    outputData[1][0::overSampleRate] = tempQvalues.astype(int16)
    return outputData