# ZT - 2026
# Based on EN 300 392-2 V2.4.2
NUM_STATES = 8                                 
ALPHA = 0.35                                          
RAISED_COSINE_FILTER_SPAN = 4
from numpy import cumprod, complex64, pi, float64, uint8, mod, array, abs, sum, exp, ndarray
from numpy.typing import NDArray

DQPSK_PHASE_TRANSITION_LUT = array([pi/4, 3*pi/4, -pi/4, -3*pi/4], dtype=float64)
DQPSK_TRANSITION_PHASOR_LUT = exp(1j * DQPSK_PHASE_TRANSITION_LUT).astype(complex64)


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
    

def calculatePhaseAdjustmentBits(inputData: NDArray[uint8], 
                                 inclusiveIndices:tuple[int, int],
                                 guardOffset:int=0) -> NDArray[uint8]:
    assert inputData.ndim == 1
    assert inputData.size % 2 == 0 # must have even number of bits

    # reshape to have even and odd bits [b0, b1], also skip the first guardOffset bits to get to data 
    bitPairs = inputData[guardOffset:].reshape(-1,2)

    # map the even odd bits into a 4-entry code to map phase transistion quickly from LUT
    codedTransistions = (bitPairs[:, 0] << 1 ) | bitPairs[:, 1] # maps into value [b0b1, b2b3, ...]
    
    # grayTransistion -> 0=0b00, 1=0b01, 2=0b10, 3=0b11, lookup the phase transistion from table
    dphi = DQPSK_PHASE_TRANSITION_LUT[codedTransistions]

    phiAccumulated = float64(sum(dphi[inclusiveIndices[0]:inclusiveIndices[1]+1]))

    resultantAngle = mod(-phiAccumulated, (2*pi))
    if resultantAngle > pi:
        resultantAngle -= (2*pi)
    
    resultantBitPair = abs(DQPSK_PHASE_TRANSITION_LUT - resultantAngle).argmin()

    bits = array([(resultantBitPair >> 1) & 1, resultantBitPair & 1],dtype=uint8)

    return bits