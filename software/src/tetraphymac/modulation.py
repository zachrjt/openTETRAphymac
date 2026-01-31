# ZT - 2026
# Based on EN 300 392-2 V2.4.2
from numpy import cumprod, complex64, pi, float64, uint8, mod, array, abs, sum, exp
from numpy import uint8, concatenate, full
from numpy.typing import NDArray
from typing import List

DQPSK_PHASE_TRANSITION_LUT = array([pi/4, 3*pi/4, -pi/4, -3*pi/4], dtype=float64)
DQPSK_TRANSITION_PHASOR_LUT = exp(1j * DQPSK_PHASE_TRANSITION_LUT).astype(complex64)

def calculatePhaseAdjustmentBits(inputData: NDArray[uint8], 
                                 inclusiveIndices:tuple[int, int],
                                 guardOffset:int=0) -> NDArray[uint8]:
    """
    Modulates input data within the inclusiveIndices accounting for a guardoffset, to determine the total 
    cummulative phase and returns the correct bit pair that would set the cumulative phase to zero
    
    :param inputData: Binary 1's and 0's input to evaluate total Pi/4-dqpsk phase angle over
    :type inputData: NDArray[uint8]
    :param inclusiveIndices: The start and stop indices (inclusive) to evaluate the cumulative phase
    :type inclusiveIndices: tuple[int, int]
    :param guardOffset: An offset to account for an initial guard period in input data, offsets inclusive indices
    :type guardOffset: int
    :return: Returns two bits that if added into the cumulative phase calculation results in the cumulative phase being zero
    :rtype: NDArray[uint8]
    """
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
                   burstGuardRampPeriods:List[int],
                   phaseRef: complex64 = complex64(1 + 0j)) -> NDArray[complex64]:
    
    if inputData.ndim != 1:
        raise ValueError(f"Input data dimensions are: {inputData.ndim}, expected 1")
    if inputData.size % 2 != 0: # must have even number of bits
        raise ValueError(f"Number of input burst bits is: {inputData.size}, expected even number")
    
    if burstGuardRampPeriods[0] % 2 != 0:
        raise ValueError(f"Start ramp guard period in number of bits is: {burstGuardRampPeriods[0]}, expected even number")
    if burstGuardRampPeriods[1] % 2 != 0:
        raise ValueError(f"End ramp guard period in number of bits is: {burstGuardRampPeriods[1]}, expected even number")

    Nstart = int(burstGuardRampPeriods[0] / 2) # in number of symbols
    Nend = int(burstGuardRampPeriods[1] / 2) # in number of symbols

    # The ramping period at the start and end shall have constant phase,
    # Therefore we exclude the ramping periods from the calculations,
    # Note that the start ramping phase is constant and equal to reference phase, 
    # while the final down ramp phase is constant and equal to whichever was the last phase in the useful part of the burst
    # This prevents phase discontinuity

    # reshape to have even and odd bits [b0, b1]
    usefulBurstSegment = inputData[(Nstart * 2) : inputData.size - (Nend * 2)]
    bitPairs = usefulBurstSegment.reshape(-1,2)
    # map the even odd bits into a 4-entry code to map phase transistion quickly from LUT
    codedTransistions = (bitPairs[:, 0] << 1 ) | bitPairs[:, 1] # maps into value [b0b1, b2b3, ...]
    
    # grayTransistion -> 0=0b00, 1=0b01, 2=0b10, 3=0b11, lookup the phase transistion table in phasor form
    dPhasor = DQPSK_TRANSITION_PHASOR_LUT[codedTransistions]
    
    burstSegment = (cumprod(dPhasor) * phaseRef).astype(complex64)

    # prepend and postpend the constant phase ramping periods if they are needed
    if Nstart > 0:
        startRampPhase = full(Nstart, phaseRef, dtype=complex64)
        burstSegment = concatenate((startRampPhase, burstSegment)).astype(complex64)

    if Nend > 0:
        endPhase = burstSegment[-1] if burstSegment.size > 0 else phaseRef
        endRampPhase = full(Nend, endPhase, dtype=complex64)
        burstSegment = concatenate((burstSegment, endRampPhase)).astype(complex64)

    return burstSegment