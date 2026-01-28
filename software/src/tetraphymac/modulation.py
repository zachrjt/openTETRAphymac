# ZT - 2026
# Based on EN 300 392-2 V2.4.2              
from numpy import cumprod, complex64, pi, float64, uint8, mod, array, abs, sum, exp, zeros
from numpy import int64, uint8, empty, concatenate, round, convolve, bitwise_right_shift, clip
from numpy.typing import NDArray
import scipy as sp
from abc import ABC, abstractmethod
from typing import ClassVar, Tuple, List
from .physical_channels import SUBSLOT_BIT_LENGTH, TIMESLOT_SYMBOL_LENGTH

DQPSK_PHASE_TRANSITION_LUT = array([pi/4, 3*pi/4, -pi/4, -3*pi/4], dtype=float64)
DQPSK_TRANSITION_PHASOR_LUT = exp(1j * DQPSK_PHASE_TRANSITION_LUT).astype(complex64)

GUARD_PERIOD_RAMP_BLANKING_SYMBOL_INTERVAL = 2
NUMBER_OF_FRACTIONAL_BITS = 17

BASEBAND_SAMPLING_FACTOR = 64
TETRA_SYMBOL_RATE = 18000

RRC_Q1_17_COEFFICIENTS = array([94, 42, 69, 77, 61, 26, -21, -66, -96, -100, -73, -21, 45, 108, 150, 157, 123, 52, -39, -128, 
                                -190, -206, -168, -81, 33, 144, 218, 226, 156, 19, -156, -319, -416, -399, -242, 47, 419, 794, 
                                1068, 1141, 934, 420, -361, -1297, -2210, -2881, -3084, -2625, -1385, 653, 3382, 6582, 9936, 
                                13080, 15648, 17328, 17912, 17328, 15648, 13080, 9936, 6582, 3382, 653, -1385, -2625, -3084, 
                                -2881, -2210, -1297, -361, 420, 934, 1141, 1068, 794, 419, 47, -242, -399, -416, -319, -156, 
                                19, 156, 226, 218, 144, 33, -81, -168, -206, -190, -128, -39, 52, 123, 157, 150, 108, 45, 
                                -21, -73, -100, -96, -66, -21, 26, 61, 77, 69, 42, 4], dtype=int64)

FIR_LPF_Q1_17_COEFFICIENTS = array([40, 136, 205, 218, 156, 27, -136, -283, -361, -329, -179, 57, 312, 503, 555, 429, 140, 
                                    -240, -595, -804, -776, -485, 9, 570, 1022, 1199, 1004, 447, -339, -1128, -1659, -1720, 
                                    -1216, -226, 1003, 2104, 2694, 2483, 1388, -411, -2476, -4191, -4904, -4088, -1501, 2723, 
                                    8064, 13701, 18671, 22078, 23290, 22078, 18671, 13701, 8064, 2723, -1501, -4088, -4904, 
                                    -4191, -2476, -411, 1388, 2483, 2694, 2104, 1003, -226, -1216, -1720, -1659, -1128, -339, 
                                    447, 1004, 1199, 1022, 570, 9, -485, -776, -804, -595, -240, 140, 429, 555, 503, 312, 57, 
                                    -179, -329, -361, -283, -136, 27, 156, 218, 205, 136, 40], dtype=int64)

FIR_HALFBAND1_Q1_17_COEFFICIENTS = array([-145, 0, 193, 0, -323, 0, 555, 0, -914, 0, 1434, 0, -2170, 0, 3221, 0, -4805, 0, 
                                          7491, 0, -13392, 0, 41587, 65606, 41587, 0, -13392, 0, 7491, 0, -4805, 0, 3221, 0, 
                                          -2170, 0, 1434, 0, -914, 0, 555, 0, -323, 0, 193, 0, -145], dtype=int64)

FIR_HALFBAND2_Q1_17_COEFFICIENTS = array([-304, 0, 711, 0, -2084, 0, 5063, 0, -11725, 0, 41035, 65681, 41035, 0, -11725, 0, 
                                          5063, 0, -2084, 0, 711, 0, -304], dtype=int64)

FIR_HALFBAND3_Q1_17_COEFFICIENTS = array([0, 1181, 0, -7504, 0, 39118, 65482, 39118, 0, -7504, 0, 1181, 0], dtype=int64)

FIR_TOTAL_NUMBER_OF_TAPS = len(RRC_Q1_17_COEFFICIENTS) + len(FIR_LPF_Q1_17_COEFFICIENTS) + len(FIR_HALFBAND1_Q1_17_COEFFICIENTS) + len(FIR_HALFBAND2_Q1_17_COEFFICIENTS) + len(FIR_HALFBAND3_Q1_17_COEFFICIENTS)

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

class Transmitter(ABC):
    phaseReference = complex64(1 + 0j)
    def __init__(self):
        pass

    @abstractmethod
    def _basebandProcessing(self):
        """
        Converts modulation bits into symbols, performs upsampling, ramping, and filtering
        """
        pass

    @abstractmethod
    def _dacConversion(self):
        """
        Converts baseband processed data into dac code representation, 
        """
        pass

    @abstractmethod
    def _analogReconstruction(self):
        """
        Takes in DAC codes at rate Rs, converts to real floats with ZOH with 
        sampling rate Rif which is x9 more than Rs, then filters with analog reconstruction filter.
        """
        # coupling
        # gain error
        # offset

        pass
    
    @abstractmethod
    def transmitBurst(self):
        """
        Wrapper that calls and links functions that perform conversion of modulation bits to real float data
        """
        pass

def _dspBlockFIRconvolve(inputSymbols:NDArray[int64], hCoef:NDArray[int64], flushLength:int, inputState: NDArray[int64] | None = None,  ):
    '''
    Handles the convolution process with continous state control, 
    but models the 54bit accumulation results and subsequent truncation and rounding down to Q1.17
    '''
    Ntaps = hCoef.size
    if inputState is None:
        state = zeros(Ntaps-1, dtype=int64)
    else:
        if inputState.size != (Ntaps - 1):
            raise ValueError(f"Length of FIR state passed is {inputState.size}, expected {(Ntaps - 1)} based on hCoef passed.")
        
    inputDataExt = concatenate((state, inputSymbols, zeros(shape=flushLength,dtype=int64)))

    outputAccumulated = convolve(inputDataExt, hCoef, mode="full")
    # Round to Infinity by adding 2^16 and right shifting by 17 bits
    outputAccumulated += int64(2 ** (NUMBER_OF_FRACTIONAL_BITS-1))
    outputAccumulated = bitwise_right_shift(outputAccumulated, NUMBER_OF_FRACTIONAL_BITS)

    # Clip if we are out of the range expected, this is not expected given the gains and such
    outputAccumulated = clip(outputAccumulated, -(1<<NUMBER_OF_FRACTIONAL_BITS), (1<<NUMBER_OF_FRACTIONAL_BITS)-1)

    return outputAccumulated



class realTransmitter(Transmitter):
    
    def _basebandProcessing(self, inputComplexSymbols:NDArray[complex64], burstGuardRampPeriods:List, rampUpandDown:Tuple[bool, bool]=(True, True)):
        """
        Converts modulation bits into symbols, performs upsampling, ramping, and filtering
        """
        #1. Upsample by x8 with zero insertions, and quantize data to Q1.17 fixed format stored in int64
        stage1Bits = concatenate((oversampleData(inputComplexSymbols, 8)))


        #2. Perform RRC filtering
        


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
    
    def transmitBurst(self, burstbitSequence:NDArray[uint8], burstGuardRampPeriods:List, rampUpandDown:Tuple[bool, bool]=(True, True)):
        """
        """
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
                        nullSubslot = [0]
                        nBlocks = 1
                    elif (burstbitSequence[1] == 0).all():
                        nullSubslot = [1]
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
            # TODO: Implement subslot tx handling
            raise NotImplementedError
        
        #3. Determine the state of the phase reference usage
        # if we are not ramping up at the start of the burst, then we can assume it is continous with a previous burst and use the the internal phase reference state
        burstRampUpDownState = rampUpandDown
        burstPhaseReference = complex64(1 + 0j) if burstRampUpDownState[0] else self.phaseReference
        
        #4. Modulate the burst bits into 255 symbols
        inputComplexSymbolsN = dqpskModulator(concatenate((burstbitSequence[0], burstbitSequence[1])) if halfSlotUsage else burstbitSequence, burstPhaseReference)
        # if we ramp down at the end, reset the phase reference for the next burst, if we don't ramp down then it is assume phase continous and we set the reference to the last symbol phase of the burst
        self.phaseReference = inputComplexSymbolsN[-1] if not burstRampUpDownState[1] else complex64(1 + 0j)

        #5. Pass modulated symbols and guard information to baseband processing function
        temp = self._basebandProcessing(inputComplexSymbolsN, burstGuardRampPeriods, burstRampUpDownState)
        

        #6. Perform RRC and cleanup filtering
        return None

class idealTransmitter(Transmitter):
    pass



def oversampleData(inputData: NDArray[complex64], overSampleRate:int) -> NDArray[int64]:
    outputData = zeros(shape=(2, inputData.size*overSampleRate), dtype=int64)
    # because the modulation mapped only on the unit circle, the max value is 1, and min is zero, 
    # the gain of the filters and processing is slightly under one, so we do not need to scale to prevent overflow in this case
    tempIvalues = round((inputData.real * (2 ** NUMBER_OF_FRACTIONAL_BITS)))
    tempQvalues = round((inputData.imag * (2 ** NUMBER_OF_FRACTIONAL_BITS)))
    outputData[0][0::overSampleRate] = tempIvalues.astype(int64)
    outputData[1][0::overSampleRate] = tempQvalues.astype(int64)
    return outputData