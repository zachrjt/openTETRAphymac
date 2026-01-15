# ZT - 2026
# Based on EN 300 392-2 V2.4.2
from typing import List
from numpy.random import randint
from numpy import uint8, array, empty
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from .constants import SlotLength, ChannelKind, ChannelName
from .coding_scrambling import crc16Decoder, crc16Encoder, rcpcDecoder, rcpcEncoder, rm3014Decoder, rm3014Encoder, blockDeInterleaver
from .coding_scrambling import descrambler, nBlockInterleaver, blockInterleaver, nBlockDeInterleaver, scrambler

TAIL_BITS_ARRAY = array([0,0,0,0], uint8)

class LogicalChannel_VD(ABC):
    '''
    logical channels contain:
        1. a method to construct type-5 bits in type 5 blocks
        from type-1 information bits in type-1 blocks from MAC

        2. a method to construct type-1 bits in type 1 blocks from
        type-5 recieved bits in type-5 blocks from PHY
    '''

    # attributes contain the various types of bits, all are stored for now for analysis when needed
    type5Blocks: NDArray
    type4Blocks: NDArray
    type3Blocks: NDArray
    type2Blocks: NDArray
    type1Blocks: NDArray

    channel = ""      # Describes the written name of the logical channel
    channelType = ""  # Channel type either traffic or control

    M = 0             # Number of input blocks

    K1 = 0            # Number of input bits per block
    K5 = 0            # Number of output bits per block

    CRC = 0

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def encodeType5Bits(self):
        pass

    @abstractmethod
    def decodeType5Bits(self):
        pass

    def generateRndInput(self, M:int=1) -> NDArray[uint8]:
        return randint(0, 2, size=(M, self.K1), dtype=uint8)
    
    def validateKLength(self, K:int):
        '''
        Validate the number of bits in the K_ type blocks of the logical channel for K=1 or 5
        
        :param self: Description
        :param K: int={1,5}, either verifiy type1Blocks for K=1, or type5Blocks for K=5
        '''
        if self.M == 0:
            raise ValueError (f"M unspecified")
        if K == 5:
            if self.K5 == 0:
                raise ValueError (f"K5 unspecified")
            if self.type5Blocks is None:
                raise ValueError (f"type5Blocks unspecified")
            if self.type5Blocks.shape[1] != self.K5:
                raise ValueError (f"For logical channel {self.__class__.__name__}, expected K5: {self.K5} output bits, recieved {self.type5Blocks.shape}")
        elif K == 1:
            if self.K1 == 0:
                raise ValueError (f"K1 unspecified")
            if self.type1Blocks is None:
                raise ValueError (f"type1Blocks unspecified")
            if self.type1Blocks.shape[1] != self.K1:
                raise ValueError (f"For logical channel {self.__class__.__name__}, expected K1: {self.K1} output bits, recieved {self.type1Blocks.shape}")

###################################################################################################
# Control CHannels (CCH)
class ControlChannel(LogicalChannel_VD):

    def __init__(self):
        self.channelType = ChannelKind.CONTROL_TYPE
    

###################################################################################################
class BCCH(ControlChannel):
    '''
    Broadcast Control CHannel (BCCH)

    The BCCH shall be a uni-directional channel for common use by all MSs. It shall broadcast general information to all MSs
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)




class BNCH(BCCH):
    '''
    Broadcast Network CHannel (BNCH)

    down-link only, broadcasts network information to MSs.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.K1 = 124
        self.K5 = 216
        self.channel = ChannelName.BNCH_CHANNEL
    
    def encodeType5Bits(self, inputDataBlocks:NDArray[uint8]):
        self.M = inputDataBlocks.shape[0]
        if self.M != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}, only 1 data block is supported but {self.M} was passed")
        self.type1Blocks = inputDataBlocks
        self.validateKLength(1)

        self.type2Blocks = empty(shape=(self.M, 144), dtype=uint8)
        self.type2Blocks[0][:-4] = crc16Encoder(self.type1Blocks[0])
        self.type2Blocks[0][-4:] = TAIL_BITS_ARRAY

        self.type3Blocks = empty(shape=(self.M, 216), dtype=uint8)
        self.type3Blocks[0] = rcpcEncoder(self.type2Blocks[0],2,3)

        self.type4Blocks = empty(shape=(self.M, 216), dtype=uint8)
        self.type4Blocks[0] = blockInterleaver(self.type3Blocks[0],101)

        self.type5Blocks = empty(shape=(self.M, 216), dtype=uint8)
        self.type5Blocks[0] = scrambler(self.type4Blocks[0])
        self.validateKLength(5)
    
    def decodeType5Bits(self, inputDataBlocks:NDArray[uint8]):
        
        self.M = inputDataBlocks.shape[0]
        if self.M != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}, only 1 data block is supported but {self.M} was passed")
        self.type5Blocks = inputDataBlocks
        self.validateKLength(5)
        self.type4Blocks = empty(shape=(self.M, 216), dtype=uint8)
        self.type4Blocks[0] = descrambler(self.type5Blocks[0])

        self.type3Blocks = empty(shape=(self.M, 216), dtype=uint8)
        self.type3Blocks[0] = blockDeInterleaver(self.type4Blocks[0],101)

        self.type2Blocks = empty(shape=(self.M, 144), dtype=uint8)
        self.type2Blocks[0] = rcpcDecoder(self.type3Blocks[0],144,2,3)

        self.type1Blocks = empty(shape=(self.M, 124), dtype=uint8)
        self.type1Blocks[0], self.CRC = crc16Decoder(self.type2Blocks[0][:-4])
        self.validateKLength(1)

class BSCH(BCCH):
    '''
    Broadcast Synchronization Channel (BNCH)

    down-link only, broadcast information used for time and scrambling synchronization of the MSs
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.K1 = 60
        self.K5 = 120
        self.channel = ChannelName.BSCH_CHANNEL

    def encodeType5Bits(self, inputDataBlocks:NDArray[uint8]):
        self.M = inputDataBlocks.shape[0]
        if self.M != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}, only 1 data block is supported but {self.M} was passed")
        
        self.type1Blocks = inputDataBlocks
        self.validateKLength(1)

        self.type2Blocks = empty(shape=(self.M, 80), dtype=uint8)
        self.type2Blocks[0][:-4] = crc16Encoder(inputDataBlocks[0])
        self.type2Blocks[0][-4:] = TAIL_BITS_ARRAY

        self.type3Blocks = empty(shape=(self.M, 120), dtype=uint8)
        self.type3Blocks[0] = rcpcEncoder(self.type2Blocks[0],2,3)

        self.type4Blocks = empty(shape=(self.M, 120), dtype=uint8)
        self.type4Blocks[0] = blockInterleaver(self.type3Blocks[0],11)

        self.type5Blocks = empty(shape=(self.M, 120), dtype=uint8)
        self.type5Blocks[0] = scrambler(self.type4Blocks[0],BSCH=True)
        self.validateKLength(5)
    
    def decodeType5Bits(self, inputDataBlocks:NDArray[uint8]):
        self.M = inputDataBlocks.shape[0]
        if self.M != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}, only 1 data block is supported but {self.M} was passed")
        self.type5Blocks = inputDataBlocks
        self.validateKLength(5)

        self.type4Blocks = empty(shape=(self.M, 120), dtype=uint8)
        self.type4Blocks[0] = descrambler(self.type5Blocks[0],BSCH=True)

        self.type3Blocks = empty(shape=(self.M, 120), dtype=uint8)
        self.type3Blocks[0] = blockDeInterleaver(self.type4Blocks[0],11)

        self.type2Blocks = empty(shape=(self.M, 80), dtype=uint8)
        self.type2Blocks[0] = rcpcDecoder(self.type3Blocks[0],80,2,3)

        self.type1Blocks = empty(shape=(self.M, 60), dtype=uint8)
        self.type1Blocks[0], self.CRC = crc16Decoder(self.type2Blocks[0][:-4])
        self.validateKLength(1)

###################################################################################################

###################################################################################################

# note linearization channels are not included here since they really only need to be considered at burst level

class SCH(ControlChannel):
    '''
    Signalling CHannel (SCH)

    The SCH shall be shared by all MSs, but may carry messages specific to one or one group of MSs. System operation
    requires the establishment of at least one SCH per BS. SCH may be divided into 3 categories, depending on the size of
    the message:
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class SCH_F(SCH):
    '''
    Full size Signalling Channel (SCH/F)

    bidirectional channel used for full size messages.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.K1 = 268
        self.K5 = 432
        self.channel = ChannelName.SCH_F_CHANNEL

    def encodeType5Bits(self, inputDataBlocks:NDArray[uint8]):
        self.M = inputDataBlocks.shape[0]
        if self.M != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}, only 1 data block is supported but {self.M} was passed")
        self.type1Blocks = inputDataBlocks
        self.validateKLength(1)

        self.type2Blocks = empty(shape=(self.M, 288), dtype=uint8)
        self.type2Blocks[0][:-4] = crc16Encoder(self.type1Blocks[0])
        self.type2Blocks[0][-4:] = TAIL_BITS_ARRAY

        self.type3Blocks = empty(shape=(self.M, 432), dtype=uint8)
        self.type3Blocks[0] = rcpcEncoder(self.type2Blocks[0],2,3)

        self.type4Blocks = empty(shape=(self.M, 432), dtype=uint8)
        self.type4Blocks[0] = blockInterleaver(self.type3Blocks[0],103)

        self.type5Blocks = empty(shape=(self.M, 432), dtype=uint8)
        self.type5Blocks[0] = scrambler(self.type4Blocks[0])
        self.validateKLength(5)
    
    def decodeType5Bits(self, inputDataBlocks:NDArray[uint8]):
        self.M = inputDataBlocks.shape[0]
        if self.M != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}, only 1 data block is supported but {self.M} was passed")
        self.type5Blocks = inputDataBlocks
        self.validateKLength(5)

        self.type4Blocks = empty(shape=(self.M, 432), dtype=uint8)
        self.type4Blocks[0] = descrambler(self.type5Blocks[0])

        self.type3Blocks = empty(shape=(self.M, 432), dtype=uint8)
        self.type3Blocks[0] = blockDeInterleaver(self.type4Blocks[0],103)

        self.type2Blocks = empty(shape=(self.M, 288), dtype=uint8)
        self.type2Blocks[0] = rcpcDecoder(self.type3Blocks[0],288,2,3)

        self.type1Blocks = empty(shape=(self.M, 268), dtype=uint8)
        self.type1Blocks[0], self.CRC = crc16Decoder(self.type2Blocks[0][:-4])
        self.validateKLength(1)

class SCH_HD(SCH):
    '''
    Half size Downlink Signalling Channel (SCH/HD)

    downlink only, used for half size messages.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.K1 = 124
        self.K5 = 216
        self.channel = ChannelName.SCH_HD_CHANNEL
    
    def encodeType5Bits(self, inputDataBlocks:NDArray[uint8]):
        self.M = inputDataBlocks.shape[0]
        if self.M != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}, only 1 data block is supported but {self.M} was passed")
        self.type1Blocks = inputDataBlocks
        self.validateKLength(1)

        self.type2Blocks = empty(shape=(self.M, 144), dtype=uint8)
        self.type2Blocks[0][:-4] = crc16Encoder(self.type1Blocks[0])
        self.type2Blocks[0][-4:] = TAIL_BITS_ARRAY

        self.type3Blocks = empty(shape=(self.M, 216), dtype=uint8)
        self.type3Blocks[0] = rcpcEncoder(self.type2Blocks[0],2,3)

        self.type4Blocks = empty(shape=(self.M, 216), dtype=uint8)
        self.type4Blocks[0] = blockInterleaver(self.type3Blocks[0],101)

        self.type5Blocks = empty(shape=(self.M, 216), dtype=uint8)
        self.type5Blocks[0] = scrambler(self.type4Blocks[0])
        self.validateKLength(5)
    
    def decodeType5Bits(self, inputDataBlocks:NDArray[uint8]):
        
        self.M = inputDataBlocks.shape[0]
        if self.M != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}, only 1 data block is supported but {self.M} was passed")
        self.type5Blocks = inputDataBlocks
        self.validateKLength(5)
        self.type4Blocks = empty(shape=(self.M, 216), dtype=uint8)
        self.type4Blocks[0] = descrambler(self.type5Blocks[0])

        self.type3Blocks = empty(shape=(self.M, 216), dtype=uint8)
        self.type3Blocks[0] = blockDeInterleaver(self.type4Blocks[0],101)

        self.type2Blocks = empty(shape=(self.M, 144), dtype=uint8)
        self.type2Blocks[0] = rcpcDecoder(self.type3Blocks[0],144,2,3)

        self.type1Blocks = empty(shape=(self.M, 124), dtype=uint8)
        self.type1Blocks[0], self.CRC = crc16Decoder(self.type2Blocks[0][:-4])
        self.validateKLength(1)


class SCH_HU(SCH):
    '''
    Half size Uplink Signalling Channel (SCH/HU)

    uplink only, used for half size messages.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.K1 = 92
        self.K5 = 168
        self.channel = ChannelName.SCH_HU_CHANNEL
    
    def encodeType5Bits(self, inputDataBlocks:NDArray[uint8]):
        self.M = inputDataBlocks.shape[0]
        if self.M != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}, only 1 data block is supported but {self.M} was passed")
        self.type1Blocks = inputDataBlocks
        self.validateKLength(1)

        self.type2Blocks = empty(shape=(self.M, 112), dtype=uint8)
        self.type2Blocks[0][:-4] = crc16Encoder(inputDataBlocks[0]) 
        self.type2Blocks[0][-4:] = TAIL_BITS_ARRAY

        self.type3Blocks = empty(shape=(self.M, 168), dtype=uint8)
        self.type3Blocks[0] = rcpcEncoder(self.type2Blocks[0],2,3)

        self.type4Blocks = empty(shape=(self.M, 168), dtype=uint8)
        self.type4Blocks[0] = blockInterleaver(self.type3Blocks[0],13)

        self.type5Blocks = empty(shape=(self.M, 168), dtype=uint8)
        self.type5Blocks[0] = scrambler(self.type4Blocks[0])
        self.validateKLength(5)
    
    def decodeType5Bits(self, inputDataBlocks:NDArray[uint8]):
        self.M = inputDataBlocks.shape[0]
        if self.M != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}, only 1 data block is supported but {self.M} was passed")
        self.type5Blocks = inputDataBlocks
        self.validateKLength(5)

        self.type4Blocks = empty(shape=(self.M, 168), dtype=uint8)
        self.type4Blocks[0] = descrambler(self.type5Blocks[0])

        self.type3Blocks = empty(shape=(self.M, 168), dtype=uint8)
        self.type3Blocks[0] = blockDeInterleaver(self.type4Blocks[0],13)

        self.type2Blocks = empty(shape=(self.M, 112), dtype=uint8)
        self.type2Blocks[0] = rcpcDecoder(self.type3Blocks[0],112,2,3)

        self.type1Blocks = empty(shape=(self.M, 92), dtype=uint8)
        self.type1Blocks[0], self.CRC = crc16Decoder(self.type2Blocks[0][:-4])
        self.validateKLength(1)


class AACH(ControlChannel):
    '''
    Access Assignment CHannel (AACH)

    The AACH shall be present on all transmitted downlink slots. It shall be used to indicate on each physical channel the
    assignment of the uplink and downlink slots. The AACH shall be internal to the MAC.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.K1 = 14
        self.K5 = 30
        self.channel = ChannelName.AACH_CHANNEL

    def encodeType5Bits(self, inputDataBlocks:NDArray[uint8]):
        self.M = inputDataBlocks.shape[0]
        self.type1Blocks = inputDataBlocks
        self.validateKLength(1)
        if self.M != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}, only 1 data block is supported but {self.M} was passed")
        
        self.type2Blocks = empty(shape=(self.M, 30), dtype=uint8)
        self.type2Blocks[0] = rm3014Encoder(self.type1Blocks[0])

        self.type5Blocks = empty(shape=(self.M, 30), dtype=uint8)
        self.type5Blocks[0] = scrambler(self.type2Blocks[0], False)
        self.validateKLength(5)
    
    def decodeType5Bits(self, inputDataBlocks:NDArray[uint8]):
        self.M = inputDataBlocks.shape[0]
        if self.M != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}, only 1 data block is supported but {self.M} was passed")
        self.type5Blocks = inputDataBlocks
        self.validateKLength(5)
        self.type2Blocks = empty(shape=(self.M, 30), dtype=uint8)
        self.type2Blocks[0] = descrambler(self.type5Blocks[0])
        
        self.type1Blocks = empty(shape=(self.M, 14), dtype=uint8)
        self.type1Blocks[0] = rm3014Decoder(self.type2Blocks[0])
        self.validateKLength(1)

class STCH(ControlChannel):
    '''
    STealing CHannel (STCH)

    The STCH is a channel associated to a TCH that temporarily "steals" a part of the associated TCH capacity to transmit
    control messages. It may be used when fast signalling is required. In half duplex mode the STCH is unidirectional and
    has the same direction as the associated TCH.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.K1 = 124
        self.K5 = 216
        self.channel = ChannelName.STCH_CHANNEL

    def encodeType5Bits(self, inputDataBlocks:NDArray[uint8]):
        self.M = inputDataBlocks.shape[0]
        if self.M != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}, only 1 data block is supported but {self.M} was passed")
        self.type1Blocks = inputDataBlocks
        self.validateKLength(1)

        self.type2Blocks = empty(shape=(self.M, 144), dtype=uint8)
        self.type2Blocks[0][:-4] = crc16Encoder(self.type1Blocks[0])
        self.type2Blocks[0][-4:] = TAIL_BITS_ARRAY

        self.type3Blocks = empty(shape=(self.M, 216), dtype=uint8)
        self.type3Blocks[0] = rcpcEncoder(self.type2Blocks[0],2,3)

        self.type4Blocks = empty(shape=(self.M, 216), dtype=uint8)
        self.type4Blocks[0] = blockInterleaver(self.type3Blocks[0],101)

        self.type5Blocks = empty(shape=(self.M, 216), dtype=uint8)
        self.type5Blocks[0] = scrambler(self.type4Blocks[0])
        self.validateKLength(5)
    
    def decodeType5Bits(self, inputDataBlocks:NDArray[uint8]):
        
        self.M = inputDataBlocks.shape[0]
        if self.M != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}, only 1 data block is supported but {self.M} was passed")
        self.type5Blocks = inputDataBlocks
        self.validateKLength(5)
        self.type4Blocks = empty(shape=(self.M, 216), dtype=uint8)
        self.type4Blocks[0] = descrambler(self.type5Blocks[0])

        self.type3Blocks = empty(shape=(self.M, 216), dtype=uint8)
        self.type3Blocks[0] = blockDeInterleaver(self.type4Blocks[0],101)

        self.type2Blocks = empty(shape=(self.M, 144), dtype=uint8)
        self.type2Blocks[0] = rcpcDecoder(self.type3Blocks[0],144,2,3)

        self.type1Blocks = empty(shape=(self.M, 124), dtype=uint8)
        self.type1Blocks[0], self.CRC = crc16Decoder(self.type2Blocks[0][:-4])
        self.validateKLength(1)

###################################################################################################
# Traffic Channels

class TrafficChannel(LogicalChannel_VD):
    N = 1
    def __init__(self, N:int=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channelType = ChannelKind.TRAFFIC_TYPE
        self.N = N
        self.K5 = 432
        if self.N not in [1,2,4,8]:
            raise ValueError (f"The passed N - interleaving value of {self.N} is not valid.")

    
class TCH_S(TrafficChannel):
    '''
    Speech Traffic Channel (TCH/S)

    The traffic channels shall carry user information, defined for speech.
    '''
    slotLength = ""

    def __init__(self, slotLength:str=SlotLength.FULL_SUBSLOT, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slotLength = slotLength
        if self.slotLength != SlotLength.HALF_SUBSLOT and self.slotLength != SlotLength.FULL_SUBSLOT:
            raise ValueError (f"The passed slot length value of {self.slotLength} is not of: 'half' or 'full' ")
        if self.N not in [1]:
            raise ValueError (f"The passed N - interleaving value of {self.N} is not valid for {self.__class__.__name__}")
        self.K1 = 432 if self.slotLength == SlotLength.FULL_SUBSLOT else 216
        self.K5 = 432 if self.slotLength == SlotLength.FULL_SUBSLOT else 216
        self.channel = ChannelName.TCH_S_CHANNEL


    def encodeType5Bits(self, inputDataBlocks:NDArray[uint8]):
        self.M = inputDataBlocks.shape[0]
        self.type1Blocks = inputDataBlocks.copy()
        self.validateKLength(1)
        if self.slotLength == SlotLength.FULL_SUBSLOT:
            self.type4Blocks = inputDataBlocks
            self.type5Blocks = empty(shape=(self.M, 432), dtype=uint8)
            for i in range(self.M):
                self.type5Blocks[i] = scrambler(self.type4Blocks[i])

        elif self.slotLength == SlotLength.HALF_SUBSLOT:
            self.type3Blocks = inputDataBlocks
            self.type4Blocks = empty(shape=(self.M, 216), dtype=uint8)
            self.type5Blocks = empty(shape=(self.M, 216), dtype=uint8)
            for i in range(self.M):
                self.type4Blocks[i] = blockInterleaver(self.type3Blocks[i],101)
                self.type5Blocks[i] = scrambler(self.type4Blocks[i])

        self.validateKLength(5)
    
    def decodeType5Bits(self, inputDataBlocks:NDArray[uint8]):
        self.M = inputDataBlocks.shape[0]
        self.type5Blocks = inputDataBlocks.copy()
        self.validateKLength(5)
        self.type4Blocks = empty(shape=(self.M, self.K5), dtype=uint8) 
        if self.slotLength == SlotLength.HALF_SUBSLOT:
            self.type3Blocks = empty(shape=(self.M, self.K1), dtype=uint8) 
        for i in range(self.M):
            self.type4Blocks[i] = descrambler(self.type5Blocks[i])
            if self.slotLength == SlotLength.HALF_SUBSLOT:
                self.type3Blocks[i] = blockDeInterleaver(self.type4Blocks[-1],101)
        
        if self.slotLength == SlotLength.FULL_SUBSLOT:
            self.type1Blocks = self.type4Blocks
        elif self.slotLength == SlotLength.HALF_SUBSLOT:
            self.type1Blocks = self.type3Blocks
        
        self.validateKLength(1)

    def stealBlockA(self):
        # in the case we are allocating bursts and we must steal a traffic channel with a full slot TCH/S, we must remap into a half one
        # this method just remaps an existing full slot TCH/S into a halve one, discarding block A bits
        self.slotLength = SlotLength.HALF_SUBSLOT
        self.K1 = 216
        self.K5 = 216
        self.type1Blocks[0][:216] = self.type1Blocks[0][216:432]
        self.type1Blocks.resize(1,216)
        self.encodeType5Bits(self.type1Blocks)

class TCH_7_2(TrafficChannel):
    '''
    7,2 kbit/s net rate (TCH/7.2)

    The traffic channels shall carry user information. Different traffic channels are defined for speech or data applications
    and for different data message speeds
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.N not in [1]:
            raise ValueError (f"The passed N - interleaving value of {self.N} is not valid for {self.__class__.__name__}")
        self.K1 = 432
        self.channel = ChannelName.TCH_7_2_CHANNEL

    def encodeType5Bits(self, inputDataBlocks:NDArray[uint8]):
        self.M = inputDataBlocks.shape[0]
        self.type1Blocks = inputDataBlocks
        self.validateKLength(1)
        self.type4Blocks = self.type1Blocks

        self.type5Blocks = empty(shape=(self.M, self.K5), dtype=uint8)
        for i in range(self.M):
            self.type5Blocks[i] = scrambler(self.type4Blocks[i])

        self.validateKLength(5)

    def decodeType5Bits(self, inputDataBlocks:NDArray[uint8]):
        self.M = inputDataBlocks.shape[0]
        self.type5Blocks = inputDataBlocks
        self.validateKLength(5)
        self.type4Blocks = empty(shape=(self.M, self.K5), dtype=uint8)

        for i in range(self.M):
            self.type4Blocks[i] = descrambler(self.type5Blocks[i])

        self.type1Blocks = self.type4Blocks
        self.validateKLength(1)

    
class TCH_4_8(TrafficChannel):
    '''
    4,8 kbit/s net rate (TCH/4.8)

    The traffic channels shall carry user information. Different traffic channels are defined for speech or data applications
    and for different data message speeds. Interleaving of depths N = 1,4, or 8 possible.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.N not in [1,4,8]:
            raise ValueError (f"The passed N - interleaving value of {self.N} is not valid for {self.__class__.__name__}")
        self.K1 = 288
        self.channel = ChannelName.TCH_4_8_CHANNEL

    def encodeType5Bits(self, inputDataBlocks:NDArray[uint8]):
        self.M = inputDataBlocks.shape[0]
        self.type1Blocks = inputDataBlocks
        self.validateKLength(1)
        self.type2Blocks = empty(shape=(self.M, 292), dtype=uint8)
        self.type3Blocks = empty(shape=(self.M, 432), dtype=uint8)
        for i in range(self.M):
            self.type2Blocks[i][:-4] = self.type1Blocks[i]
            self.type2Blocks[i][-4:] = TAIL_BITS_ARRAY
            self.type3Blocks[i] = rcpcEncoder(self.type2Blocks[i],292,432)

        self.type4Blocks = nBlockInterleaver(self.type3Blocks,self.N)
        self.type5Blocks = empty(shape=((self.M+self.N-1), 432), dtype=uint8)

        # blocks are scrambled individually
        for i in range((self.M+self.N-1)):
            self.type5Blocks[i] = scrambler(self.type4Blocks[i])
        
        self.validateKLength(5)

    def decodeType5Bits(self, inputDataBlocks:NDArray[uint8]):
        self.M = (len(inputDataBlocks) + 1 - self.N)
        self.type5Blocks = inputDataBlocks
        self.validateKLength(5)

        self.type4Blocks = empty(shape=((self.M+self.N-1), 432), dtype=uint8)
        # blocks are scrambled individually
        for i in range((self.M+self.N-1)):
            self.type4Blocks[i] = descrambler(self.type5Blocks[i])
        
        self.type3Blocks = nBlockDeInterleaver(self.type4Blocks,self.M,self.N)
        
        self.type2Blocks = empty(shape=(self.M, 292), dtype=uint8)
        self.type1Blocks = empty(shape=(self.M, 288), dtype=uint8)

        for i in range(self.M):
            self.type2Blocks[i] = rcpcDecoder(self.type3Blocks[i],292,292,432)
            self.type1Blocks[i] = self.type2Blocks[i][:-4]
        
        self.validateKLength(1)
    

class TCH_2_4(TrafficChannel):
    '''
    2,4 kbit/s net rate (TCH/2.4)

    The traffic channels shall carry user information. Different traffic channels are defined for speech or data applications
    and for different data message speeds. Interleaving of depths N = 1,4, or 8 possible.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.N not in [1,4,8]:
            raise ValueError (f"The passed N - interleaving value of {self.N} is not valid for {self.__class__.__name__}")
        self.K1 = 144
        self.channel = ChannelName.TCH_2_4_CHANNEL

    def encodeType5Bits(self, inputDataBlocks:NDArray[uint8]):
        self.M = inputDataBlocks.shape[0]
        self.type1Blocks = inputDataBlocks
        self.validateKLength(1)

        self.type2Blocks = empty(shape=((self.M+self.N-1), 148), dtype=uint8)
        self.type3Blocks = empty(shape=((self.M+self.N-1), 432), dtype=uint8)
        for i in range(self.M):
            self.type2Blocks[i][:-4] = self.type1Blocks[i]
            self.type2Blocks[i][-4:] = TAIL_BITS_ARRAY
            self.type3Blocks[i] = rcpcEncoder(self.type2Blocks[i],148,432)

        self.type4Blocks = nBlockInterleaver(self.type3Blocks, self.N)
        self.type5Blocks = empty(shape=((self.M+self.N-1), 432), dtype=uint8)

        # blocks are scrambled individually
        for i in range((self.M+self.N-1)):
            self.type5Blocks[i] = scrambler(self.type4Blocks[i])

        self.validateKLength(5)
    
    def decodeType5Bits(self, inputDataBlocks:NDArray[uint8]):
        self.M = (inputDataBlocks.shape[0] + 1 - self.N)
        self.type5Blocks = inputDataBlocks
        self.validateKLength(5)

        self.type4Blocks = empty(shape=((self.M+self.N-1), 432), dtype=uint8)
        # blocks are scrambled individually
        for i in range((self.M+self.N-1)):
            self.type4Blocks[i] = descrambler(self.type5Blocks[i])
        
        self.type3Blocks = nBlockDeInterleaver(self.type4Blocks,self.M,self.N)
        
        self.type2Blocks = empty(shape=(self.M, 148), dtype=uint8)
        self.type1Blocks = empty(shape=(self.M, 144), dtype=uint8)

        for i in range(self.M):
            self.type2Blocks[i] = rcpcDecoder(self.type3Blocks[i],148,148,432)
            self.type1Blocks[i] = self.type2Blocks[i][:-4]
        
        self.validateKLength(1)
    

###################################################################################################
# Linearization Channels

class Linearization_Channel(LogicalChannel_VD):

    def __init__(self):
        self.channelType = ChannelKind.LINEARIZATION_TYPE
    

###################################################################################################
class CLCH(Linearization_Channel):
    '''
    Common Linearization CHannel (CLCH)

    up-link, shared by all the MSs;
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.K1 = 238
        self.K5 = 238
        self.M = 1
        self.channel = ChannelName.CLCH_CHANNEL

    def encodeType5Bits(self, inputDataBlocks:NDArray[uint8]):
        self.type1Blocks = inputDataBlocks
        self.validateKLength(1)
        self.type5Blocks = self.type1Blocks
        self.validateKLength(5)
    
    def decodeType5Bits(self, inputDataBlocks:NDArray[uint8]):
        self.type5Blocks = inputDataBlocks
        self.validateKLength(5)
        self.type1Blocks = self.type5Blocks
        self.validateKLength(1)

class BLCH(Linearization_Channel):
    '''
    BS Linearization CHannel (BLCH)

    downlink, used by the BS
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.K1 = 216
        self.K5 = 216
        self.M = 1
        self.channel = ChannelName.BLCH_CHANNEL
    
    def encodeType5Bits(self, inputDataBlocks:NDArray[uint8]):
        self.type1Blocks = inputDataBlocks
        self.validateKLength(1)
        self.type5Blocks = self.type1Blocks
        self.validateKLength(5)
    
    def decodeType5Bits(self, inputDataBlocks:NDArray[uint8]):
       self.type5Blocks = inputDataBlocks
       self.validateKLength(5)
       self.type1Blocks = self.type5Blocks
       self.validateKLength(1)
