# ZT - 2026
# Based on EN 300 392-2 V2.4.2
from typing import List
from random import randint
from abc import ABC, abstractmethod
from .coding_scrambling import *

# TODO: rewrite constant strs into enums
# TODO: rewrite block building to numpy ndarrays allocation, abstract the building
HALF_SUBSLOT = "half"
FULL_SUBSLOT = "full"

TRAFFIC_TYPE = "traffic"
CONTROL_TYPE = "control"
LINEARIZATION_TYPE = "linear"

# Channel variable names for MAC burst building
BNCH_CHANNEL = "BNCH"
BSCH_CHANNEL = "BSCH"
SCH_CHANNEL = "SCH"
SCH_F_CHANNEL = "SCH/F"
SCH_HD_CHANNEL = "SCH/HD"
SCH_HU_CHANNEL = "SCH/HU"
AACH_CHANNEL = "AACH"
STCH_CHANNEL = "STCH"
TCH_CHANNEL = "TCH"
TCH_S_CHANNEL = "TCH/S"
TCH_7_2_CHANNEL = "TCH/7.2"
TCH_4_8_CHANNEL = "TCH/4.8"
TCH_2_4_CHANNEL = "TCH/2.4"
CLCH_CHANNEL = "CLCH"
BLCH_CHANNEL = "BLCH"

class LogicalChannel_VD(ABC):
    '''
    logical channels contain:
        1. a method to construct type-5 bits in type 5 blocks
        from type-1 information bits in type-1 blocks from MAC

        2. a method to construct type-1 bits in type 1 blocks from
        type-5 recieved bits in type-5 blocks from PHY
    '''

    # attributes contain the various types of bits, all are stored for now for analysis when needed
    type5Blocks = []
    type4Blocks = []
    type3Blocks = []
    type2Blocks = []
    type1Blocks = []

    channel = ""      # Describes the written name of the logical channel
    channelType = ""  # Channel type either traffic or control

    M = 0             # Number of input blocks

    K1 = 0            # Number of input bits per block
    K5 = 0            # Number of output bits per block

    CRC = 0           # 

    

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def encodeType5Bits(self):
        pass

    @abstractmethod
    def decodeType5Bits(self):
        pass

    @abstractmethod
    def generateRndInput(self):
        pass
    
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
            if len(self.type5Blocks) == 0:
                raise ValueError (f"type5Blocks unspecified")
            
            for i in range(self.M):
                if len(self.type5Blocks[i]) != self.K5:
                    raise ValueError (f"For logical channel {self.__class__.__name__}, expected K5: {self.K5} output bits, recieved {len(self.type5Blocks[i])} in block {i}")
        elif K == 1:
            if self.K1 == 0:
                raise ValueError (f"K1 unspecified")
            if len(self.type1Blocks) == 0:
                raise ValueError (f"type1Blocks unspecified")
            
            for i in range(self.M):
                if len(self.type1Blocks[i]) != self.K1:
                    raise ValueError (f"For logical channel {self.__class__.__name__}, expected K1: {self.K1} output bits, recieved {len(self.type1Blocks[i])} in block {i}")

###################################################################################################
# Control CHannels (CCH)
class ControlChannel(LogicalChannel_VD):

    def __init__(self):
        self.channelType = CONTROL_TYPE
    
    def generateRndInput(self):
        return [randint(0,1) for _ in range(self.K1)]

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
        self.channel = BNCH_CHANNEL
    
    def encodeType5Bits(self, inputDataBlocks:List[List[int]]):
        self.M = len(inputDataBlocks)
        if self.M != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}, only 1 data block is supported but {self.M} was passed")
        self.type1Blocks = inputDataBlocks
        self.validateKLength(1)
        self.type2Blocks = [crc16Encoder(inputDataBlocks[0],124) + [0,0,0,0]]
        self.type3Blocks = [rcpcEncoder(self.type2Blocks[0],2,3)]
        self.type4Blocks = [blockInterleaver(self.type3Blocks[0],216,101)]
        self.type5Blocks = [scrambler(self.type4Blocks[0])]
        self.validateKLength(5)
    
    def decodeType5Bits(self, inputDataBlocks:List[List[int]]):
        self.M = len(inputDataBlocks)
        if self.M != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}, only 1 data block is supported but {self.M} was passed")
        self.type5Blocks = inputDataBlocks
        self.validateKLength(5)
        self.type4Blocks = [descrambler(self.type5Blocks[0])]
        self.type3Blocks = [blockDeInterleaver(self.type4Blocks[0],216,101)]
        self.type2Blocks = [rcpcDecoder(self.type3Blocks[0],144,2,3)]
        *self.type1Blocks, self.CRC = crc16Decoder(self.type2Blocks[0][:-4],124)
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
        self.channel = BSCH_CHANNEL

    def encodeType5Bits(self, inputDataBlocks:List[List[int]]):
        self.M = len(inputDataBlocks)
        if self.M != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}, only 1 data block is supported but {self.M} was passed")
        
        self.type1Blocks = inputDataBlocks
        self.validateKLength(1)
        self.type2Blocks = [crc16Encoder(inputDataBlocks[0],60) + [0,0,0,0]]
        self.type3Blocks = [rcpcEncoder(self.type2Blocks[0],2,3)]
        self.type4Blocks = [blockInterleaver(self.type3Blocks[0],120,11)]
        self.type5Blocks = [scrambler(self.type4Blocks[0],BSCH=True)]
        self.validateKLength(5)
    
    def decodeType5Bits(self, inputDataBlocks:List[List[int]]):
        self.M = len(inputDataBlocks)
        if self.M != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}, only 1 data block is supported but {self.M} was passed")
        self.type5Blocks = inputDataBlocks
        self.validateKLength(5)
        self.type4Blocks = [descrambler(self.type5Blocks[0],BSCH=True)]
        self.type3Blocks = [blockDeInterleaver(self.type4Blocks[0],120,11)]
        self.type2Blocks = [rcpcDecoder(self.type3Blocks[0],80,2,3)]
        *self.type1Blocks, self.CRC = crc16Decoder(self.type2Blocks[0][:-4],60)
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
        self.channel = SCH_F_CHANNEL

    def encodeType5Bits(self, inputDataBlocks:List[List[int]]):
        self.M = len(inputDataBlocks)
        if self.M != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}, only 1 data block is supported but {self.M} was passed")
        self.type1Blocks = inputDataBlocks
        self.validateKLength(1)
        self.type2Blocks = [crc16Encoder(inputDataBlocks[0],268) + [0,0,0,0]]
        self.type3Blocks = [rcpcEncoder(self.type2Blocks[0],2,3)]
        self.type4Blocks = [blockInterleaver(self.type3Blocks[0],432,103)]
        self.type5Blocks = [scrambler(self.type4Blocks[0])]
        self.validateKLength(5)
    
    def decodeType5Bits(self, inputDataBlocks:List[List[int]]):
        self.M = len(inputDataBlocks)
        if self.M != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}, only 1 data block is supported but {self.M} was passed")
        self.type5Blocks = inputDataBlocks
        self.validateKLength(5)
        self.type4Blocks = [descrambler(self.type5Blocks[0])]
        self.type3Blocks = [blockDeInterleaver(self.type4Blocks[0],432,103)]
        self.type2Blocks = [rcpcDecoder(self.type3Blocks[0],288,2,3)]
        *self.type1Blocks, self.CRC = crc16Decoder(self.type2Blocks[0][:-4],268)
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
        self.channel = SCH_HD_CHANNEL
    
    def encodeType5Bits(self, inputDataBlocks:List[List[int]]):
        self.M = len(inputDataBlocks)
        if self.M != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}, only 1 data block is supported but {self.M} was passed")
        self.type1Blocks = inputDataBlocks
        self.validateKLength(1)
        self.type2Blocks = [crc16Encoder(inputDataBlocks[0],124) + [0,0,0,0]]
        self.type3Blocks = [rcpcEncoder(self.type2Blocks[0],2,3)]
        self.type4Blocks = [blockInterleaver(self.type3Blocks[0],216,101)]
        self.type5Blocks = [scrambler(self.type4Blocks[0])]
        self.validateKLength(5)
    
    def decodeType5Bits(self, inputDataBlocks:list):
        self.M = len(inputDataBlocks)
        if self.M != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}, only 1 data block is supported but {self.M} was passed")
        self.type5Blocks = inputDataBlocks
        self.validateKLength(5)
        self.type4Blocks = [descrambler(self.type5Blocks[0])]
        self.type3Blocks = [blockDeInterleaver(self.type4Blocks[0],216,101)]
        self.type2Blocks = [rcpcDecoder(self.type3Blocks[0],144,2,3)]
        *self.type1Blocks, self.CRC = crc16Decoder(self.type2Blocks[0][:-4],124)
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
        self.channel = SCH_HU_CHANNEL
    
    def encodeType5Bits(self, inputDataBlocks:List[List[int]]):
        self.M = len(inputDataBlocks)
        if self.M != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}, only 1 data block is supported but {self.M} was passed")
        self.type1Blocks = inputDataBlocks
        self.validateKLength(1)
        self.type2Blocks = [crc16Encoder(inputDataBlocks[0],92) + [0,0,0,0]]
        self.type3Blocks = [rcpcEncoder(self.type2Blocks[0],2,3)]
        self.type4Blocks = [blockInterleaver(self.type3Blocks[0],168,13)]
        self.type5Blocks = [scrambler(self.type4Blocks[0])]
        self.validateKLength(5)
    
    def decodeType5Bits(self, inputDataBlocks:List[List[int]]):
        self.M = len(inputDataBlocks)
        if self.M != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}, only 1 data block is supported but {self.M} was passed")
        self.type5Blocks = inputDataBlocks
        self.validateKLength(5)
        self.type4Blocks = [descrambler(self.type5Blocks[0])]
        self.type3Blocks = [blockDeInterleaver(self.type4Blocks[0],168,13)]
        self.type2Blocks = [rcpcDecoder(self.type3Blocks[0],112,2,3)]
        *self.type1Blocks, self.CRC = crc16Decoder(self.type2Blocks[0][:-4],92)
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
        self.channel = AACH_CHANNEL

    def encodeType5Bits(self, inputDataBlocks:List[List[int]]):
        self.M = len(inputDataBlocks)
        self.type1Blocks = inputDataBlocks
        self.validateKLength(1)
        if self.M != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}, only 1 data block is supported but {self.M} was passed")
        self.type5Blocks = [scrambler(rm3014Encoder(self.type1Blocks[0]), False)]
        self.validateKLength(5)
    
    def decodeType5Bits(self, inputDataBlocks:List[List[int]]):
        self.M = len(inputDataBlocks)
        if self.M != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}, only 1 data block is supported but {self.M} was passed")
        self.type5Blocks = inputDataBlocks
        self.validateKLength(5)
        self.type4Blocks = [descrambler(self.type5Blocks[0])]
        self.type1Blocks = [rm3014Decoder(self.type4Blocks[0])]
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
        self.channel = STCH_CHANNEL

    def encodeType5Bits(self, inputDataBlocks:List[List[int]]):
        self.M = len(inputDataBlocks)
        if self.M != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}, only 1 data block is supported but {self.M} was passed")
        self.type1Blocks = inputDataBlocks
        self.validateKLength(1)
        self.type2Blocks = [crc16Encoder(inputDataBlocks[0],124) + [0,0,0,0]]
        self.type3Blocks = [rcpcEncoder(self.type2Blocks[0],2,3)]
        self.type4Blocks = [blockInterleaver(self.type3Blocks[0],216,101)]
        self.type5Blocks = [scrambler(self.type4Blocks[0])]
        self.validateKLength(5)
    
    def decodeType5Bits(self, inputDataBlocks:List[List[int]]):
        self.M = len(inputDataBlocks)
        if self.M != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}, only 1 data block is supported but {self.M} was passed")
        self.type5Blocks = inputDataBlocks
        self.validateKLength(5)
        self.type4Blocks = [descrambler(self.type5Blocks[0])]
        self.type3Blocks = [blockDeInterleaver(self.type4Blocks[0],216,101)]
        self.type2Blocks = [rcpcDecoder(self.type3Blocks[0],144,2,3)]
        *self.type1Blocks, self.CRC = crc16Decoder(self.type2Blocks[0][:-4],124)
        self.validateKLength(1)

###################################################################################################
# Traffic Channels

class TrafficChannel(LogicalChannel_VD):
    N = 1
    def __init__(self, N:int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channelType = TRAFFIC_TYPE
        self.N = N
        self.K5 = 432
        if self.N not in [1,2,4,8]:
            raise ValueError (f"The passed N - interleaving value of {self.N} is not valid.")

    def generateRndInput(self) -> List[int]:
        return[randint(0,1) for _ in range(self.K1)]
    
class TCH_S(TrafficChannel):
    '''
    Speech Traffic Channel (TCH/S)

    The traffic channels shall carry user information, defined for speech.
    '''
    slotLength = ""

    def __init__(self, slotLength:str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slotLength = slotLength
        if self.slotLength != HALF_SUBSLOT and self.slotLength != FULL_SUBSLOT:
            raise ValueError (f"The passed slot length value of {self.slotLength} is not of: 'half' or 'full' ")
        if self.N not in [1]:
            raise ValueError (f"The passed N - interleaving value of {self.N} is not valid for {self.__class__.__name__}")
        self.K1 = 432 if self.slotLength == FULL_SUBSLOT else 216
        self.K5 = 432 if self.slotLength == FULL_SUBSLOT else 216

    def encodeType5Bits(self, inputDataBlocks:List[List[int]]):
        self.M = len(inputDataBlocks)
        self.type1Blocks = inputDataBlocks
        self.validateKLength(1)
        if self.slotLength == FULL_SUBSLOT:
            self.type4Blocks = inputDataBlocks
            self.type5Blocks = []
            for i in range(self.M):
                self.type5Blocks.append(scrambler(self.type4Blocks[i]))
        elif self.slotLength == HALF_SUBSLOT:
            self.type3Blocks = inputDataBlocks
            self.type4Blocks = []
            self.type5Blocks = []
            for i in range(self.M):
                self.type4Blocks.append(blockInterleaver(self.type3Blocks[i],216,101))
                self.type5Blocks.append(scrambler(self.type4Blocks[-1]))

        self.validateKLength(5)
    
    def decodeType5Bits(self, inputDataBlocks:List[List[int]]):
        self.M = len(inputDataBlocks)
        self.type5Blocks = inputDataBlocks
        self.validateKLength(5)
        self.type4Blocks = [] 
        self.type3Blocks = []
        for i in range(self.M):
            self.type4Blocks.append(descrambler(self.type5Blocks[i]))
            if self.slotLength == HALF_SUBSLOT:
                self.type3Blocks.append(blockDeInterleaver(self.type4Blocks[-1],216,101))
        
        if self.slotLength == FULL_SUBSLOT:
            self.type1Blocks = self.type4Blocks
        elif self.slotLength == HALF_SUBSLOT:
            self.type1Blocks = self.type3Blocks
        
        self.validateKLength(1)

    def stealBlockA(self):
        # in the case we are allocating bursts and we must steal a traffic channel with a full slot TCH/S, we must remap into a half one
        # this method just remaps an existing full slot TCH/S into a halve one, discarding block A bits
        self.slotLength = HALF_SUBSLOT
        self.K1 = 216
        self.K5 = 216
        self.type1Blocks[:216] = self.type1Blocks[216:432]
        self.encodeType5Bits(self.type1Blocks[:216])

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
        self.channel = TCH_7_2_CHANNEL

    def encodeType5Bits(self, inputDataBlocks:List[List[int]]):
        self.M = len(inputDataBlocks)
        self.type1Blocks = inputDataBlocks
        self.validateKLength(1)
        self.type4Blocks = self.type1Blocks

        self.type5Blocks = []
        for i in range(self.M):
            self.type5Blocks.append(scrambler(self.type4Blocks[i]))

        self.validateKLength(5)

    def decodeType5Bits(self, inputDataBlocks:List[List[int]]):
        self.M = len(inputDataBlocks)
        self.type5Blocks = inputDataBlocks
        self.validateKLength(5)
        self.type4Blocks = []
        for i in range(self.M):
            self.type4Blocks.append(descrambler(self.type5Blocks[i]))

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
        self.channel = TCH_4_8_CHANNEL

    def encodeType5Bits(self, inputDataBlocks:List[List[int]]):
        self.M = len(inputDataBlocks)
        self.type1Blocks = inputDataBlocks
        self.validateKLength(1)
        self.type2Blocks = []
        self.type3Blocks = []
        for i in range(self.M):
            self.type2Blocks.append(self.type1Blocks[i]+[0,0,0,0])
            self.type3Blocks.append(rcpcEncoder(self.type2Blocks[-1],292,432))

        self.type4Blocks = nBlockInterleaver(self.type3Blocks,self.M,self.N)
        self.type5Blocks = []

        # blocks are scrambled individually
        for i in range((self.M+self.N-1)):
            self.type5Blocks.append(scrambler(self.type4Blocks[i]))
        
        self.validateKLength(5)

    def decodeType5Bits(self, inputDataBlocks:List[List[int]]):
        self.M = (len(inputDataBlocks) + 1 - self.N)
        self.type5Blocks = inputDataBlocks
        self.validateKLength(5)
        self.type4Blocks = []
        # blocks are scrambled individually
        for i in range(len(inputDataBlocks)):
            self.type4Blocks.append(descrambler(self.type5Blocks[i]))
        
        self.type3Blocks = nBlockDeInterleaver(self.type4Blocks,self.M,self.N)
        
        self.type2Blocks = []
        self.type1Blocks = []

        for i in range(self.M):
            self.type2Blocks.append(rcpcDecoder(self.type3Blocks[i],292,292,432))
            self.type1Blocks.append(self.type2Blocks[-1][:-4])
        
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
        self.channel = TCH_2_4_CHANNEL

    def encodeType5Bits(self, inputDataBlocks:List[List[int]]):
        self.M = len(inputDataBlocks)
        self.type1Blocks = inputDataBlocks
        self.validateKLength(1)
        self.type2Blocks = []
        self.type3Blocks = []
        for i in range(self.M):
            self.type2Blocks.append(self.type1Blocks[i]+[0,0,0,0])
            self.type3Blocks.append(rcpcEncoder(self.type2Blocks[-1],148,432))

        self.type4Blocks = nBlockInterleaver(self.type3Blocks,self.M,self.N)
        self.type5Blocks = []

        # blocks are scrambled individually
        for i in range((self.M+self.N-1)):
            self.type5Blocks.append(scrambler(self.type4Blocks[i]))

        self.validateKLength(5)
    
    def decodeType5Bits(self, inputDataBlocks:List[List[int]]):
        self.M = (len(inputDataBlocks) + 1 - self.N)
        self.type5Blocks = inputDataBlocks
        self.validateKLength(5)
        self.type4Blocks = []
        # blocks are scrambled individually
        for i in range(len(inputDataBlocks)):
            self.type4Blocks.append(descrambler(self.type5Blocks[i]))
        
        self.type3Blocks = nBlockDeInterleaver(self.type4Blocks,self.M,self.N)
        
        self.type2Blocks = []
        self.type1Blocks = []

        for i in range(self.M):
            self.type2Blocks.append(rcpcDecoder(self.type3Blocks[i],148,148,432))
            self.type1Blocks.append(self.type2Blocks[-1][:-4])
        
        self.validateKLength(1)
    

###################################################################################################
# Linearization Channels

class Linearization_Channel(LogicalChannel_VD):

    def __init__(self):
        self.channelType = LINEARIZATION_TYPE
    
    def generateRndInput(self):
        return [randint(0,1) for _ in range(self.K1)]

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
        self.channel = CLCH_CHANNEL

    def encodeType5Bits(self):
        raise NotImplementedError
    
    def decodeType5Bits(self):
       raise NotImplementedError

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
        self.channel = BLCH_CHANNEL
    
    def encodeType5Bits(self):
        raise NotImplementedError
    
    def decodeType5Bits(self):
       raise NotImplementedError
