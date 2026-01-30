# ZT - 2026
# Based on EN 300 392-2 V2.4.2

from numpy import uint8, array, empty, zeros
from abc import ABC, abstractmethod
from .constants import PhyType, LinkDirection, BurstContent, ChannelKind, ChannelName, SUBSLOT_BIT_LENGTH, TIMESLOT_SYMBOL_LENGTH
from .logical_channels import BLCH, BNCH, BSCH, CLCH, SCH_HD, SCH_F, SCH_HU, AACH, LogicalChannel_VD, TrafficChannel, STCH
from .modulation import calculatePhaseAdjustmentBits
from typing import List, Union, ClassVar, Optional, Protocol, Tuple
from dataclasses import dataclass

HYPERFRAME_MULTIFRAME_LENGTH = 60
MULTIFRAME_TDMAFRAME_LENGTH = 18
CONTROL_FRAME_NUMBER = 18
TDMAFRAME_TIMESLOT_LENGTH = 4
TIMESLOT_BIT_LENGTH = 510
TIMESLOT_SUBSLOT_LENGTH = 2

MULTIFRAME_DURATION_MS = 1020
MULTIFRAME_BIT_COUNT = 36720
# resulting modulation bit duration = 250/9 us

FREQUENCY_CORRECTION_FIELD = array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              1,1,1,1,1,1,1,1],dtype=uint8)

NORMAL_TRAINING_SEQUENCE = array([[1,1,0,1,0,0,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0],
                                    [0,1,1,1,1,0,1,0,0,1,0,0,0,0,1,1,0,1,1,1,1,0],
                                    [1,0,1,1,0,1,1,1,0,0,0,0,0,1,1,0,1,0,1,1,0,1]],dtype=uint8)

EXTENDED_TRAINING_SEQUENCE = array([1,0,0,1,1,1,0,1,0,0,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,0,0,1,1],dtype=uint8)

SYNCHRONIZATION_TRAINING_SEQUENCE = array([1,1,0,0,0,0,0,1,1,0,0,1,1,1,0,0,1,1,1,0,1,0,0,1,1,1,0,0,0,0,0,1,1,0,0,1,1,1],dtype=uint8)

PHASE_ADJUSTMENT_SYMBOL_RANGE = {"a":(7,121), "b":(122,248), "c":(7,107), "d":(108,248), "e":(111,229),
                             "f":(0,110), "g":(2,116), "h":(117,243), "i":(2,102), "j":(103,243)}

TAIL_BITS = array([1, 1, 0, 0],dtype=uint8)

# # Control Modes
# NCM_CTRL_MODE = "NCM"       # Normal Control Mode
# MC_CTRL_MODE = "MC"         # Minimum Control Mode

# # Transmission Modes
# D_CT_BS_MODE = "D-CT"       # Downlink-Continuous Transmission (D-CT)
# D_CTT_BS_MODE = "D-CTT"     # Downlink-Carrier Timesharing Transmission (D-CTT)
# D_MCCTT_BS_MODE = "D-MCCTT" # Downlink-Main Control Channel Timesharing Transmission (D-MCCTT)
# U_MST_BS_MODE = "U-MST"     # Multiple Slot Transmission

###################################################################################################
@dataclass(frozen=True, slots=True)
class Physical_Channel():
    channelNumber: int          # The channel number 
    mainCarrier: bool           # If the carrier used is considered the "mainCarrier"
    UL_Frequency: float         # The UL frequency for MS->BS tx
    DL_Frequency: float         # The DL frequency for BS->MS tx
    channelType: PhyType        # The type of physical channel (CP,TP,UP)
        
###################################################################################################

class Burst(ABC):
    # require class-level "spec constants"
    SNmax: ClassVar[int]                   # Max number of modulation symbols in burst
    startGuardBitPeriod: ClassVar[int]     # The initial guard period / delay in bits (default)
    endGuardBitPeriod: ClassVar[int]       # The end guard period in bits (default)
    subSlotWidth: ClassVar[int]            # How many subslots does the burst take up: 1 or 2
    linkDirection: ClassVar[LinkDirection] # either DL or UL
    
    

    # Permissible physical channel types for the burst class
    ALLOWED_PHY: ClassVar[set[PhyType]]

   

    # per instance runtime variables
    burstType: BurstContent      # either traffic, control, or mixed
    phyChannel: PhyType          # The physical RF channel type (CP,TP,UP) that the burst utilizes
    mixedBurst:bool              # if the burst has 1 or more blocks/subslots stolen for control 
                                 #   or is just a composite burst made of two seperate logical channels
    multiFrameNumber: int        # The multi-frame number of the burst
    frameNumber: int             # The frame number of the burst
    timeSlot: int                # The TDMA time slot number of the burst
    subSlot: int                 # The subslot which the burst starts in and occupies atleast

    burstStartRampPeriod: int     # Variable version of startGuardBitPeriod that allows for dynamic ramping based on if we have continuous data
    burstEndRampPeriod: int       # Variable version of startGuardBitPeriod that allows for dynamic ramping based on if we have continuous data

    def __init__(self, phyChannel: Physical_Channel, MN:int, FN:int, TN:int, SSN:int=1):
        self._validate_class_constants()
        # store runtime state
        self.phyChannel = phyChannel.channelType
        self.multiFrameNumber = MN
        self.frameNumber = FN
        self.timeSlot = TN
        self.subSlot = SSN

        # default; burst building may change later on to True
        self.mixedBurst = False
        
        # default is the guard delay periods, but burst building may change later on
        self.burstStartRampPeriod = self.startGuardBitPeriod
        self.burstEndRampPeriod = self.endGuardBitPeriod

        self._validate_common()

    def _validate_common(self):
        # TN range checks
        if not (1 <= self.timeSlot <= TDMAFRAME_TIMESLOT_LENGTH):
            raise ValueError(f"TN {self.timeSlot} invalid for {type(self).__name__}")
        # FN range checks
        if not (1<= self.frameNumber <= MULTIFRAME_TDMAFRAME_LENGTH):
            raise ValueError(f"FN {self.frameNumber} invalid for {type(self).__name__}")
        # MN range checks
        if not (1 <= self.multiFrameNumber <= HYPERFRAME_MULTIFRAME_LENGTH):
            raise ValueError(f"MN {self.multiFrameNumber} invalid for {type(self).__name__}")
        # SSN range checks depending on subSlotWidth
        if not (1 <= self.subSlot <= TIMESLOT_SUBSLOT_LENGTH):
            raise ValueError(f"SSN {self.subSlot} invalid for {type(self).__name__}")
        # Physical channel type allowed
        allowed = getattr(type(self), "ALLOWED_PHY", None)
        if allowed is not None and self.phyChannel not in allowed:
            raise ValueError(f"Phy {self.phyChannel} invalid for {type(self).__name__}")

    def _validate_class_constants(self):
        requiredVariables = ["SNmax","startGuardBitPeriod","endGuardBitPeriod",
                             "subSlotWidth","linkDirection", "ALLOWED_PHY"]
        for name in requiredVariables:
            if not hasattr(type(self), name):
                raise TypeError(f"{type(self).__name__} missing class constant {name}")
    
    @abstractmethod
    def constructBurstBitSequence(self):
        pass

    @abstractmethod
    def deconstructBurstBitSequence(self):
        pass

###################################################################################################

class Control_Uplink(Burst):
    SNmax = 103
    startGuardBitPeriod = 34
    endGuardBitPeriod = 15
    subSlotWidth = 1
    linkDirection = LinkDirection.UPLINK
    burstType = BurstContent.BURST_CONTROL_TYPE
    
    ALLOWED_PHY = {PhyType.CONTROL_CHANNEL, PhyType.TRAFFIC_CHANNEL}

    def constructBurstBitSequence(self, inputLogicalChannelSsn1:SCH_HU):
        # Must verify specific non-common TN/FN based on physical channel
        if self.phyChannel == PhyType.CONTROL_CHANNEL:
            if not (1 <= self.frameNumber <= MULTIFRAME_TDMAFRAME_LENGTH):
                raise ValueError(f"For {type(self).__name__}, phy {self.phyChannel}, FN {self.frameNumber} invalid for ssn{self.subSlot}: {inputLogicalChannelSsn1.channel}")
        elif self.phyChannel == PhyType.TRAFFIC_CHANNEL:
            if self.frameNumber != CONTROL_FRAME_NUMBER:
                raise ValueError(f"For {type(self).__name__}, phy {self.phyChannel}, FN {self.frameNumber} invalid for ssn{self.subSlot}: {inputLogicalChannelSsn1.channel}")
        # runtime check to verify channel type
        if inputLogicalChannelSsn1.channel != ChannelName.SCH_HU_CHANNEL:
            raise ValueError(f"For {type(self).__name__}, phy {self.phyChannel}, invalid ssn of {inputLogicalChannelSsn1.channel}")
        
        # Build the burst
        d = self.startGuardBitPeriod
        burstBitSequence = empty(shape=(self.SNmax*2)+d+self.endGuardBitPeriod, dtype=uint8)
        burstBitSequence[:d] = zeros(shape=self.startGuardBitPeriod, dtype=uint8) # guard period
        burstBitSequence[d:4+d] = TAIL_BITS
        burstBitSequence[d+4:88+d] = inputLogicalChannelSsn1.type5Blocks[0][:84]
        burstBitSequence[d+88:118+d] = EXTENDED_TRAINING_SEQUENCE
        burstBitSequence[d+118:202+d] = inputLogicalChannelSsn1.type5Blocks[0][84:168]
        burstBitSequence[d+202:206+d] = TAIL_BITS
        burstBitSequence[206+d:255] = zeros(shape=self.endGuardBitPeriod, dtype=uint8) # guard period
        
        return burstBitSequence
    
    def deconstructBurstBitSequence(self, inputData:List[int]):
        raise NotImplementedError

###################################################################################################

class Normal_Uplink_Burst(Burst):
    SNmax = 231
    startGuardBitPeriod = 34
    endGuardBitPeriod = 14
    subSlotWidth = 2
    linkDirection = LinkDirection.UPLINK
    burstType = BurstContent.BURST_MIXED_TYPE

    ALLOWED_PHY = {PhyType.CONTROL_CHANNEL, PhyType.TRAFFIC_CHANNEL}

    def constructBurstBitSequence(self, inputLogicalChannelBkn1:Union[TrafficChannel, SCH_F, STCH],
                                    inputLogicalChannelBkn2:Union[TrafficChannel, STCH, None]=None,
                                    rampUpandDown:Tuple[bool,bool]=(True,True)):

        bkn1 = inputLogicalChannelBkn1.channelType
        bkn2 = None if inputLogicalChannelBkn2 is None else inputLogicalChannelBkn2.channelType
        
        if inputLogicalChannelBkn1.channel not in (ChannelName.TCH_2_4_CHANNEL, ChannelName.TCH_4_8_CHANNEL, 
                                                    ChannelName.TCH_7_2_CHANNEL, ChannelName.TCH_S_CHANNEL,
                                                    ChannelName.SCH_F_CHANNEL, ChannelName.STCH_CHANNEL):
            raise ValueError (f"For {type(self).__name__}, phy {self.phyChannel} invalid bkn1:{inputLogicalChannelBkn1.channel}")
        
        if inputLogicalChannelBkn2 is not None and inputLogicalChannelBkn2.channel not in (ChannelName.TCH_2_4_CHANNEL, ChannelName.TCH_4_8_CHANNEL, 
                                                                                        ChannelName.TCH_7_2_CHANNEL, ChannelName.TCH_S_CHANNEL,
                                                                                        ChannelName.STCH_CHANNEL):
            raise ValueError (f"For {type(self).__name__}, phy {self.phyChannel} invalid bkn2:{inputLogicalChannelBkn2.channel}")
            
        trainingSequenceIndex = 0 # Depending on composition of burst, we use a different training sequence either 1 or 2 per Table 25

        match (bkn1, bkn2): 
            case (ChannelKind.TRAFFIC_TYPE, None):
                # Pure TCH on TP with FN:[1,17]
                self.mixedBurst = False
                self.burstType = BurstContent.BURST_TRAFFIC_TYPE
                trainingSequenceIndex = 0
                if self.phyChannel != PhyType.TRAFFIC_CHANNEL:
                    raise ValueError(f"For {type(self).__name__}, phy {self.phyChannel} invalid for bkn1:{bkn1} bkn2:{bkn2} (expected TP)")
                
                if not (1<= self.frameNumber < MULTIFRAME_TDMAFRAME_LENGTH):
                    raise ValueError(f"For {type(self).__name__}, phy {self.phyChannel}, FN {self.frameNumber} invalid for bkn1:{bkn1} bkn2:{bkn2}")

            case (ChannelKind.CONTROL_TYPE, None):
                # Pure SCH/F on CP with FN:[1,18] or on TP with FN:18
                self.mixedBurst = False
                self.burstType = BurstContent.BURST_CONTROL_TYPE
                trainingSequenceIndex = 0
                if self.phyChannel == PhyType.CONTROL_CHANNEL:
                    if not (1<= self.frameNumber <= MULTIFRAME_TDMAFRAME_LENGTH):
                        raise ValueError(f"For {type(self).__name__}, phy {self.phyChannel}, FN {self.frameNumber} invalid for bkn1:{bkn1} bkn2:{bkn2}")
                elif self.phyChannel == PhyType.TRAFFIC_CHANNEL:
                    if self.frameNumber != CONTROL_FRAME_NUMBER:
                        raise ValueError(f"For {type(self).__name__}, phy {self.phyChannel}, FN {self.frameNumber} invalid for bkn1:{bkn1} bkn2:{bkn2}")
                else:
                    raise ValueError (f"For {self.__class__.__name__}, phy {self.phyChannel} invalid for bkn1:{bkn1} bkn2:{bkn2}")

            case (ChannelKind.CONTROL_TYPE, ChannelKind.TRAFFIC_TYPE):
                # BKN1 stolen for STCH on TP with FN:[1,17], and BKN2 as TCH
                self.mixedBurst = True
                self.burstType = BurstContent.BURST_TRAFFIC_TYPE
                trainingSequenceIndex = 1
                if self.phyChannel != PhyType.TRAFFIC_CHANNEL:
                    raise ValueError(f"For {type(self).__name__}, phy {self.phyChannel} invalid for bkn1:{bkn1} bkn2:{bkn2} (expected TP)")
                if not (1<= self.frameNumber < MULTIFRAME_TDMAFRAME_LENGTH):
                    raise ValueError(f"For {type(self).__name__}, phy {self.phyChannel}, FN {self.frameNumber} invalid for bkn1:{bkn1} bkn2:{bkn2}")

            case (ChannelKind.CONTROL_TYPE, _):
                if bkn2 == ChannelKind.CONTROL_TYPE:
                    if inputLogicalChannelBkn2 is not None and inputLogicalChannelBkn2.channel != inputLogicalChannelBkn1.channel:
                        raise ValueError(f"For {type(self).__name__}, phy {self.phyChannel} invalid combination bkn1:{bkn1} bkn2:{bkn2}")
                    # BKN1 and BKN2 stolen for STCH on TP with FN:[1,17]
                    self.mixedBurst = True
                    self.burstType = BurstContent.BURST_TRAFFIC_TYPE
                    trainingSequenceIndex = 1
                    if self.phyChannel != PhyType.TRAFFIC_CHANNEL:
                        raise ValueError(f"For {type(self).__name__}, phy {self.phyChannel} invalid for bkn1:{bkn1} bkn2:{bkn2} (expected TP)")
                    if not (1<= self.frameNumber < MULTIFRAME_TDMAFRAME_LENGTH):
                        raise ValueError(f"For {type(self).__name__}, phy {self.phyChannel}, FN {self.frameNumber} invalid for bkn1:{bkn1} bkn2:{bkn2}")
                else:
                    # Invalid combo of channels
                    raise ValueError(f"For {type(self).__name__}, phy {self.phyChannel} invalid for bkn1:{bkn1} bkn2:{bkn2}")
            case _:
                # Invalid combo of channels
                raise ValueError(f"For {type(self).__name__}, invalid combination bkn1:{bkn1} bkn2:{bkn2}")
            
         # Build the burst
        d = self.startGuardBitPeriod
        burstBitSequence = empty(shape=(self.SNmax*2)+d+self.endGuardBitPeriod, dtype=uint8)

        burstBitSequence[d:4+d] = TAIL_BITS
        burstBitSequence[d+4:220+d] = inputLogicalChannelBkn1.type5Blocks[0][:216]
        burstBitSequence[d+220:242+d] = NORMAL_TRAINING_SEQUENCE[trainingSequenceIndex]
        if inputLogicalChannelBkn2 is not None and self.mixedBurst:
            burstBitSequence[d+242:458+d] = inputLogicalChannelBkn2.type5Blocks[0][:216]
        else:
            burstBitSequence[d+242:458+d] = inputLogicalChannelBkn1.type5Blocks[0][216:432]
        burstBitSequence[d+458:462+d] = TAIL_BITS

        # must add guard period training sequence if there is no ramping at the start
        if rampUpandDown[0]:
            burstBitSequence[:d] = zeros(shape=self.startGuardBitPeriod, dtype=uint8)
            self.burstStartRampPeriod = d
        else:
            # add preceding bits per 9.4.5.3
            burstBitSequence[:30] = EXTENDED_TRAINING_SEQUENCE
            burstBitSequence[30:32] = TAIL_BITS
            # Insert phase adjustment bits f
            burstBitSequence[32:d] = calculatePhaseAdjustmentBits(burstBitSequence,
                                                                  PHASE_ADJUSTMENT_SYMBOL_RANGE["f"], d)
            self.burstStartRampPeriod = 0
        
        # must add guard period training sequence if there is no ramping at the end
        if rampUpandDown[1]:
            burstBitSequence[462+d:510] = zeros(shape=self.endGuardBitPeriod, dtype=uint8)
            self.burstEndRampPeriod = TIMESLOT_BIT_LENGTH - (462+d)
        else:
            # add following bits per 9.4.5.3
            # Insert phase adjustment bits f
            burstBitSequence[462+d:464+d] = calculatePhaseAdjustmentBits(burstBitSequence,
                                                                  PHASE_ADJUSTMENT_SYMBOL_RANGE["e"], d)
            burstBitSequence[464+d:466+d] = TAIL_BITS
            burstBitSequence[466+d:510] = NORMAL_TRAINING_SEQUENCE[2][0:10]
            self.burstEndRampPeriod = 0

        return burstBitSequence
    
    def deconstructBurstBitSequence(self, inputData:List[int]):
        raise NotImplementedError

###################################################################################################

class DownlinkHost(Protocol):
    # host protocol for the mixin to satisfy typing for both normal and synchronous downlink mixins
    phyChannel: PhyType
    frameNumber: int
    multiFrameNumber: int
    timeSlot: int

class NormalDownlinkMixin:

    def _norm_bkin(self, logicalChannel:LogicalChannel_VD) -> str:
        return logicalChannel.channelType if logicalChannel.channelType == ChannelKind.TRAFFIC_TYPE else logicalChannel.channel

    def _validate_normal_downlink_mapping(self:DownlinkHost, bkn1: str, 
                                          bkn2: Optional[str]) -> Tuple[BurstContent, bool, int]:
        """
        Validates (bkn1, bkn2) for normal downlink burst usage across CP/TP/UP
        and returns:
            (burstType: BurstContent, mixedBurst: bool, trainingSequenceIndex: int)
        """
        if bkn2 is None:
            if bkn1 not in (ChannelKind.TRAFFIC_TYPE, ChannelName.SCH_F_CHANNEL):
                raise ValueError(
                    f"For {type(self).__name__}, bkn2 is None but bkn1:{bkn1} is not in "
                    f"({ChannelKind.TRAFFIC_TYPE}, {ChannelName.SCH_F_CHANNEL})"
                    )

        match (bkn1, bkn2):
            case (ChannelKind.TRAFFIC_TYPE, None):
                # Pure TCH on TP with FN:[1,17]
                if self.phyChannel != PhyType.TRAFFIC_CHANNEL:
                    raise ValueError(
                        f"For {type(self).__name__}, phy {self.phyChannel} invalid for pure TCH (expected TP). "
                        f"bkn1:{bkn1} bkn2:{bkn2}"
                    )
                if not (1 <= self.frameNumber < MULTIFRAME_TDMAFRAME_LENGTH):
                    raise ValueError(
                        f"For {type(self).__name__}, FN {self.frameNumber} invalid for pure TCH on TP "
                        f"(expected 1..{MULTIFRAME_TDMAFRAME_LENGTH-1})."
                    )
                return (BurstContent.BURST_TRAFFIC_TYPE, False, 0)

            case (ChannelName.SCH_F_CHANNEL, None):
                # Pure SCH/F on CP with FN:[1,18] or on TP with FN:18
                if self.phyChannel == PhyType.TRAFFIC_CHANNEL:
                    if self.frameNumber != CONTROL_FRAME_NUMBER:
                        raise ValueError(
                            f"For {type(self).__name__}, FN {self.frameNumber} invalid for SCH/F on TP "
                            f"(expected control frame {CONTROL_FRAME_NUMBER})."
                        )
                elif self.phyChannel == PhyType.CONTROL_CHANNEL:
                    if not (1 <= self.frameNumber < MULTIFRAME_TDMAFRAME_LENGTH):
                        raise ValueError(
                            f"For {type(self).__name__}, FN {self.frameNumber} invalid for SCH/F on CP "
                            f"(expected 1..{MULTIFRAME_TDMAFRAME_LENGTH-1})."
                        )
                else:
                    raise ValueError(f"For {type(self).__name__}, phy {self.phyChannel} invalid for bkn1:{bkn1} bkn2:{bkn2}")
                return (BurstContent.BURST_CONTROL_TYPE, False, 0)

            case (ChannelName.STCH_CHANNEL, ChannelKind.TRAFFIC_TYPE):
                # BKN1 stolen for STCH on TP FN:[1,17], BKN2 as TCH
                if self.phyChannel != PhyType.TRAFFIC_CHANNEL:
                    raise ValueError(
                        f"For {type(self).__name__}, phy {self.phyChannel} invalid for STCH+TCH (expected TP). "
                        f"bkn1:{bkn1} bkn2:{bkn2}"
                    )
                if not (1 <= self.frameNumber < MULTIFRAME_TDMAFRAME_LENGTH):
                    raise ValueError(
                        f"For {type(self).__name__}, FN {self.frameNumber} invalid for STCH+TCH on TP "
                        f"(expected 1..{MULTIFRAME_TDMAFRAME_LENGTH-1})."
                    )
                return (BurstContent.BURST_MIXED_TYPE, True, 1)

            case (ChannelName.STCH_CHANNEL, _):
                if bkn2 == ChannelName.STCH_CHANNEL:
                    # BKN1 and BKN2 stolen for STCH on TP FN:[1,17]
                    if self.phyChannel != PhyType.TRAFFIC_CHANNEL:
                        raise ValueError(
                            f"For {type(self).__name__}, phy {self.phyChannel} invalid for STCH+STCH (expected TP). "
                            f"bkn1:{bkn1} bkn2:{bkn2}"
                        )
                    if not (1 <= self.frameNumber < MULTIFRAME_TDMAFRAME_LENGTH):
                        raise ValueError(
                            f"For {type(self).__name__}, FN {self.frameNumber} invalid for STCH+STCH on TP "
                            f"(expected 1..{MULTIFRAME_TDMAFRAME_LENGTH-1})."
                        )
                    return (BurstContent.BURST_MIXED_TYPE, True, 1)

                raise ValueError(f"For {type(self).__name__}, invalid combination bkn1:{bkn1} bkn2:{bkn2}")

            case (ChannelName.SCH_HD_CHANNEL, ChannelName.BNCH_CHANNEL):
                # SCH/HD + BNCH on TP FN:18 or CP FN:[1,18]
                if self.phyChannel == PhyType.TRAFFIC_CHANNEL:
                    if self.frameNumber != CONTROL_FRAME_NUMBER:
                        raise ValueError(
                            f"For {type(self).__name__}, FN {self.frameNumber} invalid for SCH/HD+BNCH on TP "
                            f"(expected control frame {CONTROL_FRAME_NUMBER})."
                        )
                    if ((self.multiFrameNumber + self.timeSlot) % 4) != 1:
                        raise ValueError(
                            f"For {type(self).__name__}, (MN+TN)%4 != 1 for SCH/HD+BNCH on TP "
                            f"(MN={self.multiFrameNumber}, TN={self.timeSlot})."
                        )
                elif self.phyChannel == PhyType.CONTROL_CHANNEL:
                    if not (1 <= self.frameNumber <= MULTIFRAME_TDMAFRAME_LENGTH):
                        raise ValueError(
                            f"For {type(self).__name__}, FN {self.frameNumber} invalid for SCH/HD+BNCH on CP "
                            f"(expected 1..{MULTIFRAME_TDMAFRAME_LENGTH})."
                        )
                    if self.frameNumber == CONTROL_FRAME_NUMBER:
                        if ((self.multiFrameNumber + self.timeSlot) % 4) != 1:
                            raise ValueError(
                                f"For {type(self).__name__}, (MN+TN)%4 != 1 for SCH/HD+BNCH on CP control frame "
                                f"(MN={self.multiFrameNumber}, TN={self.timeSlot})."
                            )
                else:
                    raise ValueError(f"For {type(self).__name__}, phy {self.phyChannel} invalid for bkn1:{bkn1} bkn2:{bkn2}")
                return (BurstContent.BURST_CONTROL_TYPE, True, 1)

            case (ChannelName.SCH_HD_CHANNEL, ChannelName.BLCH_CHANNEL):
                # SCH/HD + BLCH on TP FN:18 or CP/UP FN:[1,18]
                if self.phyChannel == PhyType.TRAFFIC_CHANNEL:
                    if self.frameNumber != CONTROL_FRAME_NUMBER:
                        raise ValueError(
                            f"For {type(self).__name__}, FN {self.frameNumber} invalid for SCH/HD+BLCH on TP "
                            f"(expected control frame {CONTROL_FRAME_NUMBER})."
                        )
                elif self.phyChannel in (PhyType.CONTROL_CHANNEL, PhyType.UNASGN_CHANNEL):
                    if not (1 <= self.frameNumber <= MULTIFRAME_TDMAFRAME_LENGTH):
                        raise ValueError(
                            f"For {type(self).__name__}, FN {self.frameNumber} invalid for SCH/HD+BLCH on "
                            f"{self.phyChannel} (expected 1..{MULTIFRAME_TDMAFRAME_LENGTH})."
                        )
                else:
                    raise ValueError(f"For {type(self).__name__}, phy {self.phyChannel} invalid for bkn1:{bkn1} bkn2:{bkn2}")
                return (BurstContent.BURST_CONTROL_TYPE, True, 1)

            case (ChannelName.SCH_HD_CHANNEL, _):
                if bkn2 == ChannelName.SCH_HD_CHANNEL:
                    # SCH/HD + SCH/HD on TP FN:18 or CP/UP FN:[1,18]
                    if self.phyChannel == PhyType.TRAFFIC_CHANNEL:
                        if self.frameNumber != CONTROL_FRAME_NUMBER:
                            raise ValueError(
                                f"For {type(self).__name__}, FN {self.frameNumber} invalid for SCH/HD+SCH/HD on TP "
                                f"(expected control frame {CONTROL_FRAME_NUMBER})."
                            )
                    elif self.phyChannel in (PhyType.CONTROL_CHANNEL, PhyType.UNASGN_CHANNEL):
                        if not (1 <= self.frameNumber <= MULTIFRAME_TDMAFRAME_LENGTH):
                            raise ValueError(
                                f"For {type(self).__name__}, FN {self.frameNumber} invalid for SCH/HD+SCH/HD on "
                                f"{self.phyChannel} (expected 1..{MULTIFRAME_TDMAFRAME_LENGTH})."
                            )
                    else:
                        raise ValueError(f"For {type(self).__name__}, phy {self.phyChannel} invalid for bkn1:{bkn1} bkn2:{bkn2}")
                    return (BurstContent.BURST_CONTROL_TYPE, True, 1)

                raise ValueError(f"For {type(self).__name__}, invalid combination bkn1:{bkn1} bkn2:{bkn2}")

            case _:
                raise ValueError(f"For {type(self).__name__}, invalid combination of {bkn1} and {bkn2}")

###################################################################################################

class Normal_Cont_Downlink_Burst(NormalDownlinkMixin, Burst):
    SNmax = 255
    startGuardBitPeriod = 0
    endGuardBitPeriod = 0
    subSlotWidth = 2
    linkDirection = LinkDirection.DOWNLINK
    burstType = BurstContent.BURST_TRAFFIC_TYPE

    ALLOWED_PHY = {PhyType.CONTROL_CHANNEL, PhyType.TRAFFIC_CHANNEL, PhyType.UNASGN_CHANNEL}

    def constructBurstBitSequence(self, inputLogicalChannelBkn1:Union[TrafficChannel, SCH_F, STCH, SCH_HD],
                                        inputLogicalChannelBbk:AACH,
                                        inputLogicalChannelBkn2:Union[TrafficChannel, BLCH, BNCH, STCH, SCH_HD, None]=None,
                                        rampUpandDown:Tuple[bool,bool]=(False, False)):
        

        # If bkn1 is control, we care about the specific channel type, if it is traffic we dont care for bkn1
        bkn1 = self._norm_bkin(inputLogicalChannelBkn1)
        bkn2 = None if inputLogicalChannelBkn2 is None else self._norm_bkin(inputLogicalChannelBkn2)

        assert inputLogicalChannelBbk.channel == ChannelName.AACH_CHANNEL

        burstType, multiLogicalChannelState, tsi = self._validate_normal_downlink_mapping(bkn1, bkn2)
        
        self.burstType = burstType
        self.mixedBurst = multiLogicalChannelState
        trainingSequenceIndex = tsi

        # Build the burst
        burstBitSequence = empty(shape=(self.SNmax*2), dtype=uint8)
        
        if rampUpandDown[0]:
            # if we are ramp up (TRUE), it means that this is the first burst, 
            burstBitSequence[:12] = zeros(shape=12, dtype=uint8)
            self.burstStartRampPeriod = 12
        else:
            # other we are continuous (or we are ramping down add preceding bits per 9.4.5.1 - Table 28)
            burstBitSequence[:12] = NORMAL_TRAINING_SEQUENCE[2][10:22]
            self.burstStartRampPeriod = 0

        # temporarily skip phase adjustment bits a - [12:14]
        burstBitSequence[14:230] = inputLogicalChannelBkn1.type5Blocks[0][:216]
        burstBitSequence[230:244] = inputLogicalChannelBbk.type5Blocks[0][:14]
        burstBitSequence[244:266] = NORMAL_TRAINING_SEQUENCE[trainingSequenceIndex][:22]
        burstBitSequence[266:282] = inputLogicalChannelBbk.type5Blocks[0][14:30]
        if inputLogicalChannelBkn2 is not None and self.mixedBurst:
            burstBitSequence[282:498] = inputLogicalChannelBkn2.type5Blocks[0][:216]
        else:
            burstBitSequence[282:498] = inputLogicalChannelBkn1.type5Blocks[0][216:432]
        # temporarily skip phase adjustment bits b - [498:500]
        if rampUpandDown[1]:
            # if we are ramp down (TRUE), it means that this is the last burst we are ramping down 
            burstBitSequence[500:510] = zeros(shape=10, dtype=uint8)
            self.burstEndRampPeriod = 10
        else:
            # otherwise we are continuous (or we are have ramped up add preceding bits per 9.4.5.1 - Table 27)
            burstBitSequence[500:510] = NORMAL_TRAINING_SEQUENCE[2][0:10]
            self.burstEndRampPeriod = 0

        # Now insert phase adjustment bits
        burstBitSequence[12:14] = calculatePhaseAdjustmentBits(burstBitSequence, PHASE_ADJUSTMENT_SYMBOL_RANGE['a'], 0)
        burstBitSequence[498:500] = calculatePhaseAdjustmentBits(burstBitSequence, PHASE_ADJUSTMENT_SYMBOL_RANGE['b'], 0)

        return burstBitSequence
    
    def deconstructBurstBitSequence(self, inputData:List[int]):
        raise NotImplementedError

###################################################################################################

class Normal_Discont_Downlink_Burst(NormalDownlinkMixin, Burst):
    SNmax = 246
    startGuardBitPeriod = 10
    endGuardBitPeriod = 8
    subSlotWidth = 2
    linkDirection = LinkDirection.DOWNLINK
    burstType = BurstContent.BURST_TRAFFIC_TYPE

    ALLOWED_PHY = {PhyType.CONTROL_CHANNEL, PhyType.TRAFFIC_CHANNEL, PhyType.UNASGN_CHANNEL}

    def constructBurstBitSequence(self, inputLogicalChannelBkn1:Union[TrafficChannel, SCH_F, STCH, SCH_HD],
                                        inputLogicalChannelBbk:AACH,
                                        inputLogicalChannelBkn2:Union[TrafficChannel, BLCH, BNCH, STCH, SCH_HD, None]=None,
                                        rampUpandDown:Tuple[bool,bool]=(True,True)):
        
        # If bkn1 is control, we care about the specific channel type, if it is traffic we dont care for bkn1
        bkn1 = self._norm_bkin(inputLogicalChannelBkn1)
        bkn2 = None if inputLogicalChannelBkn2 is None else self._norm_bkin(inputLogicalChannelBkn2)

        assert inputLogicalChannelBbk.channel == ChannelName.AACH_CHANNEL

        burstType, multiLogicalChannelState, tsi = self._validate_normal_downlink_mapping(bkn1, bkn2)
        
        self.burstType = burstType
        self.mixedBurst = multiLogicalChannelState
        trainingSequenceIndex = tsi
        
        # Build the burst
        d = self.startGuardBitPeriod
        burstBitSequence = empty(shape=(self.SNmax*2)+d+self.endGuardBitPeriod, dtype=uint8)

        burstBitSequence[d:2+d] = NORMAL_TRAINING_SEQUENCE[2][20:22]
        # temporarily skip phase adjustment bits g - [d+2:4+d]
        burstBitSequence[d+4:220+d] = inputLogicalChannelBkn1.type5Blocks[0][:216]
        burstBitSequence[d+220:234+d] = inputLogicalChannelBbk.type5Blocks[0][:14]
        burstBitSequence[d+234:256+d] = NORMAL_TRAINING_SEQUENCE[trainingSequenceIndex][:22]
        burstBitSequence[d+256:272+d] = inputLogicalChannelBbk.type5Blocks[0][14:30]
        if inputLogicalChannelBkn2 is not None and self.mixedBurst:
            burstBitSequence[d+272:488+d] = inputLogicalChannelBkn2.type5Blocks[0][:216]
        else:
            burstBitSequence[d+272:488+d] = inputLogicalChannelBkn1.type5Blocks[0][216:432]
        # temporarily skip phase adjustment bits h - [d+488:490+d]
        burstBitSequence[d+490:492+d] = NORMAL_TRAINING_SEQUENCE[2][:2]
        # Now insert phase adjustment bits
        burstBitSequence[d+2:4+d] = calculatePhaseAdjustmentBits(burstBitSequence, PHASE_ADJUSTMENT_SYMBOL_RANGE['g'], d)
        burstBitSequence[d+488:490+d] = calculatePhaseAdjustmentBits(burstBitSequence, PHASE_ADJUSTMENT_SYMBOL_RANGE['h'], d)
        # must add guard period training sequence if there is no ramping at the start
        if rampUpandDown[0]:
            burstBitSequence[:d] = zeros(shape=self.startGuardBitPeriod, dtype=uint8)
            self.burstStartRampPeriod = d
        else:
            # add preceding bits per 9.4.5.2
            burstBitSequence[:d] = NORMAL_TRAINING_SEQUENCE[2][10:20]
            self.burstStartRampPeriod = 0
        # must add guard period training sequence if there is no ramping at the end
        if rampUpandDown[1]:
            burstBitSequence[492+d:510] = zeros(shape=self.endGuardBitPeriod, dtype=uint8)
            self.burstEndRampPeriod = TIMESLOT_BIT_LENGTH - 492+d
        else:
            # add following bits per 9.4.5.2
            burstBitSequence[492+d:510] = NORMAL_TRAINING_SEQUENCE[2][2:10]
            self.burstEndRampPeriod = 0

        return burstBitSequence
    
    def deconstructBurstBitSequence(self):
        raise NotImplementedError

###################################################################################################

class SynchronousDownlinkMixin:

    def _validate_normal_downlink_mapping(self:DownlinkHost, bkn1: str, 
                                          bkn2: Optional[str]) -> Tuple[BurstContent, bool]:
        """
        Validates (bkn1, bkn2) for synchronous downlink burst usage across CP/TP/UP
        and returns:
            (burstType: BurstContent, mixedBurst: bool, trainingSequenceIndex: int)
        """

        if bkn1 not in (ChannelName.BSCH_CHANNEL, ChannelName.SCH_HD_CHANNEL):
            raise ValueError(f"For {type(self).__name__}, invalid bkn1:{bkn1} for synchronous downlink")
        if bkn2 not in (ChannelName.SCH_HD_CHANNEL, ChannelName.BNCH_CHANNEL, ChannelName.BLCH_CHANNEL):
            raise ValueError(f"For {type(self).__name__}, invalid bkn2:{bkn2} for synchronous downlink")
        
        burst_type = BurstContent.BURST_CONTROL_TYPE
        mixed = True

        match (bkn1, bkn2):
            case (ChannelName.BSCH_CHANNEL, ChannelName.SCH_HD_CHANNEL):
                # BSCH in BKN1, SCH/HD (or BLCH replacing SCH/HD) in BKN2
                # Valid on TP or CP, with control-frame timing and (MN+TN)%4==3
                if self.phyChannel in (PhyType.TRAFFIC_CHANNEL, PhyType.CONTROL_CHANNEL):
                    if self.frameNumber != CONTROL_FRAME_NUMBER:
                        raise ValueError(
                            f"For {type(self).__name__}, FN {self.frameNumber} invalid for phy {self.phyChannel} "
                            f"bkn1:{bkn1} bkn2:{bkn2} (expected control frame)"
                        )
                    if ((self.multiFrameNumber + self.timeSlot) % 4) != 3:
                        raise ValueError(
                            f"For {type(self).__name__}, (MN+TN)%4 != 3 for bkn1:{bkn1} bkn2:{bkn2} "
                            f"(MN={self.multiFrameNumber}, TN={self.timeSlot})"
                        )
                else:
                    raise ValueError(f"For {type(self).__name__}, phy {self.phyChannel} invalid for bkn1:{bkn1} bkn2:{bkn2}")
                return (burst_type, mixed)
            case (ChannelName.BSCH_CHANNEL, ChannelName.BLCH_CHANNEL):
                # BSCH in BKN1, SCH/HD (or BLCH replacing SCH/HD) in BKN2
                # Valid on TP or CP, with control-frame timing and (MN+TN)%4==3
                if self.phyChannel in (PhyType.TRAFFIC_CHANNEL, PhyType.CONTROL_CHANNEL):
                    if self.frameNumber != CONTROL_FRAME_NUMBER:
                        raise ValueError(
                            f"For {type(self).__name__}, FN {self.frameNumber} invalid for phy {self.phyChannel} "
                            f"bkn1:{bkn1} bkn2:{bkn2} (expected control frame)"
                        )
                    if ((self.multiFrameNumber + self.timeSlot) % 4) != 3:
                        raise ValueError(
                            f"For {type(self).__name__}, (MN+TN)%4 != 3 for bkn1:{bkn1} bkn2:{bkn2} "
                            f"(MN={self.multiFrameNumber}, TN={self.timeSlot})"
                        )
                else:
                    raise ValueError(f"For {type(self).__name__}, phy {self.phyChannel} invalid for bkn1:{bkn1} bkn2:{bkn2}")
                return (burst_type, mixed)

            case (ChannelName.BSCH_CHANNEL, ChannelName.BNCH_CHANNEL):
                # BSCH in BKN1, BNCH in BKN2: only permitted on UP, FN:[1..18]
                if self.phyChannel != PhyType.UNASGN_CHANNEL:
                    raise ValueError(
                        f"For {type(self).__name__}, phy {self.phyChannel} invalid for bkn1:{bkn1} bkn2:{bkn2} "
                        f"(expected UNASGN/UP)"
                    )
                if not (1 <= self.frameNumber <= MULTIFRAME_TDMAFRAME_LENGTH):
                    raise ValueError(f"For {type(self).__name__}, FN {self.frameNumber} invalid for UP bkn1:{bkn1} bkn2:{bkn2}")
                return (burst_type, mixed)

            case (ChannelName.SCH_HD_CHANNEL, ChannelName.BNCH_CHANNEL):
                # SCH/HD in BKN1, BNCH in BKN2:
                # - on TP: FN == control frame and (MN+TN)%4==1
                # - on CP: FN:[1..18], and if FN==control frame then (MN+TN)%4==1
                if self.phyChannel == PhyType.TRAFFIC_CHANNEL:
                    if self.frameNumber != CONTROL_FRAME_NUMBER:
                        raise ValueError(f"For {type(self).__name__}, FN {self.frameNumber} invalid for TP bkn1:{bkn1} bkn2:{bkn2}")
                    if ((self.multiFrameNumber + self.timeSlot) % 4) != 1:
                        raise ValueError(
                            f"For {type(self).__name__}, (MN+TN)%4 != 1 for TP bkn1:{bkn1} bkn2:{bkn2} "
                            f"(MN={self.multiFrameNumber}, TN={self.timeSlot})"
                        )
                elif self.phyChannel == PhyType.CONTROL_CHANNEL:
                    if not (1 <= self.frameNumber <= MULTIFRAME_TDMAFRAME_LENGTH):
                        raise ValueError(f"For {type(self).__name__}, FN {self.frameNumber} invalid for CP bkn1:{bkn1} bkn2:{bkn2}")
                    if self.frameNumber == CONTROL_FRAME_NUMBER:
                        if ((self.multiFrameNumber + self.timeSlot) % 4) != 1:
                            raise ValueError(
                                f"For {type(self).__name__}, (MN+TN)%4 != 1 for CP control frame "
                                f"(MN={self.multiFrameNumber}, TN={self.timeSlot})"
                            )
                else:
                    raise ValueError(f"For {type(self).__name__}, phy {self.phyChannel} invalid for bkn1:{bkn1} bkn2:{bkn2}")
                return (burst_type, mixed)

            case (ChannelName.SCH_HD_CHANNEL, ChannelName.BLCH_CHANNEL):
                # SCH/HD in BKN1, and BKN2 is SCH/HD (or BLCH replacing it):
                # - on TP: FN == control frame
                # - on CP or UP: FN:[1..18]
                if self.phyChannel == PhyType.TRAFFIC_CHANNEL:
                    if self.frameNumber != CONTROL_FRAME_NUMBER:
                        raise ValueError(f"For {type(self).__name__}, FN {self.frameNumber} invalid for TP bkn1:{bkn1} bkn2:{bkn2}")
                elif self.phyChannel in (PhyType.CONTROL_CHANNEL, PhyType.UNASGN_CHANNEL):
                    if not (1 <= self.frameNumber <= MULTIFRAME_TDMAFRAME_LENGTH):
                        raise ValueError(
                            f"For {type(self).__name__}, FN {self.frameNumber} invalid for phy {self.phyChannel} "
                            f"bkn1:{bkn1} bkn2:{bkn2}"
                        )
                else:
                    raise ValueError(f"For {type(self).__name__}, phy {self.phyChannel} invalid for bkn1:{bkn1} bkn2:{bkn2}")
                return (burst_type, mixed)
            case (ChannelName.SCH_HD_CHANNEL, _):
                if bkn2 == ChannelName.SCH_HD_CHANNEL:
                    # SCH/HD in BKN1, and BKN2 is SCH/HD (or BLCH replacing it):
                    # - on TP: FN == control frame
                    # - on CP or UP: FN:[1..18]
                    if self.phyChannel == PhyType.TRAFFIC_CHANNEL:
                        if self.frameNumber != CONTROL_FRAME_NUMBER:
                            raise ValueError(f"For {type(self).__name__}, FN {self.frameNumber} invalid for TP bkn1:{bkn1} bkn2:{bkn2}")
                    elif self.phyChannel in (PhyType.CONTROL_CHANNEL, PhyType.UNASGN_CHANNEL):
                        if not (1 <= self.frameNumber <= MULTIFRAME_TDMAFRAME_LENGTH):
                            raise ValueError(
                                f"For {type(self).__name__}, FN {self.frameNumber} invalid for phy {self.phyChannel} "
                                f"bkn1:{bkn1} bkn2:{bkn2}"
                            )
                    else:
                        raise ValueError(f"For {type(self).__name__}, phy {self.phyChannel} invalid for bkn1:{bkn1} bkn2:{bkn2}")
                    return (burst_type, mixed)
                else:
                        raise ValueError(f"For {type(self).__name__}, phy {self.phyChannel} invalid for bkn1:{bkn1} bkn2:{bkn2}")
            case _:
                raise ValueError(f"For {type(self).__name__}, invalid combination bkn1:{bkn1} bkn2:{bkn2}")

###################################################################################################

class Sync_Cont_Downlink_Burst(SynchronousDownlinkMixin, Burst):
    SNmax = 255
    startGuardBitPeriod = 0
    endGuardBitPeriod = 0
    subSlotWidth = 2
    linkDirection = LinkDirection.DOWNLINK
    burstType = BurstContent.BURST_CONTROL_TYPE

    ALLOWED_PHY = {PhyType.CONTROL_CHANNEL, PhyType.TRAFFIC_CHANNEL, PhyType.UNASGN_CHANNEL}

    def constructBurstBitSequence(self, inputLogicalChannelSb:Union[BSCH, SCH_HD],
                                        inputLogicalChannelBbk:AACH,
                                        inputLogicalChannelBkn2:Union[BNCH, BLCH, SCH_HD],
                                        rampUpandDown:Tuple[bool,bool]=(False, False)):
        
        assert inputLogicalChannelSb.channel in (ChannelName.BSCH_CHANNEL, ChannelName.SCH_HD_CHANNEL)
        assert inputLogicalChannelBkn2.channel in (ChannelName.SCH_HD_CHANNEL, ChannelName.BNCH_CHANNEL, ChannelName.BLCH_CHANNEL)
        assert inputLogicalChannelBbk.channel == ChannelName.AACH_CHANNEL

        bkn1 = inputLogicalChannelSb.channel
        bkn2 = inputLogicalChannelBkn2.channel
        
        _, multiLogicalChannelState = self._validate_normal_downlink_mapping(bkn1,bkn2)
        self.mixedBurst = multiLogicalChannelState

        # Build the burst
        burstBitSequence = empty(shape=(self.SNmax*2), dtype=uint8)

        if rampUpandDown[0]:
            # if we are ramp up (TRUE), it means that this is the first burst, 
            burstBitSequence[:12] = zeros(shape=self.startGuardBitPeriod, dtype=uint8)
            self.burstStartRampPeriod = 12
        else:
            # other we are continuous (or we are ramping down add preceding bits per 9.4.5.1 - Table 28)
            burstBitSequence[:12] = NORMAL_TRAINING_SEQUENCE[2][10:22]
            self.burstStartRampPeriod = 0

        # temporarily skip phase adjustment bits C - [12:14]
        burstBitSequence[14:94] = FREQUENCY_CORRECTION_FIELD
        burstBitSequence[94:214] = inputLogicalChannelSb.type5Blocks[0][:120]
        burstBitSequence[214:252] = SYNCHRONIZATION_TRAINING_SEQUENCE
        burstBitSequence[252:282] = inputLogicalChannelBbk.type5Blocks[0]
        burstBitSequence[282:498] = inputLogicalChannelBkn2.type5Blocks[0][:216] # type: ignore[attr-defined]
        # temporarily skip phase adjustment bits D - [498:500]
        if rampUpandDown[1]:
            # if we are ramp down (TRUE), it means that this is the last burst we are ramping down 
            burstBitSequence[500:510] = zeros(shape=self.endGuardBitPeriod, dtype=uint8)
            self.burstEndRampPeriod = 10
        else:
            # otherwise we are continuous (or we are have ramped up add preceding bits per 9.4.5.1 - Table 27)
            burstBitSequence[500:510] = NORMAL_TRAINING_SEQUENCE[2][0:10]
            self.burstEndRampPeriod = 0

        # Now insert phase adjustment bits
        burstBitSequence[12:14] = calculatePhaseAdjustmentBits(burstBitSequence, PHASE_ADJUSTMENT_SYMBOL_RANGE['c'],0)
        burstBitSequence[498:500] = calculatePhaseAdjustmentBits(burstBitSequence, PHASE_ADJUSTMENT_SYMBOL_RANGE['d'],0)

        return burstBitSequence
    
    def deconstructBurstBitSequence(self):
        raise NotImplementedError
    
###################################################################################################

class Sync_Discont_Downlink_Burst(SynchronousDownlinkMixin, Burst):
    SNmax = 246
    startGuardBitPeriod = 10
    endGuardBitPeriod = 8
    subSlotWidth = 2
    linkDirection = LinkDirection.DOWNLINK
    burstType = BurstContent.BURST_CONTROL_TYPE

    ALLOWED_PHY = {PhyType.CONTROL_CHANNEL, PhyType.TRAFFIC_CHANNEL, PhyType.UNASGN_CHANNEL}

    def constructBurstBitSequence(self, inputLogicalChannelSb:Union[BSCH, SCH_HD],
                                        inputLogicalChannelBbk:AACH,
                                        inputLogicalChannelBkn2:Union[BNCH, BLCH, SCH_HD],
                                        rampUpandDown:Tuple[bool,bool]=(True, True)):
        
        assert inputLogicalChannelSb.channel in (ChannelName.BSCH_CHANNEL, ChannelName.SCH_HD_CHANNEL)
        assert inputLogicalChannelBkn2.channel in (ChannelName.SCH_HD_CHANNEL, ChannelName.BNCH_CHANNEL, ChannelName.BLCH_CHANNEL)
        assert inputLogicalChannelBbk.channel == ChannelName.AACH_CHANNEL

        bkn1 = inputLogicalChannelSb.channel
        bkn2 = inputLogicalChannelBkn2.channel

        _, multiLogicalChannelState = self._validate_normal_downlink_mapping(bkn1,bkn2)
        self.mixedBurst = multiLogicalChannelState

        # Build the burst
        d = self.startGuardBitPeriod
        burstBitSequence = empty(shape=(self.SNmax*2)+d+self.endGuardBitPeriod, dtype=uint8)

        burstBitSequence[d:2+d] = NORMAL_TRAINING_SEQUENCE[2][20:22]
        # temporarily skip phase adjustment bits i - [d+2:4+d]
        burstBitSequence[d+4:84+d] = FREQUENCY_CORRECTION_FIELD
        burstBitSequence[d+84:204+d] = inputLogicalChannelSb.type5Blocks[0][:120]
        burstBitSequence[d+204:242+d] = SYNCHRONIZATION_TRAINING_SEQUENCE
        burstBitSequence[d+242:272+d] = inputLogicalChannelBbk.type5Blocks[0][:30]
        burstBitSequence[d+272:488+d] = inputLogicalChannelBkn2.type5Blocks[0][:216]
        # temporarily skip phase adjustment bits j - [d+488:490+d]
        burstBitSequence[d+490:492+d] = NORMAL_TRAINING_SEQUENCE[2][:2]

        # Now insert phase adjustment bits
        burstBitSequence[d+2:4+d] = calculatePhaseAdjustmentBits(burstBitSequence, PHASE_ADJUSTMENT_SYMBOL_RANGE['i'], d)
        burstBitSequence[d+488:490+d] = calculatePhaseAdjustmentBits(burstBitSequence, PHASE_ADJUSTMENT_SYMBOL_RANGE['j'], d)
        
        # must add guard period training sequence if there is no ramping at the start
        if rampUpandDown[0]:
            burstBitSequence[:d] = zeros(shape=self.startGuardBitPeriod, dtype=uint8)
            self.burstStartRampPeriod = d
        else:
            # add preceding bits per 9.4.5.2
            burstBitSequence[:d] = NORMAL_TRAINING_SEQUENCE[2][10:20]
            self.burstStartRampPeriod = 0
        
        # must add guard period training sequence if there is no ramping at the end
        if rampUpandDown[1]:
            burstBitSequence[492+d:510] = zeros(shape=self.endGuardBitPeriod, dtype=uint8)
            self.burstEndRampPeriod = TIMESLOT_BIT_LENGTH - (492+d)
        else:
            # add following bits per 9.4.5.2
            burstBitSequence[492+d:510] = NORMAL_TRAINING_SEQUENCE[2][2:10]
            self.burstEndRampPeriod = 0
        
        return burstBitSequence
    
    def deconstructBurstBitSequence(self):
        raise NotImplementedError

###################################################################################################

class Linearization_Uplink_Burst(Burst):
    SNmax = 240
    startGuardBitPeriod = 0 # taken from Table 7, which is 119 symbols till SN0 (SN1 is 15 guard bits)
    endGuardBitPeriod = 15
    subSlotWidth = 1
    linkDirection = LinkDirection.UPLINK
    burstType = BurstContent.BURST_LINEARIZATION_TYPE

    ALLOWED_PHY = {PhyType.CONTROL_CHANNEL, PhyType.TRAFFIC_CHANNEL, PhyType.UNASGN_CHANNEL}

    def constructBurstBitSequence(self, inputLogicalChannelSsn1:CLCH):
        if self.phyChannel == PhyType.TRAFFIC_CHANNEL or self.phyChannel == PhyType.CONTROL_CHANNEL:
            if self.frameNumber != CONTROL_FRAME_NUMBER:
                raise ValueError(f"For {type(self).__name__}, phy {self.phyChannel}, FN {self.frameNumber} invalid for ssn{self.subSlot}: {inputLogicalChannelSsn1.channel}")
            
            if ((self.multiFrameNumber + self.timeSlot) % 4 != 3):
                raise ValueError (f"For {type(self).__name__}, (MN+TN)%4 != 3 for CP control frame "
                                f"(MN={self.multiFrameNumber}, TN={self.timeSlot})") #per 9.5.2 (74)
        
        if self.subSlot != 1:
            raise ValueError(f" For {type(self).__name__}, subslot {self.subSlot} is invalid, expected (1)")

        if inputLogicalChannelSsn1.channel != ChannelName.CLCH_CHANNEL:
            raise ValueError(f"For {type(self).__name__}, phy {self.phyChannel}, invalid ssn of {inputLogicalChannelSsn1.channel}")

        # Build the burst
        d = self.startGuardBitPeriod
        burstBitSequence = empty(shape=(self.SNmax*2)+d+self.endGuardBitPeriod, dtype=uint8)
        burstBitSequence[0:d] = zeros(shape=self.startGuardBitPeriod, dtype=uint8)
        burstBitSequence[d:238+d] = inputLogicalChannelSsn1.type5Blocks[0][:238]
        # End guard bits
        burstBitSequence[d+238:255] = zeros(shape=self.endGuardBitPeriod, dtype=uint8) 

        self.burstStartRampPeriod = SUBSLOT_BIT_LENGTH-self.endGuardBitPeriod
        self.burstEndRampPeriod = 15

        return burstBitSequence

    def deconstructBurstBitSequence(self):
        raise NotImplementedError

###################################################################################################

class Null_Halfslot_Uplink_Burst(Burst):
    SNmax = 255
    startGuardBitPeriod = 0
    endGuardBitPeriod = 0
    subSlotWidth = 1
    linkDirection = LinkDirection.UPLINK
    burstType = BurstContent.BURST_MIXED_TYPE

    ALLOWED_PHY = {PhyType.CONTROL_CHANNEL, PhyType.TRAFFIC_CHANNEL, PhyType.UNASGN_CHANNEL}
    
    def constructBurstBitSequence(self):
        return zeros(shape=self.SNmax, dtype=uint8)
    
    def deconstructBurstBitSequence(self):
        return zeros(shape=self.SNmax, dtype=uint8)
    

###################################################################################################
