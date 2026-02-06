from enum import Enum

######## PHYSICAL CHANNEL CONSTANTS ##########
# Enum use to wrap the valid types of physical channel
class PhyType(str, Enum):
    TRAFFIC_CHANNEL = "TP"
    CONTROL_CHANNEL = "CP"
    UNASGN_CHANNEL = "UP"

# Enum used to wrap the valid link directions for physical channels
class LinkDirection(str, Enum):
    """LinkDirection 
    """
    UPLINK = "UL"
    DOWNLINK = "DL"
#############################################

############## BURST CONSTANTS ##############
# Enum used to wrap a description for what kind of burst
class BurstContent(str, Enum):
    BURST_TRAFFIC_TYPE = "traffic"
    BURST_CONTROL_TYPE = "control"
    BURST_MIXED_TYPE = "mixed"
    BURST_LINEARIZATION_TYPE = "linear"

class SlotLength(str, Enum):
    HALF_SUBSLOT = "half"
    FULL_SUBSLOT = "full"
#############################################

######## LOGICAL CHANNEL CONSTANTS ##########
class ChannelKind(str, Enum):
    TRAFFIC_TYPE = "traffic"
    CONTROL_TYPE = "control"
    LINEARIZATION_TYPE = "linear"

# Channel variable names for MAC burst building
class ChannelName(str, Enum):
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

SUBSLOT_BIT_LENGTH = 255
TIMESLOT_SYMBOL_LENGTH = 255

#############################################

########### MAC LAYER CONSTANTS #############
# Control Modes
class ControlMode(str, Enum):
    NCM_CTRL_MODE = "NCM"       # Normal Control Mode
    MC_CTRL_MODE = "MC"         # Minimum Control Mode

# Transmission Modes
class TransmissionMode(str, Enum):
    D_CT_BS_MODE = "D-CT"       # Downlink-Continuous Transmission (D-CT)
    D_CTT_BS_MODE = "D-CTT"     # Downlink-Carrier Timesharing Transmission (D-CTT)
    D_MCCTT_BS_MODE = "D-MCCTT" # Downlink-Main Control Channel Timesharing Transmission (D-MCCTT)
    U_MST_BS_MODE = "U-MST"     # Multiple Slot Transmission

#############################################

########## TRANSCIEVER CONSTANTS ############
BASEBAND_SAMPLING_FACTOR = 64 # The culmative oversampling factor of bb procesing

#############################################
