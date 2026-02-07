"""
constants.py holds constains that are used across modules, including enum string types that define logical and physical
channel behaviours as well as MAC layer operation types. For modules specific constants they are included within modules
as needed or otherwise colated into related utility files.
"""
from enum import Enum

#############################################
# PHYSICAL CHANNEL CONSTANTS


class PhyType(str, Enum):
    """
    Enum class used to wrap the valid types of physical channel for a physical channel object
    """
    TRAFFIC_CHANNEL = "TP"
    CONTROL_CHANNEL = "CP"
    UNASGN_CHANNEL = "UP"


class LinkDirection(str, Enum):
    """
    Enum class used to wrap the valid link directions for a physical channels object
    """
    UPLINK = "UL"
    DOWNLINK = "DL"

#############################################
# BURST CONSTANTS


class BurstContent(str, Enum):
    """
    Enum class used to wrap a descriptor for the content of a burst, used for a physical layer burst object
    """
    BURST_TRAFFIC_TYPE = "traffic"
    BURST_CONTROL_TYPE = "control"
    BURST_MIXED_TYPE = "mixed"
    BURST_LINEARIZATION_TYPE = "linear"


class SlotLength(str, Enum):
    """
    Enum class used to wrap a the a descriptor of wether or not a physical layer burst object uses an entire slot
    of half, such as in the case of uplink control and linearization bursts.
    """
    HALF_SUBSLOT = "half"
    FULL_SUBSLOT = "full"

#############################################
# LOGICAL CHANNEL CONSTANTS


class ChannelKind(str, Enum):
    """
    Enum class used to wrap the valid types of logical channel for a logical channel object
    """
    TRAFFIC_TYPE = "traffic"
    CONTROL_TYPE = "control"
    LINEARIZATION_TYPE = "linear"


# Channel variable names for MAC burst building
class ChannelName(str, Enum):
    """
    Enum class used to wrap the valid names of logical channels for a logical channel object used in verifying burst
    building to prevent invalid combinations
    """
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


SUBSLOT_BIT_LENGTH = 255        # The number of modulation bits in a subslot
TIMESLOT_SYMBOL_LENGTH = 255    # The number of symbols in a full slot

#############################################
# MAC LAYER CONSTANTS


# Control Modes
class ControlMode(str, Enum):
    """
    Enum class used to wrap the valid types of control modes for BS and MS MAC layer control plane signalling
    """
    NCM_CTRL_MODE = "NCM"       # Normal Control Mode
    MC_CTRL_MODE = "MC"         # Minimum Control Mode


# Transmission Modes
class TransmissionMode(str, Enum):
    """
    Enum class used to wrap the valid types of transmission modes for BS and MS MAC layer traffic plane signalling
    """
    D_CT_BS_MODE = "D-CT"        # Downlink-Continuous Transmission (D-CT)
    D_CTT_BS_MODE = "D-CTT"      # Downlink-Carrier Timesharing Transmission (D-CTT)
    D_MCCTT_BS_MODE = "D-MCCTT"  # Downlink-Main Control Channel Timesharing Transmission (D-MCCTT)
    U_MST_BS_MODE = "U-MST"      # Multiple Slot Transmission

#############################################
# TRANSCIEVER CONSTANTS


TX_BB_SAMPLING_FACTOR = 64  # The culmative oversampling factor of the tx baseband processing

#############################################
